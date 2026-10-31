"""NLP Project Update 1: continued fine-tuning and false-positive evaluation.

Author: Jaida Calvin (course project; implementation prepared with AI assistance).
No experimental outputs are bundled. Run in a fresh Google Colab GPU runtime.
Raw dataset text and model weights are intentionally excluded from exports.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import random
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

MODEL_IDS = {
    "dehatebert": "Hate-speech-CNERG/dehatebert-mono-english",
    "roberta": "cardiffnlp/twitter-roberta-base-hate-latest",
}
DEFAULTS = {
    "seed": 42, "epochs": 2, "learning_rate": 2e-5,
    "batch_size": 8, "eval_batch_size": 16, "gradient_accumulation": 2,
    "max_length": 128, "weight_decay": 0.01, "warmup_ratio": 0.1,
}
LABEL_MAP = {"normal": 0, "offensive": 0, "hatespeech": 1}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def normal_key(text: str) -> str:
    """For duplicate checks only, never a replacement for the model's input."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def majority_label(annotators: list[dict[str, Any]]) -> str | None:
    labels = [str(a["label"]).strip().lower() for a in annotators]
    if not labels or any(x not in LABEL_MAP for x in labels):
        raise ValueError("Missing or unsupported original HateXplain label.")
    label, count = Counter(labels).most_common(1)[0]
    # Resolve the ORIGINAL three-way vote before applying the binary mapping.
    return label if count > len(labels) / 2 else None


def binary_metrics(y_true: Any, scores: Any, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and scores must be nonempty equal-length vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Invalid labels or non-finite scores.")
    if np.any(p < 0) or np.any(p > 1) or not 0 <= threshold <= 1:
        raise ValueError("Scores and threshold must lie in [0, 1].")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return {
        "n": len(y), "threshold": float(threshold), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_hate": float(precision_score(y, pred, zero_division=0)),
        "recall_hate": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
    }


def http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers["User-Agent"] = "Jaida-Calvin-NLP-course-research"
    return s


def get_bytes(url: str) -> bytes:
    try:
        r = http_session().get(url, timeout=90)
        r.raise_for_status()
        return r.content
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download {url}. Check the Colab connection and retry the cell.") from exc


def github_commit(repo: str, branch: str) -> str:
    info = json.loads(get_bytes(f"https://api.github.com/repos/{repo}/commits/{branch}"))
    sha = info.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"Could not resolve a fixed revision for {repo}.")
    return sha


def download_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_file = root / "results" / "resources_lock.json"
    lock = json.loads(lock_file.read_text()) if lock_file.exists() else {"created_utc": now_utc(), "sources": {}}
    specs = {
        "hatexplain": ("hate-alert/HateXplain", "master", "Data/dataset.json", "hatexplain.json"),
        "hatexplain_splits": ("hate-alert/HateXplain", "master", "Data/post_id_divisions.json", "splits.json"),
        "hatecheck": ("paul-rottger/hatecheck-data", "main", "test_suite_cases.csv", "hatecheck.csv"),
    }
    revisions: dict[str, str] = {}
    for name, (repo, branch, rel, filename) in specs.items():
        saved = lock["sources"].get(name)
        if saved:
            sha = saved["revision"]
            revisions[repo] = sha
        else:
            if repo not in revisions:
                revisions[repo] = github_commit(repo, branch)
            sha = revisions[repo]
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{rel}"
        dest = raw_dir / filename
        payload = dest.read_bytes() if dest.exists() else get_bytes(url)
        digest = hashlib.sha256(payload).hexdigest()
        if saved and saved["sha256"] != digest:
            raise RuntimeError(f"Checksum changed for cached {name}; do not mix dataset versions.")
        dest.write_bytes(payload)
        lock["sources"][name] = {"url": url, "revision": sha, "sha256": digest}
        write_json(lock_file, lock)
    return (
        json.loads((raw_dir / "hatexplain.json").read_text(encoding="utf-8")),
        json.loads((raw_dir / "splits.json").read_text(encoding="utf-8")),
        pd.read_csv(raw_dir / "hatecheck.csv"),
    )


def parse_hatexplain(raw: dict[str, Any], divisions: dict[str, list[str]]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames, audit = {}, {}
    seen_ids: set[str] = set()
    for split in ("train", "val", "test"):
        ids = divisions[split]
        if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
            raise ValueError("Official split IDs are duplicated or overlap.")
        seen_ids.update(ids)
        rows, no_majority, empty = [], 0, 0
        for post_id in ids:
            item = raw[post_id]
            label = majority_label(item["annotators"])
            if label is None:
                no_majority += 1
                continue
            text = " ".join(item["post_tokens"])
            if not text.strip():
                empty += 1
                continue
            rows.append({"id": str(post_id), "text": text, "label_original": label,
                         "label": LABEL_MAP[label], "key": normal_key(text)})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"No usable examples in {split}.")
        # Drop all conflicting duplicates; keep only one of identical same-label texts.
        conflicts = frame.groupby("key")["label_original"].nunique()
        conflict_keys = set(conflicts[conflicts > 1].index)
        cleaned = frame[~frame["key"].isin(conflict_keys)].copy()
        n_conflicts = len(frame) - len(cleaned)
        n_dupes = int(cleaned.duplicated("key").sum())
        cleaned = cleaned.drop_duplicates("key", keep="first").reset_index(drop=True)
        frames[split] = cleaned
        audit[split] = {"official_ids": len(ids), "excluded_no_majority": no_majority,
                        "excluded_empty": empty, "excluded_conflicting_rows": n_conflicts,
                        "removed_within_split_duplicates": n_dupes}
    return frames, audit


def prepare_hatecheck(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "test_case", "label_gold", "functionality", "target_ident", "templ_id"}
    if not required.issubset(raw.columns):
        raise ValueError(f"HateCheck is missing required columns: {required - set(raw.columns)}")
    df = raw.copy()
    mapping = {"non-hateful": 0, "hateful": 1}
    df["label"] = df["label_gold"].map(mapping)
    if df["label"].isna().any() or df["case_id"].duplicated().any() or df["test_case"].isna().any():
        raise ValueError("HateCheck contains unknown labels, duplicate IDs, or missing text.")
    if len(df) != 3728 or df["functionality"].nunique() != 29:
        raise ValueError("Unexpected HateCheck version/size; review before proceeding.")
    df["label"] = df["label"].astype(int)
    df["id"] = df["case_id"].astype(str)
    df["text"] = df["test_case"].astype(str)
    df["key"] = df["text"].map(normal_key)
    return df.reset_index(drop=True)


def isolate_splits(frames: dict[str, pd.DataFrame], hc: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Preserve HateCheck, then test, then validation, removing overlaps from lower-priority sets."""
    out, removed = {}, {}
    forbidden = set(hc["key"])
    for split in ("test", "val", "train"):
        df = frames[split]
        kept = df[~df["key"].isin(forbidden)].copy().reset_index(drop=True)
        removed[split] = len(df) - len(kept)
        if len(kept) == 0 or set(kept["label"]) != {0, 1}:
            raise ValueError(f"Both binary classes are required after filtering in {split}.")
        out[split] = kept
        forbidden.update(kept["key"])
    return out, removed


def prepare_data(root: str | Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    data, divisions, raw_hc = download_sources(root)
    hc = prepare_hatecheck(raw_hc)
    frames, audit = parse_hatexplain(data, divisions)
    frames, removed = isolate_splits(frames, hc)
    summary = []
    for split in ("train", "val", "test"):
        df = frames[split]
        audit[split]["removed_cross_split_or_hatecheck_overlap"] = removed[split]
        audit[split]["final_n"] = len(df)
        summary.append({"dataset": "HateXplain", "split": split, "n": len(df),
                        "non_hate": int((df["label"] == 0).sum()), "hate": int((df["label"] == 1).sum())})
        # Save IDs and labels, not the raw post text, in the shareable results folder.
        dest = root / "results" / f"hatexplain_{split}_ids.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df[["id", "label_original", "label"]].to_csv(dest, index=False)
    summary.append({"dataset": "HateCheck", "split": "evaluation", "n": len(hc),
                    "non_hate": int((hc["label"] == 0).sum()), "hate": int((hc["label"] == 1).sum())})
    audit["hatecheck"] = {"n": len(hc), "functionalities": int(hc["functionality"].nunique()),
                           "used_for_training": False, "used_for_checkpoint_selection": False}
    write_json(root / "results" / "data_audit.json", audit)
    pd.DataFrame(summary).to_csv(root / "results" / "dataset_counts.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print("Data preparation complete. These counts are not model performance results.")
    return frames, hc, audit


def positive_class_index(id2label: dict[Any, str]) -> int:
    normalized = {int(i): re.sub(r"[^a-z]", "", str(v).lower()) for i, v in id2label.items()}
    positives = [i for i, text in normalized.items() if text in {"hate", "hateful", "hatespeech"}]
    negatives = [i for i, text in normalized.items() if text in {"nonhate", "nothate", "nonhateful", "nothateful", "nonhatespeech"}]
    if len(normalized) != 2 or len(positives) != 1 or len(negatives) != 1:
        raise ValueError(f"Unverified class mapping {id2label}. Stop rather than guessing.")
    return positives[0]


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment() -> dict[str, Any]:
    import torch
    libs = {}
    for name in ("torch", "transformers", "huggingface_hub", "numpy", "pandas", "scikit-learn", "requests"):
        try:
            libs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libs[name] = "not installed"
    return {"python": platform.python_version(), "libraries": libs,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def model_revision(root: Path, model_id: str) -> str:
    from huggingface_hub import HfApi
    path = root / "results" / "model_revisions.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if model_id not in saved:
        saved[model_id] = HfApi().model_info(model_id).sha
        write_json(path, saved)
    return saved[model_id]


def token_dataset(frame: pd.DataFrame, tokenizer: Any, max_length: int, hate_index: int):
    import torch
    enc = tokenizer(frame["text"].tolist(), truncation=True, max_length=max_length, padding=False)
    labels = [hate_index if int(x) == 1 else 1 - hate_index for x in frame["label"]]
    class EncodedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)
        def __getitem__(self, idx):
            return {**{k: v[idx] for k, v in enc.items()}, "labels": labels[idx]}
    return EncodedDataset()


def make_loader(dataset: Any, tokenizer: Any, batch_size: int, shuffle: bool, seed: int):
    import torch
    from transformers import DataCollatorWithPadding
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        generator=torch.Generator().manual_seed(seed),
    )


def infer_scores(model: Any, loader: Any, device: Any, hate_index: int) -> np.ndarray:
    import torch
    from tqdm.auto import tqdm
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            values.extend(torch.softmax(logits.float(), dim=-1)[:, hate_index].cpu().tolist())
    return np.asarray(values)


def save_evaluation(root: Path, model_key: str, condition: str, hc: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    prefix = f"{model_key}_{condition}"
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    columns = ["id", "label", "functionality", "target_ident", "templ_id"]
    pred = hc[columns].copy()
    pred["hate_score"] = scores
    pred["prediction"] = (scores >= 0.5).astype(int)
    pred.to_csv(out / f"{prefix}_predictions.csv", index=False)
    m = binary_metrics(hc["label"], scores)
    write_json(out / f"{prefix}_metrics.json", m)
    rows = []
    for functionality, indexes in hc.groupby("functionality").groups.items():
        ix = np.asarray(list(indexes), dtype=int)
        cm = binary_metrics(hc.loc[ix, "label"], scores[ix])
        rows.append({"functionality": functionality, **{k: cm[k] for k in ("n", "accuracy", "tn", "fp", "fn", "tp", "fpr", "fnr")}})
    pd.DataFrame(rows).to_csv(out / f"{prefix}_categories.csv", index=False)
    pd.DataFrame([binary_metrics(hc["label"], scores, t) for t in np.linspace(0.1, 0.9, 9)]).to_csv(
        out / f"{prefix}_thresholds.csv", index=False)
    # Error IDs only in shareable outputs. Join locally with HateCheck to review text.
    pred.loc[(pred.label == 0) & (pred.prediction == 1)].to_csv(out / f"{prefix}_false_positive_ids.csv", index=False)
    return m


def run_model(root: str | Path, frames: dict[str, pd.DataFrame], hc: pd.DataFrame,
              model_key: str = "dehatebert", config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate released checkpoint, update ALL parameters, select on validation, evaluate again.

    HateCheck metrics must not be used to change this run's hyperparameters.
    Any repeated run is a new experiment and must be documented, not silently overwritten.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm.auto import tqdm
    root = Path(root)
    cfg = {**DEFAULTS, **(config or {})}
    if model_key not in MODEL_IDS:
        raise ValueError(f"Choose one of {list(MODEL_IDS)}.")
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU was not detected. In Colab choose Runtime > Change runtime type > T4 GPU, save, then rerun setup.")
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        raise RuntimeError("Use a current Colab runtime with PyTorch 2.6+; do not disable model-loading safety checks.")
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / model_key
    if (run_dir / "status.json").exists():
        existing = json.loads((run_dir / "status.json").read_text())
        if existing.get("status") == "completed":
            print(f"{model_key} already completed. Returning saved metrics without retraining.")
            return json.loads((output / f"{model_key}_finetuned_metrics.json").read_text())
        raise RuntimeError("A previous attempt exists. Export its outputs before starting a separate documented run.")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    model_id = MODEL_IDS[model_key]
    revision = model_revision(root, model_id)
    metadata = {"started_utc": now_utc(), "model_id": model_id, "revision": revision,
                "config": cfg, "environment": environment(),
                "train_n": len(frames["train"]), "validation_n": len(frames["val"]),
                "hatecheck_n": len(hc), "condition": "continued full-parameter fine-tuning",
                "selection": "highest HateXplain validation macro-F1 among the two trained epochs; earlier epoch wins ties",
                "caveat": "Released checkpoints may have seen HateXplain previously; validation is not claimed independent of upstream training."}
    write_json(run_dir / "run_config.json", metadata)
    write_json(run_dir / "status.json", {"status": "started", "utc": now_utc(), "epochs_completed": 0})
    model = tokenizer = optimizer = scheduler = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        hate_index = positive_class_index(model.config.id2label)
        metadata["id2label"] = {str(k): v for k, v in model.config.id2label.items()}
        metadata["hate_index"] = hate_index
        metadata["truncated_examples"] = {}
        for name, df in {"train": frames["train"], "val": frames["val"], "hatecheck": hc}.items():
            lengths = tokenizer(df.text.tolist(), truncation=False, padding=False, return_length=True)["length"]
            metadata["truncated_examples"][name] = int(np.sum(np.asarray(lengths) > cfg["max_length"]))
        datasets = {name: token_dataset(df, tokenizer, cfg["max_length"], hate_index)
                    for name, df in {"train": frames["train"], "val": frames["val"], "hatecheck": hc}.items()}
        train_loader = make_loader(datasets["train"], tokenizer, cfg["batch_size"], True, cfg["seed"])
        val_loader = make_loader(datasets["val"], tokenizer, cfg["eval_batch_size"], False, cfg["seed"])
        hc_loader = make_loader(datasets["hatecheck"], tokenizer, cfg["eval_batch_size"], False, cfg["seed"])
        device = torch.device("cuda")
        model.to(device)
        baseline_scores = infer_scores(model, hc_loader, device, hate_index)
        save_evaluation(root, model_key, "released", hc, baseline_scores)
        print("Released-model outputs saved. Do not change hyperparameters based on these test results.")
        for p in model.parameters():
            p.requires_grad_(True)
        metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        write_json(run_dir / "run_config.json", metadata)
        # Full network updates, not just fitting the final classification head.
        model.gradient_checkpointing_enable()
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(s in n for s in no_decay)], "weight_decay": cfg["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if any(s in n for s in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=cfg["learning_rate"])
        steps_per_epoch = math.ceil(len(train_loader) / cfg["gradient_accumulation"])
        total_steps = steps_per_epoch * cfg["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)
        scaler = torch.amp.GradScaler("cuda")
        best_f1, history = -1.0, []
        best_path = root / "checkpoints" / model_key / "best"
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss, example_count = 0.0, 0
            bar = tqdm(train_loader, desc=f"{model_key}: training epoch {epoch}/{cfg['epochs']}")
            for step, batch in enumerate(bar):
                batch = {k: v.to(device) for k, v in batch.items()}
                # Correct normalization for the last, potentially incomplete accumulation group.
                start = (step // cfg["gradient_accumulation"]) * cfg["gradient_accumulation"]
                accumulation_group = min(cfg["gradient_accumulation"], len(train_loader) - start)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw_loss = model(**batch).loss
                    loss = raw_loss / accumulation_group
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; do not report this attempt as successful.")
                scaler.scale(loss).backward()
                n = len(batch["labels"])
                total_loss += float(raw_loss.detach()) * n
                example_count += n
                if (step + 1) % cfg["gradient_accumulation"] == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(loss=f"{total_loss / example_count:.4f}")
            val_scores = infer_scores(model, val_loader, device, hate_index)
            val_metrics = binary_metrics(frames["val"]["label"], val_scores)
            row = {"epoch": epoch, "train_loss": total_loss / example_count,
                   "validation_macro_f1": val_metrics["macro_f1"], "validation_accuracy": val_metrics["accuracy"]}
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                metadata["selected_epoch"] = epoch
                model.save_pretrained(best_path, safe_serialization=True)
                tokenizer.save_pretrained(best_path)
                write_json(run_dir / "selected_validation_metrics.json", val_metrics)
                write_json(run_dir / "run_config.json", metadata)
            write_json(run_dir / "status.json", {"status": "training", "epochs_completed": epoch, "utc": now_utc()})
            print(f"Epoch {epoch}: validation macro-F1 = {val_metrics['macro_f1']:.4f}")
        del optimizer, scheduler
        optimizer = scheduler = None
        model.cpu()
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(best_path, trust_remote_code=False).to(device)
        ft_scores = infer_scores(model, hc_loader, device, hate_index)
        ft = save_evaluation(root, model_key, "finetuned", hc, ft_scores)
        write_json(run_dir / "status.json", {"status": "completed", "epochs_completed": cfg["epochs"],
                                             "selected_epoch": metadata["selected_epoch"], "completed_utc": now_utc()})
        write_summary(root)
        print(f"Completed {model_key}. Real measurements are in results/. No improvement is assumed.")
        return ft
    except Exception as exc:
        prev = json.loads((run_dir / "status.json").read_text())
        write_json(run_dir / "status.json", {**prev, "status": "interrupted_or_failed", "error": str(exc), "utc": now_utc()})
        raise
    finally:
        if model is not None:
            model.cpu()
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def write_summary(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for key in MODEL_IDS:
        for condition in ("released", "finetuned"):
            path = root / "results" / f"{key}_{condition}_metrics.json"
            if path.exists():
                rows.append({"model": key, "condition": condition, **json.loads(path.read_text())})
    table = pd.DataFrame(rows)
    if table.empty:
        print("No model measurements yet. A missing result is not a score of zero.")
        return table
    table.to_csv(root / "results" / "model_comparison.csv", index=False)
    cols = ["model", "condition", "n", "accuracy", "macro_f1", "fpr", "fnr"]
    lines = ["# Measured results for Project Update 1", "", "Only completed evaluation outputs are listed. Scores are fractions, not percentages.", "",
             "```", table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"), "```", "",
             "The labels are HateCheck's annotations. These are benchmark results, not platform-level error rates.",
             "HateCheck was not used for this project's training/checkpoint selection. Threshold 0.5 was fixed in advance.", ""]
    for key in MODEL_IDS:
        part = table[table.model == key].set_index("condition")
        if {"released", "finetuned"}.issubset(part.index):
            delta_fp = 100 * (part.loc["finetuned", "fpr"] - part.loc["released", "fpr"])
            delta_fn = 100 * (part.loc["finetuned", "fnr"] - part.loc["released", "fnr"])
            lines.append(f"{key}: fine-tuned minus released FPR = {delta_fp:+.2f} percentage points; FNR = {delta_fn:+.2f} percentage points. Interpret BOTH changes; neither improvement is assumed.")
        else:
            lines.append(f"{key}: paired before/after comparison is not complete.")
    lines += ["", "Still pending unless separately documented: matched-identity bootstrap analysis, manual error review, repeated training seeds, and publication of the GitHub repository.",
              "Cardiff's original training already included HateXplain. Do not call its HateXplain validation/test scores independent unseen-data results."]
    (root / "results" / "results_for_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return table


def export_results(root: str | Path) -> Path:
    root = Path(root)
    write_summary(root)
    result_dir = root / "results"
    if not result_dir.exists():
        raise RuntimeError("No results folder exists yet; run data preparation first.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = root.parent / f"Jaida_Update1_Results_{stamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file in result_dir.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(root))
        if (root / "project.py").exists():
            z.write(root / "project.py", "project.py")
    print(f"Saved {archive.name}. Raw texts and large model weights are not included.")
    return archive
