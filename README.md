# Evaluating False Positives in AI Hate-Speech Detection
Jaida Calvin — Natural Language Processing, Purdue University Fort Wayne, Fall 2026.

## Status
**Prepared implementation and working paper draft. No real model run is bundled.**
The paper is not submission-ready until actual progress/results and a verified GitHub link are added.
Twelve offline unit tests pass on synthetic fixtures; they are not experimental performance.
Full network download, GPU inference and fine-tuning have not been executed in the preparation environment.

## Revised question
How does continued supervised fine-tuning change false positives and false negatives on HateCheck, relative to each model's released checkpoint? Which kinds of non-hateful statements remain difficult?

## Models and data
- Released references: `Hate-speech-CNERG/dehatebert-mono-english` and `cardiffnlp/twitter-roberta-base-hate-latest`.
- Fine-tuning: all parameters updated on the original HateXplain training split, with checkpoint selection on its validation split.
- Binary labels: resolve the majority of ORIGINAL three-class annotations first. Map hatespeech to 1, normal/offensive to 0; exclude no-majority examples. Not-hate is not the same as harmless.
- HateCheck: evaluation only, no training or checkpoint/threshold selection. Main threshold 0.5; descriptive sensitivity grid 0.1–0.9.
- Published Cardiff training includes HateXplain. Additional training is continued adaptation, not first exposure. Source validation/test scores are not claimed independent of upstream training. Removing within-project duplicates cannot undo prior training exposure.
- Raw text is used for model inputs. Unicode/case/whitespace normalization is used ONLY for overlap checks. The code records post-filtering sizes and truncation counts.

## First run
Open `Jaida_NLP_Update1_FineTuning.ipynb` in a fresh Colab GPU runtime. Follow the numbered cells. Begin with DeHateBERT; the second model is off by default. Both are in the overall study plan.
No paid API or Docker is required. GPU availability is not guaranteed.

Preset: two epochs, AdamW, learning rate 2e-5, batch size 8, accumulation 2 (effective batch up to 16), max length 128, weight decay 0.01, warmup 10%, seed 42. Select the trained epoch with the best validation macro-F1; earlier epoch wins ties. These are chosen settings, not empirically optimized settings.

## Reproducibility
Source commits, SHA-256 hashes, model revisions, versions, GPU, class mappings, data counts and settings are saved in `results/`. Prediction exports contain IDs and scores but not raw offensive text. Checkpoints are kept locally and excluded from the small export ZIP. Re-execution retrains from the pinned released checkpoint; exact bitwise reproducibility across hardware is not promised.

## Offline checks
`python -m unittest discover -s tests -v`
Checks label mapping, no-majority handling, metric formulas, absent denominators, overlap removal, export content and missing-result handling. These checks are not neural model tests.

## Planned analyses not yet completed
Matched identity templates and bootstrap intervals, manual review of up to 60 false positives, and optional contextual pairs remain future analyses. The existing threshold/category code produces real tables only after inference has run. Before/after comparisons are within a checkpoint; cross-model differences cannot isolate architecture because training histories differ.

## Sharing
Create a separate GitHub repository and add its verified URL to the paper. No repository has been created by this package. Do not upload credentials, raw text caches, large weights or private notes. Review the dataset/model licenses before redistributing them; this bundle contains neither datasets nor weights. Follow course requirements on acknowledging AI assistance.

## Sources
- Röttger et al. (2021), HateCheck: https://aclanthology.org/2021.acl-long.4/
- HateCheck data, CC BY 4.0: https://github.com/paul-rottger/hatecheck-data
- Mathew et al. (2021), HateXplain: https://doi.org/10.1609/aaai.v35i17.17745
- Original data/splits: https://github.com/hate-alert/HateXplain
- Aluru et al. (2020): https://arxiv.org/abs/2004.06465
- Antypas & Camacho-Collados (2023): https://aclanthology.org/2023.woah-1.25/
- Sap et al. (2019): https://aclanthology.org/P19-1163/
- DeHateBERT model card: https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english
- Cardiff model card: https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest
- Colab usage: https://research.google.com/colaboratory/faq.html

The sources support the design and resource descriptions, not claims about this project's unrun experiments.
