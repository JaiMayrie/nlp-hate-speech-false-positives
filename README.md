# Evaluating False Positives in AI Hate-Speech Detection
Jaida Calvin — Natural Language Processing, Purdue University Fort Wayne, Fall 2026.

## Project Update 1 status
**One real before/after fine-tuning experiment is complete.** DeHateBERT-English was evaluated in its released state, fine-tuned for two epochs on HateXplain, and evaluated again on all 3,728 HateCheck cases. The executed notebook, predictions, training history, data audit, source hashes, model revision, environment and settings are saved here. CardiffNLP Twitter-RoBERTa has NOT been run; the second-model notebook cell remains unexecuted.

The run completed on October 30, 2026 in America/Indiana/Indianapolis. Its saved UTC completion time is October 31, 2026, 02:13:25; these refer to the same run.

## Main measured results
All rates below are percentages; macro-F1 is on a 0–1 scale. Threshold 0.5 was fixed before the run.

| HateCheck measure | Released | Further fine-tuned |
|---|---:|---:|
| Accuracy | 42.52% | 34.33% |
| Macro-F1 | 0.4176 | 0.2990 |
| Hate precision | 78.38% | 75.33% |
| Hate recall | 22.63% | 6.67% |
| False-positive rate | 13.73% | 4.81% |
| False-negative rate | 77.37% | 93.33% |
| False-positive count | 160 | 56 |
| False-negative count | 1,983 | 2,392 |

**Interpretation:** False alarms decreased, but 409 more hateful cases were missed. This run did NOT improve overall performance and does not establish bias mitigation. Causes remain unconfirmed. All per-case predictions and category/threshold tables are included; no scores for an unrun model are substituted or invented.

## Revised question
How does continued supervised fine-tuning change false positives and false negatives on HateCheck relative to each model's released checkpoint? Which categories and, in future matched analyses, identity mentions remain difficult?

## Models and data
- Completed: `Hate-speech-CNERG/dehatebert-mono-english`; full revision is in `results/dehatebert/run_config.json`.
- Planned: `cardiffnlp/twitter-roberta-base-hate-latest`.
- HateXplain uses original split IDs. Resolve the majority of ORIGINAL three-class annotations, then map `hatespeech` to 1 and `normal`/`offensive` to 0. Not-hate does not mean harmless. Rationales and target labels are not used for training.
- Post-filtering counts: 15,344 training, 1,921 validation, 1,923 reserved test examples. The reserved source test split has NOT been evaluated.
- HateCheck: 3,728 diagnostic evaluation cases (1,165 non-hateful; 2,563 hateful), not used for training/checkpoint selection. The main threshold is 0.5; 0.1–0.9 is a descriptive sensitivity grid, not test-set threshold optimization.
- Original tokens are joined for the model input; Unicode/case/whitespace normalization is used only for duplicate/overlap checks.
- Cardiff's published upstream training includes HateXplain. Further training is continued adaptation, not first exposure; independent unseen-data status of source validation/test data is not established. Within-project filtering cannot undo upstream exposure.

## Completed training
All 167,357,954 model parameters were trainable. Two epochs; AdamW; learning rate 2e-5; batch size 8; accumulation 2; evaluation batch 16; maximum length 128; weight decay 0.01; linear schedule with 10% warmup; clipping 1.0; seed 42; mixed precision; gradient checkpointing; Tesla T4.

Epoch 1: training loss 0.4246, validation macro-F1 0.8229. Epoch 2: loss 0.3438, validation macro-F1 0.8256. Epoch 2 was selected using HateXplain validation, not HateCheck. No released-checkpoint validation baseline was recorded.

## Completed and pending analysis
Completed: overall before/after evaluation, all 29 functionality tables, fixed nine-threshold grids, false-positive case-ID exports, and a consistency audit recomputing metrics from predictions.

Pending: second-model experiment; matched identity-template comparisons and template bootstrap intervals; manual review of up to 60 false positives; repeated seeds; a controlled class-weighted-loss comparison. No independent loss-function or frozen-encoder ablation has been completed. Further experiments will be documented as follow-up work after observing this diagnostic benchmark.

Example category finding: neutral identity false positives changed from 1/126 to 0/126 and positive identity false positives from 9/189 to 0/189, but quoting-hate counterspeech errors increased from 16/173 to 22/173. These are category totals, not matched identity-group results.

## Verification and limitations
`evidence/verification_audit.json` records 28 passed checks of saved outputs. The executed notebook's embedded source matches `project.py`. Overall metrics, all category rows, all threshold rows, and false-positive IDs agree with the saved per-case predictions. Split-ID, label-mapping, class-count and epoch-selection checks also pass. This does NOT independently rerun neural inference/training or certify every possible implementation detail.

The 12 offline helper tests were re-run successfully; `evidence/offline_tests_rechecked.txt` contains synthetic-fixture software checks, NOT an additional model experiment. Its toy scores must not be reported as research results.

Only one seed/model is complete. A nondeterministic-attention warning was logged; exact bitwise reproduction is not guaranteed. No confidence intervals or significance claims are available. These are constructed benchmark results, not estimates of platform-wide error or proof of discrimination. Source-domain validation and diagnostic benchmark scores measure different things.

## Reproducing and viewing the notebook
`Jaida_NLP_Update1_FineTuning.ipynb` is the executed notebook from the first run, renamed to replace the starter notebook. Open a copy in a fresh Colab GPU runtime for a new reproduction. Its saved outputs already document this run; rerunning is not required just to view them. Running from a fresh environment retrains from the pinned released checkpoint only if the accompanying saved lock files are copied into the working results directory first; otherwise the code resolves the current revision. Use the recorded versions/revisions for strict comparisons.

The local fine-tuned checkpoint weights are not in this small export. Keep the active Colab checkpoint or save it separately for reuse; weights should not be uploaded through this repository's normal browser flow. Code implementation and paper preparation used AI assistance.

## Repository contents
- `project.py`: exact source exported by the completed run.
- `Jaida_NLP_Update1_FineTuning.ipynb`: executed notebook with its original outputs.
- `results/`: measured scores, predictions, data IDs, provenance, and training logs.
- `evidence/`: post-run verification and helper-test evidence, plus the unedited original generated summary.
- `tests/` and `requirements.txt`: starter tests and dependencies already in the repository.

The original generated results summary contained stale wording about repository publication. `results/results_for_report.md` was updated for status clarity without changing numeric results; the original is preserved in `evidence/original_results_for_report.md`.

## Sources and responsible sharing
- Röttger et al. (2021), HateCheck: https://aclanthology.org/2021.acl-long.4/
- HateCheck data: https://github.com/paul-rottger/hatecheck-data
- Mathew et al. (2021), HateXplain: https://doi.org/10.1609/aaai.v35i17.17745
- HateXplain original data/splits: https://github.com/hate-alert/HateXplain
- Aluru et al. (2020): https://arxiv.org/abs/2004.06465
- Antypas and Camacho-Collados (2023): https://aclanthology.org/2023.woah-1.25/
- Sap et al. (2019): https://aclanthology.org/P19-1163/
- DeHateBERT model card: https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english
- Cardiff model card: https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest

Do not upload credentials, raw cached text, private notes, or large model weights. Review source licenses before redistributing resources. Exports contain IDs, labels and predictions rather than raw offensive text; upstream HateCheck content remains attributed to its creators. Sources support background/method descriptions; this project's reported results come from the saved Colab run.
