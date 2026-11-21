# Evaluating False Positives in AI Hate-Speech Detection
Jaida Calvin — Natural Language Processing, Purdue University Fort Wayne, Fall 2026.

## Project Update 2 status
One model-training experiment is complete: DeHateBERT-English was evaluated before and after two epochs of continued full-parameter fine-tuning on HateXplain. Update 2 adds **new analyses of that saved run**, not a second training experiment.

The report, executed analysis notebook, reproducible analysis code, tables, and audit records are in [`update2/`](update2/README.md). Original Update 1 predictions, source code, settings, and executed notebook remain unchanged in their existing locations.

### Original full-benchmark results (not a new run)
| Measure | Released DeHateBERT | Further fine-tuned |
|---|---:|---:|
| HateCheck accuracy | 42.52% | 34.33% |
| Macro-F1 | 0.4176 | 0.2990 |
| False-positive rate | 13.73% | 4.81% |
| False-negative rate | 77.37% | 93.33% |
| False positives | 160 | 56 |
| False negatives | 1,983 | 2,392 |

### Newly completed analysis for Update 2
- **Paired error transitions:** 128 original false positives corrected, 32 persist, and 24 new false positives introduced. Among hateful cases, 443 previously detected cases become missed; 34 previously missed cases become detected.
- **Matched identities:** 45 shared neutral/positive sentence templates, seven groups, 315 cases. Before training, there are 10 false positives; after training, none. The largest released-model FPR is 7/45 (15.56%) for mentions of immigrants. This is not a population-level estimate or a causal claim.
- **Template bootstrap:** 1,000 paired resamples, seed 42. The released max-minus-min group FPR gap is 15.56 percentage points, with empirical bootstrap interval [4.44, 26.67]. The fine-tuned interval is [0, 0] because all observed matched errors are zero; this does NOT establish zero population risk or absence of bias.
- **Score shifts:** Hate scores decreased for 90.73% of non-hateful cases and 92.24% of hateful cases. The reduction is not selective to the non-hateful class.

**Conclusion:** Fewer false alarms did not yield overall improvement. New matched results must be interpreted alongside the much higher full-benchmark false-negative rate. New analyses do not isolate why the model's behavior changed.

## Completed training and data
- Model: `Hate-speech-CNERG/dehatebert-mono-english`; exact revision in `results/dehatebert/run_config.json`.
- Source training: original HateXplain splits, after majority-label resolution and overlap filtering. Binary mapping: hatespeech=1; normal/offensive=0. Not-hate does not mean harmless.
- Counts: 15,344 training, 1,921 validation, 1,923 reserved source-test examples. The reserved source-test set has not been evaluated.
- Evaluation: all 3,728 HateCheck cases, evaluation-only. Checkpoint selection uses HateXplain validation, not HateCheck.
- Preset settings: two epochs, AdamW, learning rate 2e-5, batch size 8, accumulation 2, max length 128, seed 42. Full settings, model revision, source hashes and environment are saved.
- Main decision threshold: 0.5. Original category and threshold analyses were completed in Update 1, not newly in Update 2.

## Still planned—not completed
CardiffNLP released-versus-further-fine-tuned comparison; inverse-frequency-weighted loss comparison; repeated training seeds; manual raw-text review of up to 60 false positives. No second-model or ablation scores are invented. Cardiff's upstream training includes HateXplain, so it must not be described as previously unseen source data. Subsequent variants are exploratory because HateCheck has already been inspected.

## Reproduce the new analysis
From the repository root:

```bash
python update2/analyze_update2.py --input-dir results --output-dir update2
python update2/test_analysis.py
```

The analysis needs Python 3.10+, numpy, and pandas; it does not need a GPU. The executed `update2/Jaida_NLP_Update2_Analysis.ipynb` reproduces the CSV outputs and can also retrieve the four required small input files from the pinned Update 1 repository snapshot if they are not local. Input hashes are checked. The original training notebook remains `Jaida_NLP_Update1_FineTuning.ipynb`.

## Report and source attribution
The Update 2 PDF and editable Overleaf source are in `update2/report/`. AI assistance was used for implementation, saved-output analysis, and writing. Software tests and recomputed tables are not additional neural training experiments. Raw offensive texts, credentials and large model weights are not included.

Research resources:
- Röttger et al. (2021), HateCheck: https://aclanthology.org/2021.acl-long.4/
- HateCheck data: https://github.com/paul-rottger/hatecheck-data
- Mathew et al. (2021), HateXplain: https://doi.org/10.1609/aaai.v35i17.17745
- HateXplain data: https://github.com/hate-alert/HateXplain
- Aluru et al. (2020): https://arxiv.org/abs/2004.06465
- Antypas and Camacho-Collados (2023): https://aclanthology.org/2023.woah-1.25/
- Sap et al. (2019): https://aclanthology.org/P19-1163/
- DeHateBERT model card: https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english
- Cardiff model card: https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest

The sources support background and resource descriptions. Project numbers are computed from the saved Colab predictions, not taken from those publications.
