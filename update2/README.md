# Project Update 2 — analysis package

This folder adds analysis to the completed Update 1 DeHateBERT run. **No new neural training or inference was performed for these outputs.**

## Read first
- `report/Jaida_Calvin_NLP_Project_Update2.pdf`: ACL-formatted progress report.
- `Jaida_NLP_Update2_Analysis.ipynb`: executed notebook showing the new analysis. No rerun is required to read its saved outputs.
- `analyze_update2.py`: reusable implementation; takes the existing root `results/` inputs.
- `results/`: CSV outputs. No raw offensive text.
- `evidence/`: input hashes, environment information, analysis checks and notebook reproduction record.
- `test_analysis.py`: five software tests on small synthetic inputs. These are not research model runs.
- `report/Jaida_NLP_Update2_Overleaf.zip`: editable paper source for Overleaf.

## What is new
`prediction_transitions.csv` and `category_transitions.csv` separate corrected errors from newly introduced errors. `matched_identity_fpr.csv` and `matched_group_gaps.csv` report 45 complete seven-group neutral/positive templates. `matched_fpr_changes.csv` uses paired bootstrap changes. The template matrices make the resampling unit inspectable. `score_shift_by_label.csv` reports score changes separately for both gold classes. `overall_recheck.csv` repeats the old full-benchmark result for context, not as a new run.

## Reproduction
From the repository root, run:

```bash
python update2/analyze_update2.py --input-dir results --output-dir update2
python update2/test_analysis.py
```

The notebook's four analysis code cells were executed and all 11 resulting CSV files matched the separate script run byte-for-byte. Its optional download cell is left unexecuted. This checks consistency, not the absence of every possible implementation error.

Bootstrap settings: 1,000 draws; seed 42; sample whole `(functionality, templ_id)` clusters with replacement; keep all seven identities and both model versions in each draw; equal weight per template. The intervals concern the observed template set and do not include training-seed or platform-population uncertainty. All-zero empirical data yield [0,0] bootstrap intervals without establishing zero underlying risk.

The original input files match the Update 1 repository blobs:
- released predictions: `690571b3417a6a0a99f714ac6a8cd0248f4c3959`
- fine-tuned predictions: `ae68fec5573e6a1dabc903c9a1707ae883ee05fb`

Cardiff training, controlled loss comparisons, repeated training seeds, and manual raw-text error review are still upcoming. AI assistance was used in preparing the new code, analysis, and report. All results in this folder were actually computed from the saved predictions.
