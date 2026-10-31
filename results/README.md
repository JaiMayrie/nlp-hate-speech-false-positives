# Measured outputs — Project Update 1

A DeHateBERT released-versus-continued-fine-tuned run is complete. The previous starter message saying no experiments were complete is superseded.

- `model_comparison.csv` and `*_metrics.json`: measured aggregate scores. Rates in CSV/JSON are fractions, not percentages.
- `*_predictions.csv`: all 3,728 case-level scores and labels for each condition, with no raw text.
- `*_categories.csv`: all 29 functionalities; blank FPR/FNR entries mean no applicable denominator, not zero.
- `*_thresholds.csv`: the fixed 0.1–0.9 descriptive grid.
- `*_false_positive_ids.csv`: false-positive case IDs, not manual qualitative analysis.
- `dataset_counts.csv`, `data_audit.json`, and `hatexplain_*_ids.csv`: actual prepared split counts and IDs; source test examples remain unused for evaluation.
- `dehatebert/`: two-epoch history, selected validation metrics, configuration and completion status.
- `resources_lock.json` and `model_revisions.json`: pinned sources and model version.
- `results_for_report.md`: updated readable summary. See `../evidence/original_results_for_report.md` for the original generated text.

No Cardiff results are present because that model has not been run. No model weights are included. False positives declined, but false negatives increased substantially. The outputs do not establish an overall improvement or reduced bias.
