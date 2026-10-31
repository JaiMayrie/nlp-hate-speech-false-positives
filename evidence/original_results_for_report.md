# Measured results for Project Update 1

Only completed evaluation outputs are listed. Scores are fractions, not percentages.

```
     model condition    n  accuracy  macro_f1    fpr    fnr
dehatebert  released 3728    0.4252    0.4176 0.1373 0.7737
dehatebert finetuned 3728    0.3433    0.2990 0.0481 0.9333
```

The labels are HateCheck's annotations. These are benchmark results, not platform-level error rates.
HateCheck was not used for this project's training/checkpoint selection. Threshold 0.5 was fixed in advance.

dehatebert: fine-tuned minus released FPR = -8.93 percentage points; FNR = +15.96 percentage points. Interpret BOTH changes; neither improvement is assumed.
roberta: paired before/after comparison is not complete.

Still pending unless separately documented: matched-identity bootstrap analysis, manual error review, repeated training seeds, and publication of the GitHub repository.
Cardiff's original training already included HateXplain. Do not call its HateXplain validation/test scores independent unseen-data results.