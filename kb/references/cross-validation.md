# Cross-Validation — Hastie, Tibshirani & Friedman (2009)

**Source:** Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Ch. 7.
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Metric/Threshold | Value | Interpretation | Source |
|---|---|---|---|
| k-fold CV k | 5 or 10 | Standard choice; bias-variance tradeoff | ESL Ch. 7 |
| LOOCV k = n | Leave-one-out | Low bias, high variance | ESL Ch. 7 |
| Repeated CV | 5-10 repeats | Reduces variance of estimate | ESL Ch. 7 |
| Nested CV | Outer + inner | Unbiased performance + tuning | ESL Ch. 7 |
| Bootstrap .632+ | 0.632 × train + 0.368 × test | Corrects optimism of bootstrap | ESL Ch. 7 |

## Cross-Validation Methods

| Method | k | Bias | Variance | Computation | Best for |
|---|---|---|---|---|---|
| k-fold (k=5) | 5 | Medium | Medium | Fast | General use |
| k-fold (k=10) | 10 | Low | Low | Medium | General use (default) |
| LOOCV | n | Very low | High | Very slow | Small n (< 50) |
| Stratified k-fold | k (balanced) | Medium | Medium | Fast | Imbalanced classes |
| Repeated k-fold | k × r | Low | Low | Medium | Reducing variance |
| Nested CV | Outer × Inner | Low | Low | Slow | Model selection + tuning |
| Bootstrap .632+ | B samples | Low | Low | Medium | Small datasets |

## Bias-Variance Tradeoff

| Method | Bias | Variance | Total Error |
|---|---|---|---|
| k=5 | Higher (train < full data) | Lower (averaged) | Balanced |
| k=10 | Lower | Slightly higher | Balanced |
| LOOCV | Very low (train ≈ full data) | Very high (nearly identical folds) | Can be high |
| Bootstrap | Low | Medium | Usually good |

## When to Use

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| Model selection | Nested CV | Single train/test split | Unbiased comparison |
| Hyperparameter tuning | Inner loop of nested CV | Outer loop | Avoid data leakage |
| Imbalanced classes | Stratified k-fold | Random k-fold | Preserves class ratios |
| Small dataset (n < 50) | LOOCV or bootstrap | k-fold (k=5) | Too few samples per fold |
| Large dataset (n > 10K) | Single split or k=5 | LOOCV | Computationally expensive |
| Need unbiased estimate | Nested CV | Single CV | Single CV optimistically biased |
| Time series | Rolling/expanding window | Random k-fold | Temporal order matters |

## Decision Rules

1. **Default: k=10 stratified** — good bias-variance balance for most problems.
2. **Small n (< 50):** use LOOCV or k=n-1 — every observation matters.
3. **Model comparison:** use nested CV — outer loop for performance, inner for tuning.
4. **Never tune hyperparameters on the test fold** — use inner loop only.
5. **Imbalanced data:** always stratified — preserves class distribution in each fold.
6. **Time series:** never shuffle — use temporal splits (rolling window or expanding window).
7. **Report variance** — not just mean CV error. High variance = unstable model.

## Cross-references

- See also: `gujarati-porter.md` (in-sample diagnostics: R², F-test)
- See also: `breiman-2001-rf.md` (OOB as alternative to CV)
- See also: `model-selection.md` (when CV is part of selection process)
