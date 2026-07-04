# Friedman (2001) — Greedy Function Approximation (Gradient Boosting)

**Source:** Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189-1232.
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Metric/Threshold | Value | Interpretation | Source |
|---|---|---|---|
| learning_rate × n_estimators tradeoff | Lower lr → more trees needed | Shrinkage improves generalization | Friedman 2001 |
| Typical learning_rate | 0.01 - 0.3 | Lower = more regularized | Friedman 2001 |
| Typical n_estimators | 100 - 1000 | Must increase as lr decreases | Friedman 2001 |
| Typical max_depth (GBM) | 3 - 8 | Shallow trees = weak learners | Friedman 2001 |
| Typical subsample | 0.5 - 1.0 | < 1.0 adds stochastic boosting | Friedman 2001 |

## Hyperparameters

| Parameter | What it controls | Typical range | Relationship |
|---|---|---|---|
| learning_rate (η) | Step size per tree | 0.01 - 0.3 | Inverse with n_estimators |
| n_estimators (M) | Number of boosting rounds | 100 - 1000 | Inverse with learning_rate |
| max_depth | Complexity per tree | 3 - 8 | Higher = more overfitting risk |
| subsample | Fraction of data per tree | 0.5 - 1.0 | < 1.0 adds randomness |
| min_samples_split | Min samples to split | 2 - 20 | Higher = more regularized |
| min_samples_leaf | Min samples in leaf | 1 - 10 | Higher = more regularized |
| colsample_bytree | Features per tree | 0.5 - 1.0 | Random feature selection |

## The lr × n_estimators Tradeoff

| learning_rate | n_estimators needed | Overfitting risk | Generalization |
|---|---|---|---|
| 0.3 (high) | ~100 | Higher | Faster to train, worse on test |
| 0.1 (medium) | ~300 | Medium | Balanced |
| 0.01 (low) | ~1000+ | Lower | Best generalization, slowest |

**Rule of thumb:** Halving lr ≈ doubling n_estimators for similar performance.

## XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Split finding | Level-wise | Leaf-wise | Symmetric trees |
| Speed | Fast | Fastest | Medium |
| Categorical encoding | Manual | Native (optimal split) | Native (ordered TS) |
| Regularization | L1/L2 on leaves | L1/L2 + max_depth | Ordered boosting |
| Default lr | 0.3 | 0.1 | 0.03 |
| When to use | General purpose | Large datasets, speed | Categorical-heavy data |
| GPU support | Yes | Yes | Yes |

## When to Use

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| Non-linear + interactions | GBM | OLS | GBM captures automatically |
| Need interpretability | OLS or SHAP+GBM | GBM alone | GBM is black box |
| Small dataset (< 200) | OLS or RF | GBM | GBM overfits easily |
| Large dataset (> 10K) | LightGBM or XGBoost | OLS | OLS is slow, GBM scales |
| Categorical features | CatBoost | XGBoost | CatBoost handles natively |
| Need probability calibration | GBM + Platt scaling | Naive GBM | GBM outputs can be uncalibrated |
| Need speed | LightGBM | XGBoost | Leaf-wise is faster |

## Decision Rules

1. **Start with XGBoost** as default — well-tested, good documentation.
2. **If speed matters** → switch to LightGBM.
3. **If many categorical features** → use CatBoost.
4. **Always tune learning_rate first** — it has the largest impact on generalization.
5. **Use early stopping** — stop when validation loss plateaus (avoids overfitting).
6. **Apply SHAP** after training for interpretability.
7. **Calibrate probabilities** if using for decision-making (Platt scaling, isotonic).

## Cross-references

- See also: `breiman-2001-rf.md` (bagging vs boosting comparison)
- See also: `shap-lime.md` (interpreting GBM predictions)
- See also: `cross-validation.md` (early stopping uses CV)
