# Breiman (2001) — Random Forests

**Source:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Metric/Threshold | Value | Interpretation | Source |
|---|---|---|---|
| mtry (classification) | √p | Square root of number of features | Breiman 2001 |
| mtry (regression) | p/3 | One-third of features | Breiman 2001 |
| ntree | ≥ 500 | Sufficient trees for stability | Breiman 2001 |
| OOB error | Stabilizes ~200-500 trees | No improvement beyond ~500 | Breiman 2001 |
| Min node size (classification) | 1 | Default for classification | Breiman 2001 |
| Min node size (regression) | 5 | Default for regression | Breiman 2001 |

## Hyperparameters

| Parameter | What it controls | Default | Tuning range |
|---|---|---|---|
| mtry | Features per split | √p (class) or p/3 (reg) | 0.2p to 0.8p |
| ntree | Number of trees | 500 | 500 - 2000 |
| min_samples_split | Min samples to split a node | 2 | 2 - 20 |
| min_samples_leaf | Min samples in leaf node | 1 | 1 - 10 |
| max_features | Same as mtry | √p or p/3 | Grid search |

## OOB (Out-of-Bag) Error

| Concept | Description |
|---|---|
| How it works | Each tree trained on ~63% of data; remaining ~37% = OOB sample |
| OOB error | Error estimated on OOB samples (no separate test set needed) |
| Use | Equivalent to cross-validation; use for model selection |
| When to stop adding trees | When OOB error plateaus |

## Variable Importance

| Method | Description |
|---|---|
| Mean Decrease Accuracy | Drop in accuracy when variable is permuted |
| Mean Decrease Gini | Total decrease in node impurity (Gini) from splits on variable |
| Permutation importance | More reliable; measures actual prediction impact |

## When to Use

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| High-dimensional data (p >> n) | RF | OLS | RF handles multicollinearity naturally |
| Non-linear relationships | RF | OLS | RF captures interactions automatically |
| Need interpretability | OLS | RF | RF is a black box without importance |
| Small n (< 50) | OLS | RF | RF needs more data to train |
| Time series | RF with caution | OLS | RF ignores temporal structure |
| Need prediction intervals | Quantile RF | Point-estimate RF | Standard RF gives point predictions only |

## RF vs OLS

| Aspect | OLS | Random Forest |
|---|---|---|
| Assumptions | Linear, homoscedastic, normal | None |
| Interpretability | High (coefficients) | Low (black box) |
| Overfitting | Low (high bias) | Low (low variance, bagging) |
| Non-linear | Must specify manually | Captures automatically |
| Interactions | Must specify manually | Captures automatically |
| Small samples | Better | Worse |
| Large samples | Good | Better (if non-linear) |

## Cross-references

- See also: `friedman-2001-gbm.md` (boosting vs bagging comparison)
- See also: `shap-lime.md` (interpreting RF predictions)
- See also: `cross-validation.md` (OOB as alternative to CV)
