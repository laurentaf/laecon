# SHAP & LIME — Model Interpretability

**SHAP Source:** Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
**LIME Source:** Ribeiro, M.T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Metric/Threshold | Value | Interpretation | Source |
|---|---|---|---|
| SHAP consistency | Guaranteed | Better model → higher SHAP value for feature | Lundberg & Lee 2017 |
| SHAP local accuracy | SHAP value + bias = prediction | Exact decomposition | Lundberg & Lee 2017 |
| SHAP missingness | Sum of absent features = 0 | No contribution if feature absent | Lundberg & Lee 2017 |
| LIME perturbations | ~5000-10000 samples | Sufficient for local surrogate | Ribeiro et al. 2016 |
| LIME kernel width | sqrt(n_features) × 0.75 | Default kernel width | Ribeiro et al. 2016 |

## SHAP Properties (Axioms)

| Property | Definition | Why it matters |
|---|---|---|
| **Local accuracy** | Σ shapleys + bias = prediction | Decomposition is exact |
| **Missingness** | Feature not present → contribution = 0 | No artificial attribution |
| **Consistency** | If model changes to give more to feature, its SHAP doesn't decrease | Fair attribution |

## SHAP Plots

| Plot type | What it shows | When to use |
|---|---|---|
| Summary (bar) | Global feature importance | Quick overview |
| Summary (beeswarm) | Distribution of SHAP values | See direction of effects |
| Dependence | SHAP vs feature value | Understand non-linearity |
| Waterfall | Single prediction decomposition | Explain one observation |
| Force | Single prediction (compact) | Communication with stakeholders |
| Heatmap | Many predictions × features | Batch explanation |

## LIME vs SHAP

| Aspect | LIME | SHAP |
|---|---|---|
| Scope | Local only | Local + Global (via mean SHAP) |
| Speed | Fast | Slower (exact: exponential; approx: sampling) |
| Stability | Unstable (perturbation-sensitive) | Stable (axiom-based) |
| Model-agnostic | Yes | Yes (exact for trees; sampling for others) |
| Theoretical foundation | Local linear approximation | Cooperative game theory |
| When to use | Quick local explanation | Rigorous interpretation |
| Limitation | Unstable across runs | Slow for complex models |

## When to Use

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| Global importance | SHAP summary | LIME | SHAP aggregates naturally |
| Single prediction | Both | — | SHAP waterfall, LIME local |
| Tree model | TreeSHAP (exact, fast) | LIME | TreeSHAP is exact and fast |
| Black-box model | SHAP (sampling) or LIME | — | Both work; SHAP more stable |
| Need stability | SHAP | LIME | SHAP axioms guarantee consistency |
| Need speed | LIME | SHAP (exact) | LIME is faster for quick checks |
| Tabular data | SHAP | LIME | SHAP handles tabular well |

## Decision Rules

1. **Default to SHAP** — it's more theoretically grounded and stable.
2. **Use TreeSHAP** for tree models — exact, fast, no sampling needed.
3. **Use LIME** only for quick local checks or when SHAP is too slow.
4. **SHAP summary plot** for global importance — not just bar chart (use beeswarm to see direction).
5. **SHAP dependence plot** for non-linearity — reveals interaction effects.
6. **Always show bias term** — SHAP values sum to prediction minus bias.
7. **For communication with stakeholders** — use SHAP waterfall or force plot.

## Cross-references

- See also: `breiman-2001-rf.md` (interpreting RF via importance)
- See also: `friedman-2001-gbm.md` (interpreting GBM via SHAP)
- See also: `model-selection.md` (when interpretability is a criterion)
