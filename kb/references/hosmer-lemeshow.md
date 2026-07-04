# Hosmer & Lemeshow — Applied Logistic Regression (2nd ed, 2000)

**Source:** Hosmer, D.W. & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley.
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Metric/Threshold | Value | Interpretation | Source |
|---|---|---|---|
| H-L test p > 0.05 | Calibrated | Model fits data well (fail to reject) | H&L Ch. 5 |
| H-L test p < 0.05 | Miscalibrated | Poor fit, model needs improvement | H&L Ch. 5 |
| AUC > 0.7 | Acceptable discrimination | Better than chance | H&L Ch. 5 |
| AUC > 0.8 | Good discrimination | Clinically useful | H&L Ch. 5 |
| AUC > 0.9 | Excellent discrimination | Near-perfect separation | H&L Ch. 5 |
| AUC ≈ 0.5 | No discrimination | Model = random guess | H&L Ch. 5 |
| McFadden R² > 0.2 | Good fit | Pseudo-R² for logit models | McFadden (1974) |
| McFadden R² > 0.4 | Very good fit | Strong explanatory power | McFadden (1974) |

## Goodness-of-Fit Tests

| Test | What it measures | Ideal result | Concerning result |
|---|---|---|---|
| Hosmer-Lemeshow | Calibration (predicted vs observed) | p > 0.05 | p < 0.05 |
| AUC-ROC | Discrimination ability | > 0.8 | < 0.7 |
| Classification table | Correct predictions at threshold | High accuracy + balanced sensitivity/specificity | High accuracy with imbalanced classes |
| McFadden R² | Pseudo-R² (log-likelihood based) | > 0.2 | < 0.1 |
| Brier score | Calibration + discrimination | < 0.25 (lower = better) | > 0.25 |

## When to Use

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| Binary DV (0/1) | Logistic regression | OLS (linear probability) | DV bounded [0,1], OLS can predict outside bounds |
| Ordinal DV | Ordered logit/probit | Binary logit | Multiple ordered categories |
| Continuous DV | OLS | Logit/probit | Continuous outcome doesn't need link function |
| Small sample (< 100) | Caution with H-L | H-L is unreliable | Deciles need adequate n per group |
| Large sample (> 5000) | Use AUC primarily | H-L too sensitive | H-L rejects with trivially small misfit |

## Decision Rules

1. **Is the outcome binary?** → Logit or probit (not OLS).
2. **Run H-L test** → p > 0.05 means calibration is adequate.
3. **Check AUC** → > 0.7 acceptable, > 0.8 good, > 0.9 excellent.
4. **Logit vs Probit?** → Logit gives odds ratios (more interpretable). Probit gives marginal effects in probit-scale (less intuitive). Both converge with large n.
5. **McFadden R² > 0.2?** → Model has reasonable explanatory power.
6. **Check multicollinearity** → Same VIF rules as OLS apply.

## Logit vs Probit

| Feature | Logit | Probit |
|---|---|---|
| Link function | Logistic (sech²) | Normal CDF |
| OR interpretation | Direct (exp(β)) | Indirect (must convert) |
| Tail behavior | Heavier tails | Lighter tails |
| Marginal effects | β·p(1-p) | β·φ(Xβ) |
| Large n | Converges | Converges |
| Preferred when | Need OR | Need probit-scale effects |

## Cross-references

- See also: `gujarati-porter.md` (continuous DV, OLS diagnostics)
- See also: `long-1997.md` (ordered logit, AME/MER/MEM)
- See also: `model-selection.md` (binary DV → logit/probit decision)
