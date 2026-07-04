# Gujarati & Porter — Basic Econometrics (5th ed, 2009)

**Source:** Gujarati, D.N. & Porter, D.C. (2009). *Basic Econometrics* (5th ed.). McGraw-Hill.
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Metric/Threshold | Value | Interpretation | Source |
|---|---|---|---|
| R² > 0.7 | High | Strong explanatory power | G&P Ch. 1-2 |
| VIF > 10 | Problem | Serious multicollinearity | G&P Ch. 10 |
| Durbin-Watson ≈ 2 | OK | No autocorrelation | G&P Ch. 12 |
| Breusch-Pagan p < 0.05 | Heteroscedastic | Reject H0 of homoscedasticity | G&P Ch. 11 |
| F-test p < 0.05 | Significant | Model overall is significant | G&P Ch. 7 |
| |t| > 2 | Significant | Coefficient significantly different from 0 | G&P Ch. 7 |
| RESET p < 0.05 | Misspecification | Functional form may be wrong | G&P Ch. 9 |

## OLS Assumptions (Ch. 10-13)

1. **Normality** of residuals
2. **Linearity** in parameters
3. **Constant variance** (homoscedasticity)
4. **No autocorrelation** of residuals
5. **Exogeneity** of regressors (E(u|X) = 0)

## Diagnostic Tests

| Test | H0 | Reject if | Remedy |
|---|---|---|---|
| Breusch-Pagan | Homoscedasticity | p < 0.05 | WLS, robust SE, log-transform |
| Durbin-Watson | No autocorrelation | DW far from 2 (see tables) | GLS, Newey-West SE, lagged DV |
| White | Homoscedasticity | p < 0.05 | Robust HC3 SE |
| Ramsey RESET | Correct specification | p < 0.05 | Add polynomial terms, transform DV |
| Jarque-Bera | Normality of residuals | p < 0.05 | Non-linear transform, robust inference |
| VIF | No multicollinearity | VIF > 10 | Drop vars, ridge, PCA |

## When to Use

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| Continuous DV, linear relationship | OLS (Ch. 1-9) | Logit/probit | DV is continuous |
| Multiple regressors | Multiple OLS (Ch. 7) | Simple OLS | Need to control for confounders |
| Heteroscedastic data | WLS or robust SE (Ch. 11) | Naive OLS t-tests | SEs are biased without correction |
| Time series data | OLS with DW test (Ch. 12) | Cross-section assumptions | Autocorrelation violates independence |
| Multicollinearity suspected | VIF check + remedy (Ch. 10) | Ignore VIF | Coefficients become unstable |

## Decision Rules

1. Run R² and F-test first — does the model explain variance at all?
2. Check VIF — if any VIF > 10, address multicollinearity before interpreting coefficients.
3. Check DW — if DW < 1.5 or > 2.5, suspect autocorrelation.
4. Run Breusch-Pagan or White — if p < 0.05, use robust standard errors.
5. Check residuals — if Jarque-Bera rejects normality, consider robust inference or transformation.
6. Ramsey RESET — if rejects, functional form may be wrong.

## Cross-references

- See also: `hosmer-lemeshow.md` (discrete DV goodness-of-fit)
- See also: `model-selection.md` (when to choose OLS vs alternatives)
- See also: `cross-validation.md` (out-of-sample validation)
