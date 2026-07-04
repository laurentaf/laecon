# Model Selection — Decision Flowchart

**Source:** Synthesis from Gujarati (2009), Long (1997), Hastie et al. (2009), Breiman (2001), Friedman (2001).
**Used in LAECON:** Constitution Art. 10 §3

---

## Key Values

| Decision Point | Rule | Source |
|---|---|---|
| Continuous DV → OLS | DV is continuous, linear relationship | Gujarati Ch. 1-9 |
| Binary DV → Logit/Probit | DV is 0/1 | Long Ch. 3; Hosmer & Lemeshow |
| Ordinal DV (3+) → Ordered Logit | Categories have natural order | Long Ch. 4 |
| Unordered categorical (3+) → Multinomial Logit | Categories are unordered | Long Ch. 4 |
| Count DV → Poisson / NB | DV is non-negative integer | Cameron & Trivedi (2013) |
| n < 200 and linear → OLS | Small n, simple relationships | Gujarati |
| n > 10,000 and non-linear → GBM | Large n, complex patterns | Friedman 2001 |
| p >> n → RF or penalized | High-dimensional | Breiman 2001; Tibshirani (1996) |
| Need interpretability → OLS or Logit | Stakeholder communication | Long 1997 |
| Need prediction → GBM/RF | Out-of-sample performance | Friedman 2001 |

## Decision Flowchart

```
1. What is the outcome type?
   ├── Continuous → Step 2
   ├── Binary (0/1) → Logit or Probit (Step 5)
   ├── Ordinal (3+ ordered categories) → Ordered Logit (Step 5)
   ├── Unordered categorical (3+) → Multinomial Logit (Step 5)
   └── Count (non-negative integer) → Poisson / Negative Binomial (Step 6)

2. Is the relationship linear?
   ├── Yes → OLS (Step 3)
   └── No → Consider RF or GBM (Step 4)

3. OLS diagnostics (Gujarati Ch. 10-13):
   ├── VIF > 10? → Address multicollinearity
   ├── DW far from 2? → Address autocorrelation
   ├── BP p < 0.05? → Use robust SE
   ├── JB p < 0.05? → Consider robust inference
   └── All pass? → Report coefficients + SE + CI

4. Need interpretability?
   ├── Yes → OLS or Logit (accept linearity assumption)
   └── No → RF or GBM + SHAP for post-hoc interpretation

5. Logit/Probit choice:
   ├── Need odds ratios? → Logit (OR = exp(β))
   ├── Need probit-scale effects? → Probit
   ├── Ordered categories? → Ordered logit + Brant test for PO
   └── Large n? → Both converge; pick for interpretability

6. Count model:
   ├── Mean ≈ Variance? → Poisson
   ├── Mean << Variance (overdispersion)? → Negative Binomial
   └── Zero-inflated? → Zero-inflated Poisson/NB
```

## Sample Size Guidelines

| Model | Minimum n | Recommended n | Source |
|---|---|---|---|
| OLS | 10 × number of predictors | 20 × number of predictors | Green (1991) |
| Logit | 10 events per variable (EPV) | 20 EPV | Peduzzi et al. (1996) |
| Random Forest | 100+ | 500+ | Breiman (2001) |
| GBM | 1000+ | 5000+ | Friedman (2001) |
| Ordered Logit | 50 per category | 100 per category | Long (1997) |
| Multinomial Logit | 50 per category | 100 per category | Long (1997) |

## Green (1991) Rules of Thumb

| Criterion | Formula | Notes |
|---|---|---|
| Minimum n for OLS | n ≥ 50 + 8k | k = number of predictors |
| Preferred n for OLS | n ≥ 50 + 8k + (number of interactions) | Include interaction terms |
| Minimum EPV for logit | n ≥ 10 × (events in smallest category) | Events per variable |
| Preferred EPV for logit | n ≥ 20 × (events in smallest category) | More robust |

## When to Use Each Model

| Situation | Use this | Don't use this | Why |
|---|---|---|---|
| Simple linear, interpretable | OLS | GBM | OLS coefficients are directly interpretable |
| Non-linear, interpretable | Logit/Probit + AME | OLS | Link function captures non-linearity |
| High-dimensional, non-linear | RF or GBM | OLS | OLS fails with p >> n |
| Need probability calibration | Logistic regression | RF (without calibration) | Logit outputs are probabilities by design |
| Time series | ARIMA / VAR | OLS | Temporal dependence |
| Causal inference | IV, DiD, PSM | Correlation models | Need exogeneity assumptions |
| Panel data | FE / RE | Pooled OLS | Unobserved heterogeneity |

## Cross-references

- See also: `gujarati-porter.md` (OLS diagnostics)
- See also: `hosmer-lemeshow.md` (logit goodness-of-fit)
- See also: `long-1997.md` (categorical models, AME)
- See also: `breiman-2001-rf.md` (when RF beats OLS)
- See also: `friedman-2001-gbm.md` (when GBM beats OLS)
- See also: `cross-validation.md` (out-of-sample validation)
- See also: `shap-lime.md` (interpreting black-box models)
