# LAECON — Econometrics + interpretable ML capability for LAOS

Regression, GLM, causal inference, timeseries, and diagnostic reporting.

## MCP Tools

- `health` — server health check
- `list_supported_operations` — capability catalog
- `train_regression` — OLS, GLM, Logit, Probit models
- `train_classifier` — classification models
- `validate_assumptions` — normality, homoscedasticity, multicollinearity
- `interpret_model` — coefficients, SHAP, p-values
- `evaluate_model` — R², AIC, BIC, confusion matrix
- `predict` — generate predictions from trained models
- `export_diagnostic_report` — full model diagnostic report

## Setup

```bash
uv sync
uv run laecon-server
```
