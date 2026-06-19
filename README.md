<!--
  LAECON — Econometrics + interpretable ML capability for LAOS
  README.md  |  github.com/laurentaf/laecon
  Brand: Econometrics is the spine. ML is the muscle.
  Tone: Precise, confident, academic but accessible.
  Adapted from the LAOS README (20/20) inline-HTML pattern.
-->

<div align="center" style="margin-top:48px;margin-bottom:24px;">

<!-- Emblem: function curve + confidence band -->
<svg width="72" height="72" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:4px;">
  <rect x="4" y="4" width="56" height="56" rx="12" stroke="#3B1F5E" stroke-width="1.5" fill="none" opacity="0.3"/>
  <path d="M12 48 Q20 36 28 40 Q36 44 44 28 Q52 12 56 16" stroke="#3B1F5E" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="M12 48 Q20 36 28 40 Q36 44 44 28 Q52 12 56 16" stroke="#3B1F5E" stroke-width="6" fill="none" opacity="0.12" stroke-linecap="round"/>
  <circle cx="44" cy="28" r="3" fill="#C8A951" opacity="0.8"/>
  <circle cx="28" cy="40" r="3" fill="#C8A951" opacity="0.8"/>
</svg>

<br/>

# LAECON
### Econometrics &bull; Interpretable ML &bull; Causal Inference

<p style="margin:12px 0;">
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.11-3776AB?style=flat&logo=python&logoColor=fff" alt="Python 3.11+"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Status-STABLE-00b894?style=flat" alt="Status: STABLE"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-31c754?style=flat" alt="License: MIT"/>
  &nbsp;
  <img src="https://img.shields.io/badge/MCP-9%20tools-a29bfe?style=flat" alt="9 MCP tools"/>
  &nbsp;
  <a href="https://github.com/laurentaf/laos"><img src="https://img.shields.io/badge/Ecosystem-LAOS-6c5ce7?style=flat" alt="LAOS Ecosystem"/></a>
</p>

<hr style="width:48px;margin:24px auto;border:none;border-top:2px solid #3B1F5E;opacity:0.3;"/>

</div>

> **Econometrics is the spine. ML is the muscle.**  
> LAECON bridges classical econometrics (OLS, GLM, Logit, Probit) with interpretable machine learning (SHAP, timeseries, causal inference). Every model comes with assumptions validated, coefficients explained, and diagnostics exported — because a black-box prediction is just a guess with a confidence interval.

---

## What LAECON Is

LAECON is the econometrics & interpretable ML capability for the LAOS ecosystem. It exposes 9 MCP tools that cover the full modeling lifecycle — from training regressions and classifiers to validating assumptions, interpreting results, and generating diagnostic reports.

**Why LAECON exists:** Most ML tools optimize for accuracy alone. LAECON optimizes for *understanding*. Inspired by Gujarati & Porter (Basic Econometrics, 5th ed) and Larson & Goungetas (NPS driver analysis), it enforces likelihood-first modeling: every prediction has a paper trail.

---

## Architecture

<div align="center" style="margin:24px 0;">

<table style="border-collapse:separate;border-spacing:0;min-width:560px;">
  <tr>
    <td colspan="3" align="center" style="padding:0 0 8px 0;">
      <div style="display:inline-block;border:1.5px solid #3B1F5E;border-radius:8px;padding:12px 24px;text-align:center;">
        <div style="font-weight:700;font-size:1em;letter-spacing:0.04em;color:#3B1F5E;">LAECON MCP Server</div>
        <div style="font-size:0.8em;opacity:0.5;">Python · statsmodels · scikit-learn · SHAP</div>
      </div>
    </td>
  </tr>
  <tr>
    <td colspan="3" align="center" style="padding:0;">
      <div style="width:1.5px;height:16px;background:#3B1F5E;opacity:0.2;margin:0 auto;"></div>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding:0 8px;width:33%;">
      <div style="border:1px solid #3B1F5E;border-radius:6px;padding:8px 14px;text-align:center;">
        <div style="font-weight:600;font-size:0.85em;letter-spacing:0.02em;">Train</div>
        <div style="font-size:0.7em;opacity:0.5;">OLS · GLM · Logit ·<br/>Probit · Classifiers</div>
      </div>
    </td>
    <td align="center" style="padding:0 8px;width:33%;">
      <div style="border:1px solid #C8A951;border-radius:6px;padding:8px 14px;text-align:center;">
        <div style="font-weight:600;font-size:0.85em;letter-spacing:0.02em;">Validate &amp; Interpret</div>
        <div style="font-size:0.7em;opacity:0.5;">Assumptions · SHAP ·<br/>Coefficients · p-values</div>
      </div>
    </td>
    <td align="center" style="padding:0 8px;width:33%;">
      <div style="border:1px solid #3B1F5E;border-radius:6px;padding:8px 14px;text-align:center;">
        <div style="font-weight:600;font-size:0.85em;letter-spacing:0.02em;">Evaluate &amp; Export</div>
        <div style="font-size:0.7em;opacity:0.5;">R² · AIC · BIC · Confusion<br/>Matrix · Report</div>
      </div>
    </td>
  </tr>
</table>

</div>

---

## MCP Tools

<table style="width:100%;border-collapse:separate;border-spacing:0;font-size:0.9em;">
  <tr style="border-bottom:1px solid #3B1F5E;border-bottom-opacity:0.2;">
    <th align="left" style="padding:10px 14px;font-weight:600;opacity:0.5;width:25%;">Tool</th>
    <th align="left" style="padding:10px 14px;font-weight:600;opacity:0.5;">Description</th>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>train_regression</code></td>
    <td style="padding:8px 14px;opacity:0.7;">OLS, GLM, Logit, Probit — full specification + fit</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>train_classifier</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Classification models with train/test split + metrics</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>validate_assumptions</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Normality, homoscedasticity, multicollinearity (VIF), linearity</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>interpret_model</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Coefficient tables, SHAP values, p-values, marginal effects</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>evaluate_model</code></td>
    <td style="padding:8px 14px;opacity:0.7;">R², adjusted R², AIC, BIC, confusion matrix, ROC-AUC</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>predict</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Generate predictions from trained models with confidence intervals</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>export_diagnostic_report</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Full model diagnostic report — assumptions, coefficients, fit metrics</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>health</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Server liveness probe</td>
  </tr>
  <tr>
    <td style="padding:8px 14px;font-weight:500;"><code>list_supported_operations</code></td>
    <td style="padding:8px 14px;opacity:0.7;">Full capability catalog</td>
  </tr>
</table>

---

## Quick Start

```bash
# Install dependencies
uv sync

# Start the MCP server
uv run laecon-server
```

### Requirements

<table style="font-size:0.85em;border-collapse:separate;border-spacing:0;">
  <tr><td style="padding:6px 12px;font-weight:500;">Python</td><td style="padding:6px 12px;opacity:0.6;">≥ 3.11</td></tr>
  <tr><td style="padding:6px 12px;font-weight:500;">Dependencies</td><td style="padding:6px 12px;opacity:0.6;">statsmodels, scikit-learn, SHAP, pandas, numpy</td></tr>
  <tr><td style="padding:6px 12px;font-weight:500;">Infrastructure</td><td style="padding:6px 12px;opacity:0.6;">LAOS orchestrator with OpenCode CLI</td></tr>
</table>

---

## Example: Regression with Diagnostics

When a dashboard-designer needs to understand *why* a metric moved, LAECON provides the econometric backbone:

```python
# Via MCP call (from orchestrated agent):
result = laecon.train_regression(
    formula="revenue ~ marketing_spend + customer_count + seasonality",
    data=df,
    model_type="ols"
)

# Validate assumptions:
laecon.validate_assumptions(model=result.model)
# → "Breusch-Pagan p=0.32 ✓ | VIF < 5 ✓ | Shapiro-Wilk p=0.21 ✓"

# Get coefficient interpretation:
laecon.interpret_model(model=result.model)
# → "marketing_spend: +0.42 (p<0.001) — significant positive driver"
```

---

## Contributing

| Scope | Path |
|-------|------|
| **New model type** | LACOUNCIL proposal → Conselho majority → implement |
| **Bug fix** | Open an [issue](https://github.com/laurentaf/laecon/issues) or submit a PR |
| **Documentation** | PR with description — no gate required |

See the [LAOS governance model](https://github.com/laurentaf/laos) for full details.

---

## License

<div style="margin:16px 0;">

**MIT** — see [`LICENSE`](https://github.com/laurentaf/laecon/blob/main/LICENSE) for the full text.

Econometrics is the spine. ML is the muscle. The interpretation is yours.

</div>

---

<div align="center" style="margin:36px 0;opacity:0.25;font-size:0.8em;">
<svg width="28" height="28" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:4px;">
  <rect x="4" y="4" width="56" height="56" rx="12" stroke="#3B1F5E" stroke-width="1.5" fill="none" opacity="0.15"/>
  <path d="M12 48 Q20 36 28 40 Q36 44 44 28 Q52 12 56 16" stroke="#3B1F5E" stroke-width="2.5" fill="none" stroke-linecap="round" opacity="0.4"/>
</svg>
<br/>
LAECON — part of the <a href="https://github.com/laurentaf/laos" style="text-decoration:none;">LAOS</a> ecosystem
</div>