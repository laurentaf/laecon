"""Diagnostic report generation — export model summary, diagnostics, plots.

Output path deterministic per AE-3:
  - artifacts/reports/<model_id>/diagnostic_report.md (or .html)
  - artifacts/reports/<model_id>/plots/ (PNG plots)

Plots: residuals vs fitted, QQ, scale-location, leverage, partial dependence.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import seaborn as sns
import statsmodels.api as sm

from laecon.models.registry import plots_path, report_path


def export_diagnostic_report(
    model_id: str,
    model_obj: Any,
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    format_type: str = "markdown",
    diagnostics: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    interpretation: dict[str, Any] | None = None,
    registry_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and export a diagnostic report.

    Args:
        model_id: Model identifier.
        model_obj: Fitted statsmodels result object.
        df: Original training data.
        features: Feature names.
        target: Target column name.
        format_type: "markdown" or "html".
        diagnostics: Pre-computed diagnostics from validate_assumptions.
        evaluation: Pre-computed evaluation metrics.
        interpretation: Pre-computed interpretation results.
        registry_data: Registry metadata.

    Returns:
        dict with:
          - status, model_id, format
          - report_path: str (absolute path to report file)
          - plots: list of plot file paths
    """
    format_type = format_type.lower()
    if format_type not in ("markdown", "html"):
        raise ValueError(f"Unsupported format: {format_type}. Use 'markdown' or 'html'.")

    # Detect model type
    model_class = type(model_obj).__name__
    has_rsquared = hasattr(model_obj, "rsquared")
    has_llnull = hasattr(model_obj, "llnull")
    is_ols = has_rsquared or "RegressionResults" in model_class or "OLS" in model_class
    is_logit = hasattr(model_obj, "model") and "Logit" in type(model_obj.model).__name__
    is_probit = hasattr(model_obj, "model") and "Probit" in type(model_obj.model).__name__
    is_binary = is_logit or is_probit or has_llnull
    is_ordered = hasattr(model_obj, "model") and "OrderedModel" in type(model_obj.model).__name__

    # Ensure output directories
    rpath = report_path(model_id)
    ppath = plots_path(model_id)

    # --- Generate plots ---
    plot_files: list[str] = []

    if is_ols:
        try:
            plot_files.append(
                _save_plot(ppath, model_id, "residuals_vs_fitted", _plot_resid_vs_fitted, model_obj)
            )
        except Exception as e:
            plot_files.append(f"Error: residuals_vs_fitted - {e}")

        try:
            plot_files.append(
                _save_plot(ppath, model_id, "qq_plot", _plot_qq, model_obj)
            )
        except Exception as e:
            plot_files.append(f"Error: qq_plot - {e}")

        try:
            plot_files.append(
                _save_plot(ppath, model_id, "scale_location", _plot_scale_location, model_obj)
            )
        except Exception as e:
            plot_files.append(f"Error: scale_location - {e}")

        try:
            plot_files.append(
                _save_plot(ppath, model_id, "leverage", _plot_leverage, model_obj)
            )
        except Exception as e:
            plot_files.append(f"Error: leverage - {e}")

    if is_binary or is_ols:
        try:
            plot_files.append(
                _save_plot(ppath, model_id, "residuals_hist", _plot_residuals_hist, model_obj)
            )
        except Exception as e:
            plot_files.append(f"Error: residuals_hist - {e}")

    if is_binary:
        try:
            plot_files.append(
                _save_plot(ppath, model_id, "roc_curve", _plot_roc, model_obj)
            )
        except Exception as e:
            plot_files.append(f"Error: roc_curve - {e}")

    # --- Build report content ---
    if format_type == "markdown":
        content = _build_markdown_report(
            model_id, model_class, model_obj, features, target,
            diagnostics, evaluation, interpretation, registry_data, plot_files
        )
        ext = "md"
    else:
        content = _build_html_report(
            model_id, model_class, model_obj, features, target,
            diagnostics, evaluation, interpretation, registry_data, plot_files
        )
        ext = "html"

    # Write report file
    report_file = rpath / f"diagnostic_report.{ext}"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "status": "ok",
        "model_id": model_id,
        "format": format_type,
        "report_path": str(report_file),
        "plots": plot_files,
    }


# === Plot functions ===

def _save_plot(ppath: Path, model_id: str, name: str, plot_fn, *args) -> str:
    """Generate a plot, save it, and return the file path."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_fn(*args, ax=ax)
    plt.tight_layout()
    filepath = ppath / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)


def _plot_resid_vs_fitted(model_obj, ax):
    """Residuals vs fitted values plot."""
    fitted = model_obj.fittedvalues
    resid = model_obj.resid
    ax.scatter(fitted, resid, alpha=0.6, s=30)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    # Lowess smoother
    try:
        lowess = sm.nonparametric.lowess
        smoothed = lowess(resid, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=1.5)
    except Exception:
        pass


def _plot_qq(model_obj, ax):
    """Q-Q plot of residuals."""
    resid = model_obj.resid
    sm.qqplot(resid, line="45", ax=ax, fit=True)
    ax.set_title("Normal Q-Q")


def _plot_scale_location(model_obj, ax):
    """Scale-location plot (sqrt(|residuals|) vs fitted)."""
    fitted = model_obj.fittedvalues
    std_resid = np.sqrt(np.abs(model_obj.resid_pearson))
    ax.scatter(fitted, std_resid, alpha=0.6, s=30)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("√|Standardized residuals|")
    ax.set_title("Scale-Location")
    try:
        lowess = sm.nonparametric.lowess
        smoothed = lowess(std_resid, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=1.5)
    except Exception:
        pass


def _plot_leverage(model_obj, ax):
    """Leverage vs residuals squared plot."""
    try:
        influence = model_obj.get_influence()
        leverage = influence.hat_matrix_diag
        resid_student = influence.resid_studentized_external
    except Exception:
        leverage = model_obj.get_influence().hat_matrix_diag
        resid_student = model_obj.resid / np.sqrt(model_obj.mse_resid)

    ax.scatter(leverage, resid_student ** 2, alpha=0.6, s=30)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentized residuals²")
    ax.set_title("Leverage vs Residuals")
    # Cook's distance contour lines
    try:
        n = len(leverage)
        p = model_obj.df_model + 1
        cook_levels = [0.5, 1.0]
        for c in cook_levels:
            x_arr = np.linspace(max(min(leverage), 0.01), max(leverage) * 1.1, 50)
            # Cook's D contour: y = c * (1 - h) / h * (p / (n - p))
            y_arr = c * (1 - x_arr) / x_arr * (p / max(n - p, 1))
            y_arr = np.clip(y_arr, 0, 10)
            ax.plot(x_arr, y_arr, linestyle="--", color="gray", alpha=0.5, label=f"Cook's D={c}")
    except Exception:
        pass


def _plot_residuals_hist(model_obj, ax):
    """Histogram of residuals with normal curve overlay."""
    resid = model_obj.resid
    ax.hist(resid, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    # Normal curve
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy_stats.norm.pdf(x, np.mean(resid), np.std(resid))
    ax.plot(x, p, "r-", linewidth=2, label="Normal")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residuals Distribution")
    ax.legend()


def _plot_roc(model_obj, ax):
    """ROC curve for binary classifiers."""
    try:
        y_true = model_obj.model.endog
        y_pred = model_obj.predict()
        from sklearn.metrics import roc_curve as sk_roc
        fpr, tpr, _ = sk_roc(y_true, y_pred)
        from sklearn.metrics import auc as sk_auc
        auc_score = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_score:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    except Exception as e:
        ax.text(0.5, 0.5, f"ROC error: {e}", ha="center", va="center")


# === Report builders ===

def _build_markdown_report(
    model_id: str,
    model_class: str,
    model_obj: Any,
    features: list[str] | None,
    target: str | None,
    diagnostics: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    interpretation: dict[str, Any] | None,
    registry_data: dict[str, Any] | None,
    plot_files: list[str],
) -> str:
    """Build a markdown diagnostic report."""
    lines: list[str] = []
    lines.append(f"# Diagnostic Report — `{model_id}`")
    lines.append("")
    lines.append(f"- **Model class:** `{model_class}`")
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"- **Features:** {', '.join(features) if features else 'N/A'}")
    lines.append(f"- **Target:** {target or 'N/A'}")
    lines.append("")

    # --- Model Summary ---
    lines.append("## Model Summary")
    lines.append("")
    if hasattr(model_obj, "summary"):
        try:
            summary_text = model_obj.summary().as_text()
            lines.append("```")
            lines.append(summary_text)
            lines.append("```")
        except Exception:
            lines.append("*Summary unavailable.*")
    else:
        lines.append("*No standard summary available.*")
    lines.append("")

    # --- Coefficients ---
    lines.append("## Coefficients")
    lines.append("")
    try:
        coef_df = pd.DataFrame({
            "coef": model_obj.params,
            "std_err": model_obj.bse,
            "z": model_obj.tvalues,
            "P>|z|": model_obj.pvalues,
            "[0.025": model_obj.conf_int().iloc[:, 0],
            "0.975]": model_obj.conf_int().iloc[:, 1],
        })
        lines.append(coef_df.to_markdown())
    except Exception:
        lines.append("*Coefficient extraction failed.*")
    lines.append("")

    # --- Diagnostics ---
    if diagnostics:
        lines.append("## Assumption Diagnostics")
        lines.append("")
        lines.append(f"**Overall:** {diagnostics.get('summary', {}).get('overall', 'N/A')}")
        lines.append("")
        lines.append(f"- PASS: {diagnostics.get('summary', {}).get('pass_count', 0)}")
        lines.append(f"- WARN: {diagnostics.get('summary', {}).get('warn_count', 0)}")
        lines.append(f"- FAIL: {diagnostics.get('summary', {}).get('fail_count', 0)}")
        lines.append("")

        for group_name, group_data in diagnostics.get("tests", {}).items():
            lines.append(f"### {group_name.replace('_', ' ').title()}")
            lines.append("")
            for test_name, test_result in group_data.items():
                if isinstance(test_result, dict):
                    flag = test_result.get("flag", "?")
                    lines.append(f"- **{test_name}**: `{flag}`")
                    if "statistic" in test_result:
                        lines.append(f"  - Statistic: {test_result['statistic']}")
                    if "p_value" in test_result:
                        lines.append(f"  - p-value: {test_result['p_value']}")
            lines.append("")

    # --- Evaluation ---
    if evaluation:
        lines.append("## Model Performance")
        lines.append("")
        ins = evaluation.get("in_sample", {})
        if ins:
            lines.append("### In-Sample")
            lines.append("")
            for k, v in ins.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        oos = evaluation.get("out_of_sample", {})
        if oos and oos.get("error") is None:
            lines.append("### Out-of-Sample")
            lines.append("")
            for k, v in oos.items():
                if k != "error":
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        baseline = evaluation.get("baseline_comparison", {})
        if baseline:
            lines.append("### Baseline Comparison (Art. 3.5)")
            lines.append("")
            for k, v in baseline.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

    # --- Interpretation ---
    if interpretation:
        lines.append("## Interpretation")
        lines.append("")
        ame = interpretation.get("average_marginal_effects", [])
        if ame and "error" not in ame[0]:
            lines.append("### Average Marginal Effects")
            lines.append("")
            ame_df = pd.DataFrame(ame)
            lines.append(ame_df.to_markdown())
            lines.append("")

        odds = interpretation.get("odds_ratios", [])
        if odds and "error" not in odds[0]:
            lines.append("### Odds Ratios")
            lines.append("")
            odds_df = pd.DataFrame(odds)
            lines.append(odds_df.to_markdown())
            lines.append("")

    # --- Plots ---
    if plot_files:
        lines.append("## Diagnostic Plots")
        lines.append("")
        for pf in plot_files:
            if pf.startswith("Error"):
                lines.append(f"- {pf}")
            else:
                rel = os.path.relpath(pf, report_path(model_id))
                lines.append(f"![{Path(pf).stem}]({rel})")
        lines.append("")

    # --- Natural language interpretation ---
    lines.append("## Natural Language Summary")
    lines.append("")
    nl_lines = _natural_language_summary(
        model_id, model_class, model_obj, diagnostics, evaluation
    )
    lines.extend(nl_lines)
    lines.append("")

    return "\n".join(lines)


def _build_html_report(*args) -> str:
    """Build an HTML diagnostic report."""
    # For now, convert markdown to basic HTML
    md_content = _build_markdown_report(*args)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Diagnostic Report — {args[0]}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 960px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ color: #2c3e50; margin-top: 1.5em; }}
h3 {{ color: #34495e; }}
pre {{ background: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; }}
code {{ background: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
th {{ background: #f8f8f8; text-align: center; }}
img {{ max-width: 100%; margin: 1em 0; border: 1px solid #eee; border-radius: 4px; }}
.flag-PASS {{ color: green; font-weight: bold; }}
.flag-WARN {{ color: orange; font-weight: bold; }}
.flag-FAIL {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
{md_content}
</body>
</html>"""
    return html


def _natural_language_summary(
    model_id: str,
    model_class: str,
    model_obj: Any,
    diagnostics: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
) -> list[str]:
    """Generate natural language interpretation of model results."""
    lines: list[str] = []
    lines.append(f"The model **{model_id}** is a **{model_class}** estimator.")

    # Sample size
    nobs = getattr(model_obj, "nobs", 0)
    lines.append(f"It was estimated on **{nobs}** observations.")
    if nobs < 200:
        lines.append("⚠ The sample size is relatively small; consider interpreting results with caution.")
    elif nobs > 10000:
        lines.append("✓ The large sample size supports reliable inference.")

    # R-squared
    if hasattr(model_obj, "rsquared"):
        r2 = model_obj.rsquared
        lines.append(f"The model explains **{r2*100:.1f}%** of the variance in the target (R² = {r2:.4f}).")
        if r2 < 0.1:
            lines.append("⚠ The explanatory power is low. Consider adding relevant features.")
        elif r2 > 0.7:
            lines.append("✓ The model has strong explanatory power.")

    # Diagnostics
    if diagnostics:
        diag_summary = diagnostics.get("summary", {})
        overall = diag_summary.get("overall", "?")
        fails = diag_summary.get("fail_count", 0)
        warns = diag_summary.get("warn_count", 0)

        if overall == "PASS":
            lines.append("✓ All diagnostic tests passed. The model satisfies classical assumptions.")
        elif overall == "WARN":
            lines.append(f"⚠ {warns} assumption test(s) returned warnings. The model may have minor violations.")
        elif overall == "FAIL":
            lines.append(f"❌ {fails} assumption test(s) failed. The model violates classical assumptions.")
            lines.append("Consider robust standard errors, data transformations, or alternative specifications.")

        # Specific guidance
        het = diagnostics.get("tests", {}).get("heteroscedasticity", {})
        bp = het.get("breusch_pagan", {})
        if bp.get("flag") == "FAIL":
            lines.append("  - Heteroscedasticity detected: use Huber-White robust standard errors.")

        auto = diagnostics.get("tests", {}).get("autocorrelation", {})
        dw = auto.get("durbin_watson", {})
        if dw.get("flag") == "FAIL":
            lines.append("  - Autocorrelation detected: consider HAC standard errors or a time-series model.")

    # Evaluation
    if evaluation:
        oos = evaluation.get("out_of_sample", {})
        oos_auc = oos.get("auc")
        if oos_auc is not None:
            lines.append(f"Out-of-sample AUC is **{oos_auc:.4f}**.")
            if oos_auc > 0.8:
                lines.append("✓ The model has good discriminative ability.")
            elif oos_auc < 0.6:
                lines.append("⚠ The model's discriminative ability is limited.")

    return lines
