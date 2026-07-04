"""LAECON MCP Server — Econometrics + interpretable ML.

All 7 M1 tools implemented: train_regression, train_classifier,
validate_assumptions, interpret_model, evaluate_model, predict,
export_diagnostic_report.

Constitution compliance:
  - DR-1: DataFrame empty guards on every data operation
  - AE-1: model_id versioning with registry.json
  - AE-3: deterministic output paths for reports
  - Art. 3.5: baseline comparison in evaluate
  - Art. 3.6: confidence reporting
  - DA-2: anti-black-box (enforced in train_classifier)
"""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from laecon.models.classifier import train_classifier as _train_classifier
from laecon.models.data_utils import check_empty, load_dataset
from laecon.models.diagnostics import validate_assumptions as _validate_assumptions
from laecon.models.evaluation import evaluate_model as _evaluate_model
from laecon.models.interpretation import interpret_model as _interpret_model
from laecon.models.prediction import predict as _predict
from laecon.models.registry import (
    compute_confidence,
    load_model,
    report_path,
    save_model,
)
from laecon.models.reporting import export_diagnostic_report as _export_diagnostic_report
from laecon.models.regression import train_regression as _train_regression

logger = logging.getLogger("laecon.mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

mcp = FastMCP(
    "laecon",
    instructions=(
        "LAECON MCP server — econometrics + interpretable ML with explicit "
        "likelihood. Status: M1 (STABLE track). All 7 core tools are "
        "implemented: train_regression, train_classifier, validate_assumptions, "
        "interpret_model, evaluate_model, predict, export_diagnostic_report. "
        "See ../LAOS/projects/_meta/capability-evolution/laecon.md for the "
        "evolution plan and vinculating conditions."
    ),
)

# Convenience check for latade availability
_LATADE_AVAILABLE: bool | None = None


def _check_latade() -> bool:
    """Check if latade MCP tools are available for DuckDB queries."""
    global _LATADE_AVAILABLE
    if _LATADE_AVAILABLE is not None:
        return _LATADE_AVAILABLE
    try:
        # Attempt to import latade's execute_sql capabilities
        # (we use DuckDB directly in data_utils, but this check is for doc)
        import duckdb  # noqa: F401
        _LATADE_AVAILABLE = True
    except ImportError:
        _LATADE_AVAILABLE = False
    return _LATADE_AVAILABLE


# ========================================================================
# IMPLEMENTED (BASIC) — health & list_supported_operations
# ========================================================================


@mcp.tool()
def health() -> dict[str, Any]:
    """Liveness probe for the LAECON MCP server.

    Returns:
        dict with status, capability, version, status_detail, and tool_status.
    """
    return {
        "status": "ok",
        "capability": "laecon",
        "version": "0.2.0-M1",
        "status_detail": (
            "M1 (STABLE track) — all 7 core tools implemented. "
            "See ../LAOS/projects/_meta/capability-evolution/laecon.md for tracking."
        ),
        "tool_status": {
            "train_regression": "implemented",
            "train_classifier": "implemented",
            "validate_assumptions": "implemented",
            "interpret_model": "implemented",
            "evaluate_model": "implemented",
            "predict": "implemented",
            "export_diagnostic_report": "implemented",
        },
    }


@mcp.tool()
def list_supported_operations() -> dict[str, Any]:
    """Catalog all supported operations and their implementation status.

    Returns:
        dict with operations list and evolution plan.
    """
    ops = [
        {
            "name": "health",
            "status": "implemented",
            "description": "Liveness probe.",
            "planned_in": "BASIC",
        },
        {
            "name": "list_supported_operations",
            "status": "implemented",
            "description": "Catalog of all operations.",
            "planned_in": "BASIC",
        },
        {
            "name": "train_regression",
            "status": "implemented",
            "description": (
                "OLS, GLS, WLS with robust SE. Returns coefficients, SE, "
                "t-stats, p-values, F-stat, R², adj-R², AIC, BIC, log-likelihood."
            ),
            "planned_in": "M1 (STABLE)",
        },
        {
            "name": "train_classifier",
            "status": "implemented",
            "description": (
                "Logit, probit, ordered logit/probit, grouped logit. "
                "Returns coefficients, marginal effects, odds ratios, cutpoints."
            ),
            "planned_in": "M1 (STABLE)",
        },
        {
            "name": "validate_assumptions",
            "status": "implemented",
            "description": (
                "Heteroscedasticity (BP, White), autocorrelation (DW, BG), "
                "multicollinearity (VIF), normality (JB, SW). Returns structured flags."
            ),
            "planned_in": "M1 (STABLE)",
        },
        {
            "name": "interpret_model",
            "status": "implemented",
            "description": (
                "AME, MER, odds ratios, elasticities, partial dependence."
            ),
            "planned_in": "M1 (STABLE)",
        },
        {
            "name": "evaluate_model",
            "status": "implemented",
            "description": (
                "In-sample and out-of-sample metrics with baseline comparison."
            ),
            "planned_in": "M1 (STABLE)",
        },
        {
            "name": "predict",
            "status": "implemented",
            "description": (
                "Apply trained model to new data. Returns predictions + CI."
            ),
            "planned_in": "M1 (STABLE)",
        },
        {
            "name": "export_diagnostic_report",
            "status": "implemented",
            "description": (
                "Markdown/HTML report with coefficients, diagnostics, plots, "
                "and natural language interpretation."
            ),
            "planned_in": "M1 (STABLE)",
        },
    ]
    return {
        "operations": ops,
        "evolution_plan": {
            "current": "M1 (all 7 core tools implemented)",
            "next_milestone": "M2 (+60d) — tree models with SHAP, cross-validation, panel models",
            "deferred": {
                "deep_learning": "post-STABLE if demand emerges",
                "big_data_distribuido": "DuckDB sufficient (Constitution Art. 3.4)",
            },
        },
    }


# ========================================================================
# IMPLEMENTED (M1) — 7 core tools
# ========================================================================


@mcp.tool()
def train_regression(
    data_source: str,
    target: str,
    features: list[str],
    model_id: str,
    source_type: str = "auto",
    algorithm: str = "ols",
    se_type: str = "homoskedastic",
    sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Train a linear regression model (OLS, GLS, WLS) with robust SE.

    Args:
        data_source: DuckDB table name, CSV file path, or database path.
        target: Target column name.
        features: List of feature column names.
        model_id: Unique identifier for the model (AE-1). Must be a
                  non-empty string (alphanumeric, hyphens, underscores).
        source_type: "auto" (try CSV first, then DuckDB), "csv", or "duckdb".
        algorithm: "ols" (default), "wls", or "gls".
        se_type: "homoskedastic" (default), "huber-white", or "hac".
        sql_query: Optional SQL query for DuckDB source.
        db_path: Optional DuckDB file path.

    Returns:
        dict with:
          - status, model_id, algorithm, se_type
          - coefficients: list of {name, coefficient, se, t_stat, p_value, ci_lower, ci_upper}
          - metrics: {r_squared, adj_r_squared, f_stat, f_pvalue, aic, bic, log_likelihood, n_obs, n_features}
          - confidence: float (0-1)
          - registry: path info for saved artifacts
          - warnings: list[str]

    Raises:
        ValueError: If data empty, columns missing, or invalid params.
    """
    try:
        df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
    except Exception as exc:
        return {"status": "error", "error": f"Data loading failed: {exc}"}

    try:
        result = _train_regression(
            df=df,
            target=target,
            features=features,
            model_id=model_id,
            algorithm=algorithm,
            se_type=se_type,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Training failed: {exc}"}

    # Persist model to registry (AE-1)
    try:
        fitted = result.get("_fitted")
        metadata = {
            "algorithm": algorithm,
            "features": features,
            "target": target,
            "metrics": result.get("metrics", {}),
            "params": {"se_type": se_type},
            "n_obs": result.get("metrics", {}).get("n_obs", 0),
        }
        registry_info = save_model(model_id, fitted, metadata)
    except Exception as exc:
        registry_info = {"error": f"Registry save failed: {exc}"}

    # Clean internal key
    result.pop("_fitted", None)

    # Confidence with registry
    confidence = result.get("confidence", 0.5)
    result["confidence"] = confidence
    result["registry"] = registry_info

    return result


@mcp.tool()
def train_classifier(
    data_source: str,
    target: str,
    features: list[str],
    model_id: str,
    source_type: str = "auto",
    family: str = "logit",
    sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Train a classifier (logit, probit, ordered, grouped).

    Args:
        data_source: DuckDB table name or CSV file path.
        target: Target column name.
        features: List of feature column names.
        model_id: Unique model identifier (AE-1).
        source_type: "auto", "csv", or "duckdb".
        family: "logit" (default), "probit", "ordered_logit",
                "ordered_probit", "grouped_logit".
        sql_query: Optional SQL query for DuckDB source.
        db_path: Optional DuckDB file path.

    Returns:
        dict with coefficients, marginal_effects, odds_ratios, cutpoints,
        metrics, confidence, and registry info.

    Raises:
        ValueError: If family == "grouped_logit" without target being categorical.
    """
    try:
        df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
    except Exception as exc:
        return {"status": "error", "error": f"Data loading failed: {exc}"}

    try:
        result = _train_classifier(
            df=df,
            target=target,
            features=features,
            model_id=model_id,
            family=family,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Training failed: {exc}"}

    # Persist model (AE-1)
    try:
        fitted = result.get("_fitted")
        metadata = {
            "algorithm": family,
            "features": features,
            "target": target,
            "metrics": result.get("metrics", {}),
            "params": {"family": family},
            "n_obs": result.get("metrics", {}).get("n_obs", 0),
        }
        registry_info = save_model(model_id, fitted, metadata)
    except Exception as exc:
        registry_info = {"error": f"Registry save failed: {exc}"}

    result.pop("_fitted", None)
    result["registry"] = registry_info

    return result


@mcp.tool()
def validate_assumptions(
    model_id: str,
    data_source: str | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    source_type: str = "auto",
    n_lags: int = 1,
    sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Validate econometric assumptions for a trained model.

    Tests: heteroscedasticity (Breusch-Pagan, White), autocorrelation
    (Durbin-Watson, Breusch-Godfrey), multicollinearity (VIF), normality
    (Jarque-Bera, Shapiro-Wilk).

    Returns structured PASS/WARN/FAIL flags per test, ideal for n8n
    IF nodes (AE-2).

    Args:
        model_id: Trained model identifier (AE-1).
        data_source: Optional data source to reload data for diagnostics.
                     If omitted, tries to load from the model's registry.
        features: Feature column names (required unless loaded from registry).
        target: Target column (required unless from registry).
        source_type: "auto", "csv", or "duckdb".
        n_lags: Number of lags for Breusch-Godfrey test (default: 1).
        sql_query: Optional SQL query.
        db_path: Optional DuckDB file path.

    Returns:
        dict with per-test results and summary.
    """
    # Try to load model
    try:
        model_obj, registry = load_model(model_id)
    except FileNotFoundError:
        return {
            "status": "error",
            "error": f"Model '{model_id}' not found. Train it first.",
        }

    # Get features from registry if not provided
    if features is None:
        features = registry.get("features", [])
    if target is None:
        target = registry.get("target", "")

    # Load data if needed
    exog = None
    resid = None
    try:
        if data_source:
            df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
            # Reconstruct exog and resid for diagnostics
            if features and target in df.columns:
                model_df = df[[target] + features].dropna()
                if len(model_df) > 0:
                    import statsmodels.api as sm
                    X = model_df[features].values.astype(float)
                    X = sm.add_constant(X, has_constant="add")
                    y = model_df[target].values.astype(float)
                    y_pred = model_obj.predict(X)
                    resid = y - y_pred
                    exog = X
    except Exception as exc:
        # If data reload fails, try to extract from model object
        try:
            resid = model_obj.resid
        except Exception:
            pass
        if resid is None:
            return {
                "status": "error",
                "error": (
                    f"Cannot run diagnostics without data. "
                    f"Provide data_source or ensure the model object has residuals. "
                    f"Detail: {exc}"
                ),
            }

    # Run diagnostics
    try:
        result = _validate_assumptions(
            model_id=model_id,
            exog=exog,
            resid=resid,
            feature_names=features,
            n_lags=n_lags,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Diagnostics failed: {exc}"}

    return result


@mcp.tool()
def interpret_model(
    model_id: str,
    data_source: str | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    source_type: str = "auto",
    pdp_feature: str | None = None,
    sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Interpret a trained model (marginal effects, odds ratios, PDP).

    Args:
        model_id: Trained model identifier (AE-1).
        data_source: Optional data to compute interpretations (uses
                     training data from model registry if omitted).
        features: Feature column names.
        target: Target column name.
        source_type: "auto", "csv", or "duckdb".
        pdp_feature: Optional feature name for partial dependence.
        sql_query: Optional SQL query.
        db_path: Optional DuckDB file path.

    Returns:
        dict with AME, MER, odds ratios, elasticities, partial dependence.
    """
    try:
        model_obj, registry = load_model(model_id)
    except FileNotFoundError:
        return {"status": "error", "error": f"Model '{model_id}' not found."}

    if features is None:
        features = registry.get("features", [])
    if target is None:
        target = registry.get("target", "")

    df = None
    if data_source:
        try:
            df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
        except Exception as exc:
            return {"status": "error", "error": f"Data loading failed: {exc}"}

    try:
        result = _interpret_model(
            model_id=model_id,
            model_obj=model_obj,
            df=df,
            features=features,
            target=target,
            pdp_feature=pdp_feature,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Interpretation failed: {exc}"}

    return result


@mcp.tool()
def evaluate_model(
    model_id: str,
    data_source: str | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    source_type: str = "auto",
    test_data_source: str | None = None,
    test_source_type: str | None = None,
    sql_query: str | None = None,
    test_sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate model performance (in-sample and out-of-sample).

    Args:
        model_id: Trained model identifier (AE-1).
        data_source: Optional training data (for in-sample metrics).
        features: Feature column names.
        target: Target column name.
        source_type: "auto", "csv", or "duckdb".
        test_data_source: Optional test data for out-of-sample metrics.
        test_source_type: Source type for test data.
        sql_query: Optional SQL query for training data.
        test_sql_query: Optional SQL query for test data.
        db_path: Optional DuckDB file path.

    Returns:
        dict with in_sample, out_of_sample, baseline_comparison, confidence.
    """
    try:
        model_obj, registry = load_model(model_id)
    except FileNotFoundError:
        return {"status": "error", "error": f"Model '{model_id}' not found."}

    if features is None:
        features = registry.get("features", [])
    if target is None:
        target = registry.get("target", "")

    # Load training data
    X_train = None
    y_train = None
    if data_source:
        try:
            df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
            if features and target in df.columns:
                train_df = df[[target] + features].dropna()
                if len(train_df) > 0:
                    import statsmodels.api as sm
                    X_train = sm.add_constant(train_df[features].values.astype(float), has_constant="add")
                    y_train = train_df[target].values.astype(float)
        except Exception as exc:
            return {"status": "error", "error": f"Training data loading failed: {exc}"}

    # Load test data
    X_test = None
    y_test = None
    df_test = None
    if test_data_source:
        try:
            ts = test_source_type or source_type
            df_test = load_dataset(test_data_source, source_type=ts, sql_query=test_sql_query, db_path=db_path)
        except Exception as exc:
            return {"status": "error", "error": f"Test data loading failed: {exc}"}

    try:
        result = _evaluate_model(
            model_id=model_id,
            model_obj=model_obj,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            df_test=df_test,
            target=target,
            features=features,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Evaluation failed: {exc}"}

    return result


@mcp.tool()
def predict(
    model_id: str,
    data_source: str,
    features: list[str] | None = None,
    source_type: str = "auto",
    return_ci: bool = True,
    sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Apply a trained model to new data.

    Args:
        model_id: Trained model identifier (AE-1).
        data_source: New data (DuckDB table or CSV path).
        features: Feature column names (defaults to training features).
        source_type: "auto", "csv", or "duckdb".
        return_ci: Whether to return confidence intervals (default: True).
        sql_query: Optional SQL query.
        db_path: Optional DuckDB file path.

    Returns:
        dict with predictions, confidence_intervals, feature_values.
    """
    try:
        model_obj, registry = load_model(model_id)
    except FileNotFoundError:
        return {"status": "error", "error": f"Model '{model_id}' not found."}

    if features is None:
        features = registry.get("features", [])
    if not features:
        return {"status": "error", "error": "No features specified and none found in registry."}

    try:
        df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
    except Exception as exc:
        return {"status": "error", "error": f"Data loading failed: {exc}"}

    try:
        result = _predict(
            model_id=model_id,
            model_obj=model_obj,
            df=df,
            features=features,
            return_ci=return_ci,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Prediction failed: {exc}"}

    return result


@mcp.tool()
def export_diagnostic_report(
    model_id: str,
    format_type: str = "markdown",
    data_source: str | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    source_type: str = "auto",
    sql_query: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Export a diagnostic report with plots and interpretation.

    Args:
        model_id: Trained model identifier (AE-1).
        format_type: "markdown" (default) or "html".
        data_source: Optional data source for diagnostics/interpretation.
        features: Feature column names.
        target: Target column name.
        source_type: "auto", "csv", or "duckdb".
        sql_query: Optional SQL query.
        db_path: Optional DuckDB file path.

    Returns:
        dict with report_path, plots list.
    """
    try:
        model_obj, registry = load_model(model_id)
    except FileNotFoundError:
        return {"status": "error", "error": f"Model '{model_id}' not found."}

    if features is None:
        features = registry.get("features", [])
    if target is None:
        target = registry.get("target", "")

    # Load data for plots and diagnostics
    df = None
    if data_source:
        try:
            df = load_dataset(data_source, source_type=source_type, sql_query=sql_query, db_path=db_path)
        except Exception as exc:
            return {"status": "error", "error": f"Data loading failed: {exc}"}

    # Run diagnostics
    diagnostics = None
    try:
        if df is not None and features and target in df.columns:
            model_df = df[[target] + features].dropna()
            if len(model_df) > 0:
                import statsmodels.api as sm
                X = sm.add_constant(model_df[features].values.astype(float), has_constant="add")
                y = model_df[target].values.astype(float)
                y_pred = model_obj.predict(X)
                resid = y - y_pred
                diagnostics = _validate_assumptions(
                    model_id=model_id,
                    exog=X,
                    resid=resid,
                    feature_names=features,
                )
    except Exception:
        try:
            resid = model_obj.resid
            diagnostics = _validate_assumptions(
                model_id=model_id,
                resid=resid,
                feature_names=features,
            )
        except Exception:
            pass

    # Run evaluation
    evaluation = None
    try:
        if df is not None and features and target in df.columns:
            import statsmodels.api as sm
            model_df = df[[target] + features].dropna()
            X_all = sm.add_constant(model_df[features].values.astype(float), has_constant="add")
            y_all = model_df[target].values.astype(float)
            evaluation = _evaluate_model(
                model_id=model_id,
                model_obj=model_obj,
                X_train=X_all,
                y_train=y_all,
            )
    except Exception:
        pass

    # Run interpretation
    interpretation = None
    try:
        interpretation = _interpret_model(
            model_id=model_id,
            model_obj=model_obj,
            df=df,
            features=features,
            target=target,
        )
    except Exception:
        pass

    try:
        result = _export_diagnostic_report(
            model_id=model_id,
            model_obj=model_obj,
            df=df,
            features=features,
            target=target,
            format_type=format_type,
            diagnostics=diagnostics,
            evaluation=evaluation,
            interpretation=interpretation,
            registry_data=registry,
        )
    except Exception as exc:
        return {"status": "error", "error": f"Report generation failed: {exc}"}

    return result


# ========================================================================
# ENTRY POINT
# ========================================================================


def main() -> None:
    """Run the LAECON MCP server with stdio transport."""
    logger.info("Starting LAECON MCP server (M1 — all 7 tools implemented)...")
    mcp.run()


if __name__ == "__main__":
    main()
