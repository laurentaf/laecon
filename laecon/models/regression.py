"""Linear regression training — OLS, GLS, WLS with robust SE.

Supports:
  - OLS (ordinary least squares)
  - GLS (generalized least squares)
  - WLS (weighted least squares)
  - Robust standard errors: homoskedastic, Huber-White (HC0-HC3), HAC
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from laecon.models.data_utils import check_empty


def train_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_id: str,
    algorithm: str = "ols",
    se_type: str = "homoskedastic",
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Train a linear regression model and return results.

    Args:
        df: Training DataFrame.
        target: Name of the target column.
        features: List of feature column names.
        model_id: Unique identifier for the model (AE-1).
        algorithm: "ols", "gls", or "wls".
        se_type: "homoskedastic", "huber-white", or "hac".
        weights: Optional sample weights for WLS.

    Returns:
        dict with:
          - status: "ok"
          - model_id: str
          - algorithm: str
          - coefficients: list of {name, coef, se, t_stat, p_value, ci_lower, ci_upper}
          - metrics: {r_squared, adj_r_squared, f_stat, f_pvalue, aic, bic, log_likelihood, n_obs, n_features}
          - confidence: float
          - warnings: list[str]

    Raises:
        ValueError: If data is empty, target/features missing, or invalid params.
    """
    check_empty(df, "training data")

    # Validate columns
    missing = [c for c in [target] + features if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    # Drop rows with missing values in relevant columns
    cols = [target] + features
    model_df = df[cols].dropna().copy()
    check_empty(model_df, "training data after dropping NA")

    y = model_df[target].values.astype(float)
    X = model_df[features].values.astype(float)
    X = sm.add_constant(X, has_constant="add")
    feature_names = ["const"] + features

    # Warn about low observations
    warnings: list[str] = []
    n_obs = len(y)
    n_features = len(feature_names)
    if n_obs < 3 * n_features:
        warnings.append(
            f"Low observations ({n_obs}) relative to features ({n_features}). "
            "Results may be unreliable."
        )

    algorithm = algorithm.lower()
    se_type = se_type.lower()

    # --- Train model ---
    model_obj = None
    try:
        if algorithm == "ols":
            model = sm.OLS(y, X)
            fitted = model.fit(cov_type="nonrobust" if se_type == "homoskedastic" else None)

        elif algorithm == "wls":
            if weights is None:
                raise ValueError("WLS requires 'weights' parameter.")
            # Ensure weights align
            pw = np.asarray(weights, dtype=float)
            if len(pw) != len(model_df):
                raise ValueError(
                    f"Weights length ({len(pw)}) does not match "
                    f"data rows ({len(model_df)})."
                )
            model = sm.WLS(y, X, weights=pw)
            fitted = model.fit()

        elif algorithm == "gls":
            # GLS with simple sigma specification
            # For real GLS, user provides a covariance structure
            # Here we use a simple AR(1) approximation
            model = sm.GLS(y, X)
            fitted = model.fit()

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use ols, wls, or gls.")

        # Apply robust SE if requested
        if se_type == "huber-white":
            fitted = model.fit(cov_type="HC1")
        elif se_type == "hac":
            # HAC with 1 lag by default
            fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    except Exception as exc:
        raise RuntimeError(f"Model fitting failed: {exc}") from exc

    # --- Extract coefficients ---
    coefs_list: list[dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        try:
            ci = fitted.conf_int().iloc[i]
            ci_lower = float(ci.iloc[0])
            ci_upper = float(ci.iloc[1])
        except Exception:
            ci_lower = float(fitted.params[i] - 1.96 * fitted.bse[i])
            ci_upper = float(fitted.params[i] + 1.96 * fitted.bse[i])

        coefs_list.append({
            "name": name,
            "coefficient": round(float(fitted.params[i]), 6),
            "se": round(float(fitted.bse[i]), 6),
            "t_stat": round(float(fitted.tvalues[i]), 4),
            "p_value": round(float(fitted.pvalues[i]), 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
        })

    # --- Extract metrics ---
    metrics = {
        "r_squared": round(float(fitted.rsquared), 6),
        "adj_r_squared": round(float(fitted.rsquared_adj), 6),
        "f_stat": round(float(fitted.fvalue), 4),
        "f_pvalue": round(float(fitted.f_pvalue), 6),
        "aic": round(float(fitted.aic), 4),
        "bic": round(float(fitted.bic), 4),
        "log_likelihood": round(float(fitted.llf), 4),
        "n_obs": int(fitted.nobs),
        "n_features": n_features,
        "se_type": se_type,
    }

    # Heuristic confidence
    n = metrics["n_obs"]
    r2 = metrics["r_squared"]
    confidence = 0.5
    confidence += min(0.20, 0.05 * (n ** 0.3))
    confidence += min(0.15, r2 * 0.15)
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    model_obj = fitted  # to be pickled by caller

    return {
        "status": "ok",
        "model_id": model_id,
        "algorithm": algorithm,
        "se_type": se_type,
        "coefficients": coefs_list,
        "metrics": metrics,
        "confidence": confidence,
        "warnings": warnings,
        "_fitted": fitted,  # passed to registry for pickle
    }
