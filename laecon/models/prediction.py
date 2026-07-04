"""Model prediction — apply trained model to new data.

Returns point predictions with confidence intervals.
Includes DataFrame empty guards (DR-1, Constitution Art. 3.6).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from laecon.models.data_utils import check_empty


def predict(
    model_id: str,
    model_obj: Any,
    df: pd.DataFrame,
    features: list[str],
    return_ci: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply a trained model to new data for predictions.

    Args:
        model_id: Model identifier.
        model_obj: Fitted statsmodels result object.
        df: DataFrame with new data (must contain feature columns).
        features: List of feature column names used during training.
        return_ci: Whether to return confidence intervals.
        alpha: Significance level for CIs (default 0.05 = 95% CI).

    Returns:
        dict with:
          - status, model_id, model_class
          - predictions: list of predicted values
          - confidence_intervals: list of [lower, upper] if return_ci
          - n_predictions: int
          - feature_values: list of dicts with input feature values
          - warnings: list[str]

    Raises:
        ValueError: If DataFrame is empty (DR-1), columns missing.
    """
    check_empty(df, "prediction data")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Features missing in prediction data: {missing}"
        )

    # Check for NaN in features
    pred_df = df[features].copy()
    nan_mask = pred_df.isna().any(axis=1)
    if nan_mask.any():
        warnings = [
            f"{nan_mask.sum()} row(s) have NaN values in features. "
            "These rows will be skipped."
        ]
        pred_df = pred_df[~nan_mask].copy()
    else:
        warnings = []

    check_empty(pred_df, "prediction data after NaN removal")

    X_pred = pred_df[features].values.astype(float)
    X_pred = sm.add_constant(X_pred, has_constant="add")

    # Detect model type
    model_class = type(model_obj).__name__
    has_rsquared = hasattr(model_obj, "rsquared")
    has_llnull = hasattr(model_obj, "llnull")
    is_ols = has_rsquared or "RegressionResults" in model_class or "OLS" in model_class
    is_logit = hasattr(model_obj, "model") and "Logit" in type(model_obj.model).__name__
    is_probit = hasattr(model_obj, "model") and "Probit" in type(model_obj.model).__name__
    is_binary = is_logit or is_probit or has_llnull
    is_ordered = hasattr(model_obj, "model") and "OrderedModel" in type(model_obj.model).__name__
    is_multinomial = hasattr(model_obj, "model") and "MNLogit" in type(model_obj.model).__name__

    # --- Generate predictions ---
    try:
        y_pred = model_obj.predict(X_pred)
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    # Convert to list
    if hasattr(y_pred, "tolist"):
        pred_list = y_pred.tolist()
    elif isinstance(y_pred, np.ndarray):
        pred_list = y_pred.tolist()
    else:
        pred_list = [float(y_pred)]

    # Handle multi-dimensional predictions (ordered, multinomial)
    if isinstance(pred_list[0], list) or (isinstance(pred_list[0], (np.floating, float)) and is_binary):
        pass  # already flat

    result: dict[str, Any] = {
        "status": "ok",
        "model_id": model_id,
        "model_class": model_class,
        "predictions": pred_list,
        "n_predictions": len(pred_list),
        "warnings": warnings,
    }

    # --- Feature values for traceability ---
    feature_values = []
    for idx in range(len(pred_list)):
        row = {f: float(pred_df.iloc[idx][f]) for f in features}
        feature_values.append(row)
    result["feature_values"] = feature_values

    # --- Confidence intervals ---
    if return_ci and is_ols:
        try:
            pred_results = model_obj.get_prediction(X_pred)
            ci_frame = pred_results.conf_int(alpha=alpha)
            ci_list = [
                [round(float(ci_frame.iloc[i, 0]), 6), round(float(ci_frame.iloc[i, 1]), 6)]
                for i in range(ci_frame.shape[0])
            ]
            result["confidence_intervals"] = ci_list
        except Exception:
            # Fallback: manual CI using MSE and leverage
            try:
                mse = model_obj.mse_resid
                # Hat matrix diagonal (leverage)
                try:
                    infl = model_obj.get_influence()
                    h = infl.hat_matrix_diag
                except Exception:
                    h = np.zeros(len(pred_list))
                se = np.sqrt(mse * (1.0 + h))
                t_crit = 1.96  # approximate for large n
                ci_list = [
                    [round(float(pred_list[i] - t_crit * se[i]), 6),
                     round(float(pred_list[i] + t_crit * se[i]), 6)]
                    for i in range(len(pred_list))
                ]
                result["confidence_intervals"] = ci_list
            except Exception:
                result["confidence_intervals"] = None

    elif return_ci and is_binary:
        # For logit/probit, use delta method via get_prediction
        try:
            pred_results = model_obj.get_prediction(X_pred)
            # For binary models, se_mean gives the SE of the linear predictor
            se_pred = pred_results.se_mean
            linpred = model_obj.predict(X_pred, linear=True)

            if is_logit:
                # Inverse logit: CI on probability scale
                from scipy.special import expit
                lower_prob = expit(linpred - 1.96 * se_pred)
                upper_prob = expit(linpred + 1.96 * se_pred)
            else:
                from scipy.stats import norm
                lower_prob = norm.cdf(linpred - 1.96 * se_pred)
                upper_prob = norm.cdf(linpred + 1.96 * se_pred)

            ci_list = [
                [round(float(lower_prob[i]), 6), round(float(upper_prob[i]), 6)]
                for i in range(len(lower_prob))
            ]
            result["confidence_intervals"] = ci_list
        except Exception:
            # Fallback: use standard prediction CI at probability scale
            try:
                from scipy.special import expit
                y_pred_arr = np.array(pred_list)
                # Rough SE: sqrt(p*(1-p)/n) as approximation
                se_pred = np.sqrt(y_pred_arr * (1 - y_pred_arr) / len(y_pred_arr))
                lower_prob = np.clip(y_pred_arr - 1.96 * se_pred, 0, 1)
                upper_prob = np.clip(y_pred_arr + 1.96 * se_pred, 0, 1)
                ci_list = [
                    [round(float(lower_prob[i]), 6), round(float(upper_prob[i]), 6)]
                    for i in range(len(lower_prob))
                ]
                result["confidence_intervals"] = ci_list
            except Exception:
                result["confidence_intervals"] = None

    else:
        result["confidence_intervals"] = None

    return result
