"""Model interpretation — marginal effects, odds ratios, partial dependence.

For binary models: uses statsmodels' get_margeff() for AME and MER.
For ordered models: custom implementation.
For partial dependence: one-feature PDP using ICE approach.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from laecon.models.data_utils import check_empty


def interpret_model(
    model_id: str,
    model_obj: Any,
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    target: str | None = None,
    pdp_feature: str | None = None,
) -> dict[str, Any]:
    """Interpret a trained model.

    Args:
        model_id: Model identifier.
        model_obj: Fitted statsmodels result object.
        df: Original training data (needed for MER and PDP).
        features: Feature names used in the model.
        target: Target column name (for PDP).
        pdp_feature: Optional feature name for partial dependence plot data.

    Returns:
        dict with:
          - status, model_id
          - interpretation_type: str (what was computed)
          - average_marginal_effects: list of {feature, effect, se, z, pvalue}
          - marginal_effects_at_representative: list (if df provided)
          - odds_ratios: list of {feature, odds_ratio, ci_lower, ci_upper}
          - elasticities: list of {feature, elasticity}
          - partial_dependence: dict with x_values, y_values (if pdp_feature)
          - confidence: float
    """
    result: dict[str, Any] = {
        "status": "ok",
        "model_id": model_id,
    }

    # Detect model type by class name AND attributes
    model_class = type(model_obj).__name__
    has_rsquared = hasattr(model_obj, "rsquared")
    has_llnull = hasattr(model_obj, "llnull")
    is_ols = has_rsquared or "RegressionResults" in model_class or "OLS" in model_class
    is_logit = hasattr(model_obj, "model") and "Logit" in type(model_obj.model).__name__
    is_probit = hasattr(model_obj, "model") and "Probit" in type(model_obj.model).__name__
    is_binary = is_logit or is_probit or has_llnull
    is_ordered = hasattr(model_obj, "model") and "OrderedModel" in type(model_obj.model).__name__
    is_multinomial = hasattr(model_obj, "model") and "MNLogit" in type(model_obj.model).__name__

    result["model_class"] = model_class
    result["interpretation_type"] = (
        "binary" if is_binary else
        "ordered" if is_ordered else
        "multinomial" if is_multinomial else
        "linear"
    )

    # --- Average Marginal Effects (AME) for binary models ---
    ame_list: list[dict[str, Any]] = []
    if is_binary:
        try:
            margeff = model_obj.get_margeff(at="mean", method="dydx")
            # Determine feature names
            fnames = _get_feature_names(model_obj, features)
            for i, fname in enumerate(fnames):
                ame_list.append({
                    "feature": fname,
                    "effect": round(float(margeff.margeff[i]), 6),
                    "se": round(float(margeff.margeff_se[i]), 6),
                    "z_stat": round(float(margeff.tvalues[i]), 4),
                    "p_value": round(float(margeff.pvalues[i]), 6),
                })
        except Exception as exc:
            ame_list = [{"error": str(exc)}]

    result["average_marginal_effects"] = ame_list

    # --- Marginal Effects at Representative Values (MER) ---
    mer_list: list[dict[str, Any]] = []
    if is_binary and df is not None and features:
        try:
            # Pick a representative profile: median of numeric features
            rep_row = {}
            for f in features:
                if f in df.columns:
                    if pd.api.types.is_numeric_dtype(df[f]):
                        rep_row[f] = df[f].median()
                    else:
                        rep_row[f] = df[f].mode().iloc[0] if not df[f].mode().empty else df[f].iloc[0]

            rep_df = pd.DataFrame([rep_row])
            rep_X = sm.add_constant(rep_df[features].values.astype(float), has_constant="add")

            # Get marginal effects at this representative point
            # For binary models, we can compute dydx at specific X values
            # statsmodels get_margeff(at="overall") gives AME,
            # for MER we need to manually compute using the prediction
            beta = model_obj.params
            x_rep = rep_X[0]
            if is_logit:
                # For logit: dydx = p * (1-p) * beta
                linpred = np.dot(x_rep, beta)
                p = 1.0 / (1.0 + np.exp(-linpred))
                me_at_rep = p * (1.0 - p) * beta[1:]  # exclude const
            elif is_probit:
                linpred = np.dot(x_rep, beta)
                p = 1.0 / (1.0 + np.exp(-linpred))
                from scipy.stats import norm
                pdf_val = norm.pdf(linpred)
                me_at_rep = pdf_val * beta[1:]

            fnames = features
            for i, fname in enumerate(fnames):
                val = float(me_at_rep[i]) if i < len(me_at_rep) else 0.0
                mer_list.append({
                    "feature": fname,
                    "representative_value": float(rep_row.get(fname, 0)),
                    "effect": round(val, 6),
                })
        except Exception as exc:
            mer_list = [{"error": str(exc)}]

    result["marginal_effects_at_representative"] = mer_list

    # --- Odds ratios (logit) ---
    odds_list: list[dict[str, Any]] = []
    if is_logit:
        try:
            params = model_obj.params
            conf_int = model_obj.conf_int()
            for i, pname in enumerate(params.index):
                if pname != "const":
                    odds_list.append({
                        "feature": pname,
                        "odds_ratio": round(float(np.exp(params.iloc[i])), 6),
                        "ci_lower": round(float(np.exp(conf_int.iloc[i, 0])), 6),
                        "ci_upper": round(float(np.exp(conf_int.iloc[i, 1])), 6),
                    })
        except Exception as exc:
            odds_list = [{"error": str(exc)}]

    result["odds_ratios"] = odds_list

    # --- Elasticities (linear model, log-log interpretation) ---
    elasticity_list: list[dict[str, Any]] = []
    if is_ols and df is not None and features:
        try:
            params = model_obj.params
            # Handle both numpy array and pandas Series
            if hasattr(params, "iloc"):
                params_list = [params.iloc[i] for i in range(len(params))]
            else:
                params_list = list(params)
            # Compute elasticity at means: beta_j * mean(x_j) / mean(y)
            y_mean = float(df[target].mean()) if target and target in df.columns else 1.0
            for i, fname in enumerate(features):
                param_idx = i + 1  # skip const
                if param_idx < len(params_list) and fname in df.columns:
                    x_mean = float(df[fname].mean())
                    elasticity = float(params_list[param_idx]) * x_mean / y_mean
                    elasticity_list.append({
                        "feature": fname,
                        "elasticity": round(elasticity, 6),
                    })
        except Exception as exc:
            elasticity_list = [{"error": str(exc)}]

    result["elasticities"] = elasticity_list

    # --- Partial dependence (1 feature) ---
    pdp_result: dict[str, Any] = {}
    if pdp_feature and df is not None and pdp_feature in df.columns:
        try:
            pdp_result = _compute_partial_dependence(
                model_obj, df, features, pdp_feature
            )
        except Exception as exc:
            pdp_result = {"error": str(exc)}

    result["partial_dependence"] = pdp_result

    # Confidence heuristic
    confidence = 0.6
    n = getattr(model_obj, "nobs", 0)
    if n:
        confidence += min(0.20, 0.05 * (n ** 0.3))
    result["confidence"] = round(min(1.0, max(0.0, confidence)), 4)

    return result


def _get_feature_names(model_obj: Any, features: list[str] | None = None) -> list[str]:
    """Extract feature names from model object, excluding constant."""
    try:
        params = model_obj.params
        names = [pname for pname in params.index if pname != "const"]
        return names
    except Exception:
        return features or []


def _compute_partial_dependence(
    model_obj: Any,
    df: pd.DataFrame,
    features: list[str] | None,
    pdp_feature: str,
    grid_points: int = 20,
) -> dict[str, Any]:
    """Compute 1-feature partial dependence.

    For linear models: PDP is linear in the feature.
    For logit/probit: average predicted probability across the grid.
    """
    if features is None:
        raise ValueError("features list required for PDP")

    # Build feature grid
    feat_vals = df[pdp_feature].dropna()
    if len(feat_vals) == 0:
        raise ValueError(f"No non-null values for PDP feature '{pdp_feature}'")

    grid = np.linspace(feat_vals.min(), feat_vals.max(), grid_points)

    # Prepare base data: use all rows, but replace pdp_feature with grid values
    other_features = [f for f in features if f != pdp_feature]

    # Build X matrix template (with const)
    preds_list = []
    for g in grid:
        x_row = np.ones(len(other_features) + 1)  # const + features
        # Fill other features at their mean
        for j, of in enumerate(other_features):
            if of in df.columns:
                x_row[j + 1] = df[of].mean()
        # Place grid value at the pdp_feature position
        feat_idx = features.index(pdp_feature) + 1  # +1 for const
        x_row[feat_idx] = g

        try:
            pred = model_obj.predict(x_row, linear=True)
            pred_val = float(pred) if hasattr(pred, "__len__") else float(pred)
        except Exception:
            pred = model_obj.predict(x_row)
            pred_val = float(pred) if hasattr(pred, "__len__") else float(pred)

        preds_list.append(pred_val)

    return {
        "feature": pdp_feature,
        "x_values": [round(float(g), 6) for g in grid],
        "y_values": [round(float(p), 6) for p in preds_list],
        "grid_points": grid_points,
    }
