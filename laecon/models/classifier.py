"""Classification model training — logit, probit, ordered, grouped.

Supports:
  - Binary logit / probit (via statsmodels Logit, Probit)
  - Ordered logit / probit (via statsmodels OrderedModel)
  - Grouped logit (via statsmodels MNLogit for grouped categories)

Returns: coefficients, marginal effects, odds ratios, cutpoints (ordered).
Critical for NPS driver analysis (Larson & Goungetas, Quirk's 2013).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from laecon.models.data_utils import check_empty


def train_classifier(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_id: str,
    family: str = "logit",
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Train a classification model and return results.

    Args:
        df: Training DataFrame.
        target: Name of the target column.
        features: List of feature column names.
        model_id: Unique model identifier (AE-1).
        family: "logit", "probit", "ordered_logit", "ordered_probit",
                "grouped_logit" (multinomial).
        weights: Optional sample weights.

    Returns:
        dict with:
          - status, model_id, algorithm
          - coefficients: list of {name, coef, se, z_stat, p_value, ci_lower, ci_upper}
          - metrics: {log_likelihood, ll_null, aic, bic, n_obs, pseudo_r_squared, ...}
          - marginal_effects: list of {feature, effect, se} (AME)
          - odds_ratios: list of {feature, odds_ratio, ci_lower, ci_upper} (for logit)
          - cutpoints: list of cutpoint coefficients (for ordered)
          - n_classes: int
          - confidence: float
          - warnings: list[str]
          - _fitted: fitted model object

    Raises:
        ValueError: If data empty, columns missing, or invalid family.
    """
    check_empty(df, "training data")

    missing = [c for c in [target] + features if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    cols = [target] + features
    model_df = df[cols].dropna().copy()
    check_empty(model_df, "training data after dropping NA")

    y = model_df[target].values
    X = model_df[features].values.astype(float)
    X = sm.add_constant(X, has_constant="add")
    feature_names = ["const"] + features

    warnings: list[str] = []
    family = family.lower()

    # Determine number of classes
    unique_vals = np.unique(y)
    n_classes = len(unique_vals)

    # --- Model selection ---
    fitted = None
    model_obj = None
    try:
        if family in ("logit", "probit"):
            if n_classes > 2:
                warnings.append(
                    f"Target has {n_classes} unique values. "
                    f"Using binary {family} on binarized data. "
                    "Consider ordered_logit or grouped_logit for ordinal/multinomial."
                )
            # Binarize: keep only first two unique values
            y_bin = (y == unique_vals[1]).astype(int)

            if family == "logit":
                model = sm.Logit(y_bin, X)
            else:
                model = sm.Probit(y_bin, X)

            fitted = model.fit(disp=False, maxiter=100)

            # Marginal effects (AME)
            margeff = fitted.get_margeff(at="mean", method="dydx")

        elif family == "ordered_logit":
            try:
                from statsmodels.miscmodels.ordinal_model import OrderedModel
            except ImportError:
                raise RuntimeError(
                    "OrderedModel requires statsmodels >= 0.14. "
                    "Ordered logit not available."
                )

            # y must be ordinal integer, 0-indexed
            y_ord = pd.Categorical(y).codes
            model = OrderedModel(y_ord, X[:, 1:])  # exclude const for OrderedModel
            fitted = model.fit(method="bfgs", disp=False, maxiter=200)
            feature_names = features  # OrderedModel doesn't use const like that

        elif family == "ordered_probit":
            try:
                from statsmodels.miscmodels.ordinal_model import OrderedModel
            except ImportError:
                raise RuntimeError("OrderedModel not available.")

            y_ord = pd.Categorical(y).codes
            model = OrderedModel(y_ord, X[:, 1:], distr="probit")
            fitted = model.fit(method="bfgs", disp=False, maxiter=200)
            feature_names = features

        elif family == "grouped_logit":
            # Multinomial logit
            if n_classes < 2:
                raise ValueError(
                    f"Grouped logit requires >=2 classes, got {n_classes}."
                )
            if n_classes > 20:
                warnings.append(
                    f"Large number of classes ({n_classes}) for multinomial logit. "
                    "Results may be unstable."
                )

            model = sm.MNLogit(y, X)
            fitted = model.fit(disp=False, maxiter=100, method="newton")
            # For MNLogit, coefficients are per class
            feature_names = ["const"] + features

        else:
            raise ValueError(
                f"Unknown family: {family}. "
                "Use: logit, probit, ordered_logit, ordered_probit, grouped_logit."
            )

    except Exception as exc:
        raise RuntimeError(f"Model fitting failed ({family}): {exc}") from exc

    # --- Extract coefficients ---
    coefs_list: list[dict[str, Any]] = []

    if family in ("ordered_logit", "ordered_probit"):
        # OrderedModel: params include thresholds (cutpoints)
        # Identify cutpoints by name pattern
        is_cutpoint = [pname.startswith("cut") or pname.startswith("z") or "threshold" in pname
                       for pname in fitted.params.index]
        coef_names = [pname for pname in fitted.params.index]
        cutpoints_list = []

        for i, pname in enumerate(coef_names):
            entry = {
                "name": pname,
                "coefficient": round(float(fitted.params.iloc[i]), 6),
                "se": round(float(fitted.bse.iloc[i]), 6),
                "z_stat": round(float(fitted.tvalues.iloc[i]), 4),
                "p_value": round(float(fitted.pvalues.iloc[i]), 6),
            }
            try:
                ci = fitted.conf_int().iloc[i]
                entry["ci_lower"] = round(float(ci.iloc[0]), 6)
                entry["ci_upper"] = round(float(ci.iloc[1]), 6)
            except Exception:
                entry["ci_lower"] = None
                entry["ci_upper"] = None

            if "cut" in pname or "threshold" in pname or pname.startswith("z"):
                cutpoints_list.append(entry)
            else:
                coefs_list.append(entry)

    elif family == "grouped_logit":
        # MNLogit: params is a DataFrame (classes × features)
        for cls_idx in range(fitted.params.shape[0]):
            cls_label = str(unique_vals[cls_idx]) if cls_idx < len(unique_vals) else str(cls_idx)
            for feat_idx, fname in enumerate(feature_names):
                coefs_list.append({
                    "name": fname,
                    "class": cls_label,
                    "coefficient": round(float(fitted.params.iloc[cls_idx, feat_idx]), 6),
                    "se": round(float(fitted.bse.iloc[cls_idx, feat_idx]), 6),
                    "z_stat": round(float(fitted.tvalues.iloc[cls_idx, feat_idx]), 4),
                    "p_value": round(float(fitted.pvalues.iloc[cls_idx, feat_idx]), 6),
                })
    else:
        for i, name in enumerate(feature_names):
            entry = {
                "name": name,
                "coefficient": round(float(fitted.params[i]), 6),
                "se": round(float(fitted.bse[i]), 6),
                "z_stat": round(float(fitted.tvalues[i]), 4),
                "p_value": round(float(fitted.pvalues[i]), 6),
            }
            try:
                ci = fitted.conf_int().iloc[i]
                entry["ci_lower"] = round(float(ci.iloc[0]), 6)
                entry["ci_upper"] = round(float(ci.iloc[1]), 6)
            except Exception:
                entry["ci_lower"] = None
                entry["ci_upper"] = None
            coefs_list.append(entry)

    # --- Extract metrics ---
    try:
        llf = float(fitted.llf)
    except Exception:
        llf = None
    try:
        llnull = float(fitted.llnull)
    except Exception:
        llnull = None

    if llf is not None and llnull is not None and llnull != 0:
        pseudo_r2 = 1 - llf / llnull
    else:
        pseudo_r2 = None

    metrics = {
        "log_likelihood": round(llf, 4) if llf else None,
        "ll_null": round(llnull, 4) if llnull else None,
        "aic": round(float(fitted.aic), 4) if hasattr(fitted, "aic") and fitted.aic else None,
        "bic": round(float(fitted.bic), 4) if hasattr(fitted, "bic") and fitted.bic else None,
        "pseudo_r_squared": round(pseudo_r2, 6) if pseudo_r2 else None,
        "n_obs": int(fitted.nobs) if hasattr(fitted, "nobs") else len(model_df),
        "n_features": len(feature_names),
        "family": family,
        "n_classes": n_classes,
    }

    # --- Marginal effects (binary logit/probit) ---
    marginal_effects: list[dict[str, Any]] = []
    if family in ("logit", "probit") and fitted is not None:
        try:
            margeff = fitted.get_margeff(at="mean", method="dydx")
            # margeff.margeff is array, one per feature (excluding const usually)
            # margeff.margeff_se
            for j, fname in enumerate(features):
                marginal_effects.append({
                    "feature": fname,
                    "effect": round(float(margeff.margeff[j]), 6),
                    "se": round(float(margeff.margeff_se[j]), 6),
                    "z_stat": round(float(margeff.tvalues[j]), 4),
                    "p_value": round(float(margeff.pvalues[j]), 6),
                })
        except Exception:
            pass

    # --- Odds ratios (binary logit) ---
    odds_ratios: list[dict[str, Any]] = []
    if family == "logit":
        for entry in coefs_list:
            if entry["name"] != "const":
                odds_ratios.append({
                    "feature": entry["name"],
                    "odds_ratio": round(np.exp(entry["coefficient"]), 6),
                    "ci_lower": round(np.exp(entry["ci_lower"]), 6) if entry.get("ci_lower") else None,
                    "ci_upper": round(np.exp(entry["ci_upper"]), 6) if entry.get("ci_upper") else None,
                })

    # --- Cutpoints (ordered) ---
    cutpoints_list_local = cutpoints_list if family in ("ordered_logit", "ordered_probit") else []

    # Heuristic confidence
    n = metrics["n_obs"] or len(model_df)
    confidence = 0.5
    confidence += min(0.20, 0.05 * (n ** 0.3))
    if pseudo_r2:
        confidence += min(0.15, pseudo_r2 * 0.20)
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    return {
        "status": "ok",
        "model_id": model_id,
        "algorithm": family,
        "coefficients": coefs_list,
        "metrics": metrics,
        "marginal_effects": marginal_effects,
        "odds_ratios": odds_ratios,
        "cutpoints": cutpoints_list_local,
        "n_classes": n_classes,
        "confidence": confidence,
        "warnings": warnings,
        "_fitted": fitted,
    }
