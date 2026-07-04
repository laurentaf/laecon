"""Econometric assumption validation.

Tests:
  - Heteroscedasticity: Breusch-Pagan, White
  - Autocorrelation: Durbin-Watson, Breusch-Godfrey (LM)
  - Multicollinearity: VIF
  - Normality of residuals: Jarque-Bera, Shapiro-Wilk

Returns structured PASS/WARN/FAIL flags per test, ideal for n8n IF nodes (AE-2).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    normal_ad,
)


def validate_assumptions(
    model_id: str,
    exog: np.ndarray | None = None,
    resid: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    n_lags: int = 1,
) -> dict[str, Any]:
    """Run assumption diagnostics on a trained model's residuals/data.

    Args:
        model_id: Model identifier (for reference).
        exog: Exogenous/feature matrix (with const column) — needed for
              heteroscedasticity and VIF tests. Shape (n_obs, n_features).
        resid: Residuals array — needed for autocorrelation and normality.
        feature_names: Names of features for VIF (excluding const).
        n_lags: Number of lags for Breusch-Godfrey test.

    Returns:
        dict with:
          - status: "ok"
          - model_id: str
          - tests: dict with per-test results:
              - heteroscedasticity: {breusch_pagan: {stat, pvalue, flag},
                                     white: {stat, pvalue, flag}}
              - autocorrelation: {durbin_watson: {stat, flag},
                                  breusch_godfrey: {stat, pvalue, lag, flag}}
              - multicollinearity: {vif: {values: [{feature, vif}], max_vif, flag}}
              - normality: {jarque_bera: {stat, pvalue, flag},
                            shapiro_wilk: {stat, pvalue, flag}}
          - summary: {pass_count, warn_count, fail_count, overall}
          - confidence_adjustment: float (reduction for failing tests)

    Flags:
      PASS: test passes at alpha=0.05
      WARN: borderline (0.05 < p < 0.10 or VIF between 5-10)
      FAIL: test fails (p < 0.05 or VIF > 10)
    """
    alpha = 0.05
    results: dict[str, Any] = {
        "status": "ok",
        "model_id": model_id,
        "tests": {},
        "summary": {
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "overall": "PASS",
        },
        "confidence_adjustment": 0.0,
    }

    pass_count = 0
    warn_count = 0
    fail_count = 0

    # --- Heteroscedasticity ---
    het_tests: dict[str, Any] = {}

    if exog is not None and resid is not None:
        # Breusch-Pagan
        try:
            bp_stat, bp_pval, _, _ = het_breuschpagan(resid, exog)
            flag = _pvalue_flag(bp_pval, alpha)
            het_tests["breusch_pagan"] = {
                "statistic": round(float(bp_stat), 4),
                "p_value": round(float(bp_pval), 6),
                "flag": flag,
            }
            pass_count += 1 if flag == "PASS" else 0
            warn_count += 1 if flag == "WARN" else 0
            fail_count += 1 if flag == "FAIL" else 0
        except Exception as e:
            het_tests["breusch_pagan"] = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

        # White test
        try:
            w_stat, w_pval, _, _ = het_white(resid, exog)
            flag = _pvalue_flag(w_pval, alpha)
            het_tests["white"] = {
                "statistic": round(float(w_stat), 4),
                "p_value": round(float(w_pval), 6),
                "flag": flag,
            }
            pass_count += 1 if flag == "PASS" else 0
            warn_count += 1 if flag == "WARN" else 0
            fail_count += 1 if flag == "FAIL" else 0
        except Exception as e:
            het_tests["white"] = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

    results["tests"]["heteroscedasticity"] = het_tests

    # --- Autocorrelation ---
    auto_tests: dict[str, Any] = {}

    if resid is not None:
        # Durbin-Watson
        try:
            dw_stat = durbin_watson(resid)
            # DW ~2 = no autocorrelation; <1 or >3 = strong
            if 1.5 <= dw_stat <= 2.5:
                dw_flag = "PASS"
            elif 1.0 <= dw_stat < 1.5 or 2.5 < dw_stat <= 3.0:
                dw_flag = "WARN"
            else:
                dw_flag = "FAIL"
            auto_tests["durbin_watson"] = {
                "statistic": round(float(dw_stat), 4),
                "flag": dw_flag,
            }
            pass_count += 1 if dw_flag == "PASS" else 0
            warn_count += 1 if dw_flag == "WARN" else 0
            fail_count += 1 if dw_flag == "FAIL" else 0
        except Exception as e:
            auto_tests["durbin_watson"] = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

        # Breusch-Godfrey (requires exog for LM test)
        try:
            if exog is not None:
                bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(
                    sm.OLS(resid, exog).fit(), nlags=n_lags
                )
                flag = _pvalue_flag(bg_pval, alpha)
                auto_tests["breusch_godfrey"] = {
                    "statistic": round(float(bg_stat), 4),
                    "p_value": round(float(bg_pval), 6),
                    "lags": n_lags,
                    "flag": flag,
                }
                pass_count += 1 if flag == "PASS" else 0
                warn_count += 1 if flag == "WARN" else 0
                fail_count += 1 if flag == "FAIL" else 0
        except Exception as e:
            auto_tests["breusch_godfrey"] = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

    results["tests"]["autocorrelation"] = auto_tests

    # --- Multicollinearity (VIF) ---
    vif_results: dict[str, Any] = {}

    if exog is not None:
        try:
            # Exclude const column (first column is const if add_constant was used)
            # We'll try to detect: if first col is all 1s, skip it for VIF
            exog_for_vif = exog
            start_idx = 0
            vif_feature_names = feature_names or [f"x{i}" for i in range(exog.shape[1])]

            # If first column looks like constant, remove for VIF
            if exog.shape[1] > 1 and np.allclose(exog[:, 0], 1.0):
                exog_for_vif = exog[:, 1:]
                start_idx = 1
                vif_feature_names = vif_feature_names[1:] if len(vif_feature_names) > exog.shape[1] - 1 else [f"x{i}" for i in range(exog_for_vif.shape[1])]

            vif_values: list[dict[str, Any]] = []
            for i in range(exog_for_vif.shape[1]):
                vif_val = variance_inflation_factor(exog_for_vif, i)
                fname = vif_feature_names[i] if i < len(vif_feature_names) else f"x{i + start_idx}"
                vif_val_float = float(vif_val)
                if vif_val_float > 10:
                    vif_flag = "FAIL"
                elif vif_val_float > 5:
                    vif_flag = "WARN"
                else:
                    vif_flag = "PASS"
                vif_values.append({"feature": fname, "vif": round(vif_val_float, 4), "flag": vif_flag})
                pass_count += 1 if vif_flag == "PASS" else 0
                warn_count += 1 if vif_flag == "WARN" else 0
                fail_count += 1 if vif_flag == "FAIL" else 0

            max_vif = max(v["vif"] for v in vif_values) if vif_values else 0.0
            vif_results = {
                "values": vif_values,
                "max_vif": round(float(max_vif), 4),
                "flag": "FAIL" if max_vif > 10 else ("WARN" if max_vif > 5 else "PASS"),
            }
        except Exception as e:
            vif_results = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

    results["tests"]["multicollinearity"] = vif_results

    # --- Normality ---
    norm_tests: dict[str, Any] = {}

    if resid is not None:
        # Jarque-Bera
        try:
            jb_stat, jb_pval = scipy_stats.jarque_bera(resid)
            flag = _pvalue_flag(jb_pval, alpha)
            norm_tests["jarque_bera"] = {
                "statistic": round(float(jb_stat), 4),
                "p_value": round(float(jb_pval), 6),
                "flag": flag,
            }
            pass_count += 1 if flag == "PASS" else 0
            warn_count += 1 if flag == "WARN" else 0
            fail_count += 1 if flag == "FAIL" else 0
        except Exception as e:
            norm_tests["jarque_bera"] = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

        # Shapiro-Wilk (works best for n < 5000)
        try:
            if len(resid) < 5000:
                sw_stat, sw_pval = scipy_stats.shapiro(resid)
                flag = _pvalue_flag(sw_pval, alpha)
                norm_tests["shapiro_wilk"] = {
                    "statistic": round(float(sw_stat), 4),
                    "p_value": round(float(sw_pval), 6),
                    "flag": flag,
                }
                pass_count += 1 if flag == "PASS" else 0
                warn_count += 1 if flag == "WARN" else 0
                fail_count += 1 if flag == "FAIL" else 0
        except Exception as e:
            norm_tests["shapiro_wilk"] = {"error": str(e), "flag": "FAIL"}
            fail_count += 1

    results["tests"]["normality"] = norm_tests

    # --- Summary ---
    total = pass_count + warn_count + fail_count
    if total == 0:
        overall = "NO_TESTS_RAN"
    elif fail_count > 0:
        overall = "FAIL"
    elif warn_count > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    results["summary"] = {
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "overall": overall,
    }

    # Confidence adjustment: reduce by 0.05 per FAIL, 0.02 per WARN
    results["confidence_adjustment"] = round(
        min(0.5, fail_count * 0.05 + warn_count * 0.02), 4
    )

    return results


def _pvalue_flag(pval: float, alpha: float = 0.05) -> str:
    """Convert p-value to PASS/WARN/FAIL flag."""
    if pval >= alpha:
        return "PASS"
    elif pval >= alpha * 0.5:
        return "WARN"
    else:
        return "FAIL"
