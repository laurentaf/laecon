"""Model evaluation — in-sample and out-of-sample metrics.

In-sample: R², adj-R², AIC, BIC, log-likelihood.
Out-of-sample: RMSE, MAE, log-loss, AUC, confusion matrix, classification report.
Includes baseline comparison (Constitution Art. 3.5).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)

from laecon.models.data_utils import check_empty


def evaluate_model(
    model_id: str,
    model_obj: Any,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    df_test: pd.DataFrame | None = None,
    target: str | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a trained model's performance.

    Args:
        model_id: Model identifier.
        model_obj: Fitted statsmodels result object.
        X_train: Training feature matrix (with const).
        y_train: Training target values.
        X_test: Test feature matrix (with const).
        y_test: Test target values.
        df_test: Alternative to X_test/y_test — test DataFrame.
        target: Target column name (needed with df_test).
        features: Feature column names (needed with df_test).

    Returns:
        dict with:
          - status, model_id
          - in_sample: metrics computed from the model object
          - out_of_sample: metrics computed on test data (if provided)
          - baseline_comparison: comparison against null model (Art. 3.5)
          - classification: classification metrics (for classifiers)
    """
    result: dict[str, Any] = {
        "status": "ok",
        "model_id": model_id,
    }

    # Detect model type by class name AND attributes
    model_class = type(model_obj).__name__
    # Statsmodels wraps results: RegressionResultsWrapper, BinaryResultsWrapper, etc.
    has_rsquared = hasattr(model_obj, "rsquared")
    has_llnull = hasattr(model_obj, "llnull")
    is_ols_basic = has_rsquared and not hasattr(model_obj, "model") or (
        hasattr(model_obj, "model")
        and hasattr(model_obj.model, "__class__")
        and "OLS" in type(model_obj.model).__name__
    )
    # Broader check: RegressionResultsWrapper (OLS, WLS, GLS)
    is_ols = has_rsquared or "RegressionResults" in model_class or "OLS" in model_class
    # Binary: Logit, Probit
    is_logit = hasattr(model_obj, "model") and "Logit" in type(model_obj.model).__name__
    is_probit = hasattr(model_obj, "model") and "Probit" in type(model_obj.model).__name__
    is_binary = is_logit or is_probit or has_llnull
    is_ordered = hasattr(model_obj, "model") and "OrderedModel" in type(model_obj.model).__name__
    is_multinomial = hasattr(model_obj, "model") and "MNLogit" in type(model_obj.model).__name__

    result["model_class"] = model_class

    # === In-sample metrics ===
    in_sample: dict[str, Any] = {}

    if is_ols:
        in_sample["r_squared"] = round(float(model_obj.rsquared), 6)
        in_sample["adj_r_squared"] = round(float(model_obj.rsquared_adj), 6)
        in_sample["aic"] = round(float(model_obj.aic), 4)
        in_sample["bic"] = round(float(model_obj.bic), 4)
        in_sample["log_likelihood"] = round(float(model_obj.llf), 4)
        in_sample["f_stat"] = round(float(model_obj.fvalue), 4)
        in_sample["f_pvalue"] = round(float(model_obj.f_pvalue), 6)
        in_sample["rmse"] = round(float(np.sqrt(model_obj.mse_resid)), 6)
        in_sample["mae"] = round(float(np.mean(np.abs(model_obj.resid))), 6)
        in_sample["n_obs"] = int(model_obj.nobs)

    elif is_binary:
        in_sample["log_likelihood"] = round(float(model_obj.llf), 4)
        in_sample["ll_null"] = round(float(model_obj.llnull), 4)
        if hasattr(model_obj, "llnull") and model_obj.llnull and model_obj.llnull != 0:
            in_sample["pseudo_r_squared"] = round(
                1 - float(model_obj.llf) / float(model_obj.llnull), 6
            )
        in_sample["aic"] = round(float(model_obj.aic), 4)
        in_sample["bic"] = round(float(model_obj.bic), 4)
        in_sample["n_obs"] = int(model_obj.nobs)

        # In-sample predictions
        try:
            y_pred_proba = model_obj.predict()
            in_sample_auc = roc_auc_score(model_obj.model.endog, y_pred_proba)
            in_sample["auc"] = round(float(in_sample_auc), 6)
        except Exception:
            pass

    elif is_ordered:
        in_sample["log_likelihood"] = round(float(model_obj.llf), 4)
        in_sample["aic"] = round(float(model_obj.aic), 4)
        in_sample["bic"] = round(float(model_obj.bic), 4)
        in_sample["n_obs"] = int(model_obj.nobs)

    result["in_sample"] = in_sample

    # === Out-of-sample metrics ===
    out_of_sample: dict[str, Any] = {}
    classification_metrics: dict[str, Any] = {}

    if X_test is not None and y_test is not None:
        check_empty(pd.DataFrame(y_test), "test data")

        try:
            if is_ols:
                y_pred = model_obj.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                out_of_sample["rmse"] = round(rmse, 6)
                out_of_sample["mae"] = round(mae, 6)
                out_of_sample["n_test"] = int(len(y_test))

            elif is_binary:
                y_pred_proba = model_obj.predict(X_test)
                y_pred_class = (y_pred_proba >= 0.5).astype(int)

                oos_auc = roc_auc_score(y_test, y_pred_proba)
                oos_logloss = log_loss(y_test, y_pred_proba)
                cm = confusion_matrix(y_test, y_pred_class)
                accuracy = accuracy_score(y_test, y_pred_class)

                out_of_sample["auc"] = round(float(oos_auc), 6)
                out_of_sample["log_loss"] = round(float(oos_logloss), 6)
                out_of_sample["accuracy"] = round(float(accuracy), 6)
                out_of_sample["n_test"] = int(len(y_test))

                classification_metrics["confusion_matrix"] = cm.tolist()
                classification_metrics["classification_report"] = classification_report(
                    y_test, y_pred_class, output_dict=True, zero_division=0
                )

            elif is_ordered:
                y_pred_proba = model_obj.predict(X_test)
                y_pred_class = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba >= 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred_class)
                out_of_sample["accuracy"] = round(float(accuracy), 6)
                out_of_sample["n_test"] = int(len(y_test))

                try:
                    cm = confusion_matrix(y_test, y_pred_class)
                    classification_metrics["confusion_matrix"] = cm.tolist()
                except Exception:
                    pass

        except Exception as exc:
            out_of_sample["error"] = str(exc)

    elif df_test is not None and target and features:
        check_empty(df_test, "test data")
        missing = [c for c in [target] + features if c not in df_test.columns]
        if missing:
            raise ValueError(f"Columns missing in test data: {missing}")

        test_df = df_test[[target] + features].dropna()
        check_empty(test_df, "test data after dropping NA")

        y_test_local = test_df[target].values
        X_test_local = sm.add_constant(
            test_df[features].values.astype(float), has_constant="add"
        )

        # Recursively call with the prepared arrays
        return evaluate_model(
            model_id=model_id,
            model_obj=model_obj,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test_local,
            y_test=y_test_local,
        )

    result["out_of_sample"] = out_of_sample
    if classification_metrics:
        result["classification"] = classification_metrics

    # === Baseline comparison (Art. 3.5) ===
    baseline: dict[str, Any] = {}
    if y_train is not None:
        y_mean = np.mean(y_train)
        null_rmse = float(np.sqrt(mean_squared_error(y_train, np.full_like(y_train, y_mean))))
        null_mae = float(mean_absolute_error(y_train, np.full_like(y_train, y_mean)))

        if is_ols:
            model_rmse = float(np.sqrt(model_obj.mse_resid))
            baseline["null_model_rmse"] = round(null_rmse, 6)
            baseline["model_rmse"] = round(model_rmse, 6)
            baseline["rmse_reduction_pct"] = round(
                (null_rmse - model_rmse) / null_rmse * 100, 2
            ) if null_rmse > 0 else None
            baseline["null_model_mae"] = round(null_mae, 6)

    result["baseline_comparison"] = baseline

    # === Confidence ===
    n = in_sample.get("n_obs", 0) or model_obj.nobs
    confidence = 0.5
    if n:
        confidence += min(0.20, 0.05 * (n ** 0.3))
    if is_ols:
        r2 = in_sample.get("r_squared", 0)
        confidence += min(0.15, r2 * 0.15)
    if is_binary:
        auc_val = out_of_sample.get("auc") or in_sample.get("auc")
        if auc_val:
            confidence += min(0.20, (auc_val - 0.5) * 0.4)
    result["confidence"] = round(min(1.0, max(0.0, confidence)), 4)

    return result
