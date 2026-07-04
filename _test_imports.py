"""Quick test: verify all modules import and basic regression works."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from laecon.models.data_utils import load_dataset, check_empty
from laecon.models.registry import save_model, load_model, compute_confidence
from laecon.models.regression import train_regression
from laecon.models.classifier import train_classifier
from laecon.models.diagnostics import validate_assumptions
from laecon.models.interpretation import interpret_model
from laecon.models.evaluation import evaluate_model
from laecon.models.prediction import predict
from laecon.models.reporting import export_diagnostic_report

print("All modules imported OK")

# Quick test: train_regression with fake data
np.random.seed(42)
n = 200
x1 = np.random.randn(n)
df = pd.DataFrame({
    "x1": x1,
    "x2": np.random.randn(n),
    "y": 1.0 + 0.5 * x1 + 0.3 * np.random.randn(n),
})

result = train_regression(df, "y", ["x1"], "test_model_001", algorithm="ols")
print(f"Regression status: {result['status']}")
print(f"Coefficients: {len(result['coefficients'])} terms")
print(f"R2: {result['metrics']['r_squared']}")
print(f"Confidence: {result['confidence']}")
assert "_fitted" in result, "_fitted key missing!"
print("_fitted present: OK")

# Save to registry
meta = {
    "algorithm": "ols",
    "features": ["x1"],
    "target": "y",
    "metrics": result["metrics"],
    "params": {"se_type": "homoskedastic"},
    "extra": {"n_obs": result["metrics"]["n_obs"]},
}
reg = save_model("test_model_001", result["_fitted"], meta)
print(f"Registry saved: {reg['registry_path']}")

# Load back
model_obj, loaded_reg = load_model("test_model_001")
print(f"Model loaded back: {loaded_reg['model_id']}")

# Test classifier
df_cls = pd.DataFrame({
    "x1": np.random.randn(300),
    "y": (np.random.randn(300) > 0).astype(int),
})
cls_result = train_classifier(df_cls, "y", ["x1"], "test_cls_001", family="logit")
print(f"Classifier status: {cls_result['status']}")
print(f"Classifier coefs: {len(cls_result['coefficients'])}")
print(f"Odds ratios: {len(cls_result['odds_ratios'])}")

# Test assumptions (with data)
model_obj, reg = load_model("test_model_001")
diag = validate_assumptions(
    "test_model_001",
    resid=model_obj.resid,
    exog=model_obj.model.exog,
    feature_names=["x1"],
)
print(f"Diagnostics overall: {diag['summary']['overall']}")
print(f"Tests: {diag['summary']['pass_count']} pass, "
      f"{diag['summary']['warn_count']} warn, {diag['summary']['fail_count']} fail")
print(f"Confidence adj: {diag['confidence_adjustment']}")

# Debug the model class
print(f"Model class after load: {type(model_obj).__name__}")
print(f"Has rsquared: {hasattr(model_obj, 'rsquared')}")
print(f"Model type check: OLS: {'OLS' in type(model_obj).__name__}")

# Test evaluation
eval_result = evaluate_model(
    "test_model_001",
    model_obj=model_obj,
    X_train=model_obj.model.exog,
    y_train=model_obj.model.endog,
)
print(f"Eval result keys: {list(eval_result.keys())}")
print(f"In-sample keys: {list(eval_result.get('in_sample', {}).keys())}")
print(f"Eval in-sample R2: {eval_result['in_sample'].get('r_squared', 'MISSING')}")

# Test interpretation
interp = interpret_model(
    "test_model_001",
    model_obj=model_obj,
    df=df,
    features=["x1"],
    target="y",
)
print(f"Interpretation status: {interp['status']}")
print(f"Elasticities: {interp.get('elasticities', [])}")

# Test prediction
pred_result = predict(
    "test_model_001",
    model_obj=model_obj,
    df=df.head(10),
    features=["x1"],
    return_ci=True,
)
print(f"Predictions: {len(pred_result['predictions'])} rows")
print(f"Has CI: {pred_result.get('confidence_intervals') is not None}")

# Test report
report = export_diagnostic_report(
    "test_model_001",
    model_obj=model_obj,
    df=df,
    features=["x1"],
    target="y",
    format_type="markdown",
    diagnostics=diag,
    evaluation=eval_result,
    interpretation=interp,
    registry_data=loaded_reg,
)
print(f"Report path: {report['report_path']}")
print(f"Plots: {len(report['plots'])} files")

print()
print("=== ALL TESTS PASSED ===")
