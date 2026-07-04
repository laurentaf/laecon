"""Model registry for LAECON — save/load model metadata and artifacts.

Constitution Art. 6 (AE-1): every trained model gets a model_id.
Metadata is saved to artifacts/models/<model_id>/registry.json.

Artifacts:
  - registry.json: metadata (timestamp, algorithm, features, target, metrics, params)
  - model.pkl: the fitted statsmodels result pickle (if available)
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_MODELS_DIR = Path(os.environ.get(
    "LAECON_ARTIFACTS_DIR",
    Path(__file__).resolve().parent.parent.parent / "artifacts",
))

_REPORTS_DIR = _MODELS_DIR / "reports"
_PLOTS_DIR = _REPORTS_DIR  # per-model subdir


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(model_id: str) -> Path:
    """Return the model directory for a given model_id."""
    return _ensure_dir(_MODELS_DIR / "models" / model_id)


def report_path(model_id: str) -> Path:
    """Return the report directory for a given model_id."""
    return _ensure_dir(_REPORTS_DIR / model_id)


def plots_path(model_id: str) -> Path:
    """Return the plots subdirectory for a given model_id."""
    return _ensure_dir(report_path(model_id) / "plots")


def save_model(model_id: str, model_obj: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Save a fitted model and its metadata to the registry (AE-1).

    Args:
        model_id: Unique identifier for the model.
        model_obj: Fitted statsmodels result object (or any pickle-able object).
        metadata: Dict with keys: algorithm, features, target, metrics, params,
                  and any extra fields.

    Returns:
        dict with path info.
    """
    mdir = model_path(model_id)

    # Build registry.json
    registry: dict[str, Any] = {
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": metadata.get("algorithm", "unknown"),
        "features": metadata.get("features", []),
        "target": metadata.get("target", ""),
        "metrics": metadata.get("metrics", {}),
        "hyperparameters": metadata.get("params", {}),
        "extra": {k: v for k, v in metadata.items()
                  if k not in ("algorithm", "features", "target", "metrics", "params")},
    }

    reg_path = mdir / "registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str, ensure_ascii=False)

    # Save model pickle
    if model_obj is not None:
        pkl_path = mdir / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model_obj, f)

    return {
        "model_id": model_id,
        "registry_path": str(reg_path),
        "model_pkl_path": str(mdir / "model.pkl") if model_obj is not None else None,
    }


def load_model(model_id: str) -> tuple[Any, dict[str, Any]]:
    """Load a fitted model and its registry metadata.

    Args:
        model_id: Unique model identifier.

    Returns:
        Tuple of (model_obj, registry_dict).

    Raises:
        FileNotFoundError: If model_id not found in registry.
    """
    mdir = model_path(model_id)
    reg_path = mdir / "registry.json"

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Model '{model_id}' not found. "
            f"Expected registry at: {reg_path}"
        )

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    pkl_path = mdir / "model.pkl"
    model_obj = None
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            model_obj = pickle.load(f)

    return model_obj, registry


def compute_confidence(registry: dict[str, Any], n_obs: int | None = None) -> float:
    """Compute a heuristic confidence score for a model.

    Based on:
      - Sample size (more = higher confidence, diminishing returns)
      - R² (for regression)
      - Metrics like AUC (for classifiers)
      - Number of assumption checks passed (if available)

    Args:
        registry: Model registry dict with metrics.
        n_obs: Number of observations (if known).

    Returns:
        Float between 0.0 and 1.0.
    """
    metrics = registry.get("metrics", {})
    extras = registry.get("extra", {})

    score = 0.5  # baseline

    # Sample size contribution
    n = n_obs or metrics.get("n_obs", 0) or extras.get("n_obs", 0)
    if n > 0:
        # log scale: 100 obs -> 0.1, 1000 -> 0.15, 10000 -> 0.2
        score += min(0.20, 0.05 * (n ** 0.3))

    # R² contribution (regression)
    r2 = metrics.get("r_squared")
    if r2 is not None and r2 > 0:
        score += min(0.15, r2 * 0.15)

    # AUC contribution (classification)
    auc = metrics.get("auc") or extras.get("auc")
    if auc is not None:
        score += min(0.20, (auc - 0.5) * 0.4)

    # Assumption checks
    assumptions = extras.get("assumptions", {})
    if assumptions:
        passed = sum(1 for v in assumptions.values() if v == "PASS")
        total = len(assumptions)
        if total > 0:
            score += 0.10 * (passed / total)

    return round(min(1.0, max(0.0, score)), 4)
