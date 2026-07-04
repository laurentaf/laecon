"""LAECON models package — econometric and ML model implementations.

Modules:
  data_utils:     Load data from DuckDB (via latade) or CSV paths
  registry:       Save/load model metadata and artifacts
  regression:     OLS, GLS, WLS with robust SE
  classifier:     Logit, probit, ordered, grouped
  diagnostics:    Assumption tests (hetero, auto, VIF, normality)
  interpretation: Marginal effects, odds ratios, partial dependence
  evaluation:     In-sample and out-of-sample metrics
  prediction:     Apply model to new data
  reporting:      Diagnostic report with plots
"""

from laecon.models.data_utils import load_dataset

__all__ = ["load_dataset"]
