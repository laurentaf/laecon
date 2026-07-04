# Constitution — LAECON

**Version:** 0.1.0-BASIC | **Status:** Vigente

---

## Article I — Likelihood explicita
Every model has a known, documented likelihood function. Tree-based models only enter with SHAP (DA-2).

## Article II — Input/Output Contracts
All data read from `latade://gold/<table>` via DuckDB. All outputs written to deterministic paths with versioned schemas.

## Article III — Interpretability Before Accuracy
Coefficients with SE, p-values, CI before scoring metrics.

## Article IV — Mandatory Methodological Detail
Every model answers 7 questions (hypothesis, algorithm choice, hyperparameters, variable selection, expected quality, acceptable failure, retrain trigger) with cited references carrying numeric thresholds.

## Article V — Anti-Overfitting
Cross-validation required. Every metric comes with a baseline.

## Article VI — Empty-Data Guards
Every function operating on DataFrames checks for emptiness and returns a friendly message, never IndexError or ValueError (P0, padroes-entrega.md).

## Article VII — Model Registry
Every model gets a `model_id`, versioned in `models/registry.json`. History preserved across retrains.

## Article VIII — Deterministic Output Paths
`export_diagnostic_report` writes to a deterministic path consumable by LADESIGN and LAN8N (AE-3).

## Article IX — PR-1 (Calibration 20/10 vs 50/1)
Level-A rigor (global standard). Ratio = Δquality% / Δtime% ≥ 0.5. No PhD overkill, no 4th-year oversimplification.

## Non-Goals
- Deep learning (PyTorch, TensorFlow) — post-STABLE if demanded
- NLP/LLM, computer vision — outside LAOS scope
- Distributed big data (Spark, Dask) — DuckDB is sufficient
- Advanced causal inference (DiD, RDD, IV, PSM) — M3+ roadmap