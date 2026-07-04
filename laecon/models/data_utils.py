"""Data loading utilities for LAECON models.

Supports loading data from:
  - DuckDB tables (via latade MCP — calling execute_sql)
  - CSV file paths
  - Inline data dicts (for testing)

Every loader includes DataFrame empty guards (DR-1, Constitution Art. 3.6).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_dataset(
    data_source: str,
    source_type: str = "auto",
    sql_query: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Load a dataset from a DuckDB table, CSV path, or inline reference.

    Args:
        data_source: DuckDB table name, CSV file path, or database path.
        source_type: One of "auto", "csv", "duckdb". When "auto", tries
                     CSV first, then DuckDB.
        sql_query: Optional SQL query override for DuckDB. If None and
                   source_type is "duckdb", uses SELECT * FROM <data_source>.
        db_path: Optional DuckDB file path for persistence. Defaults to
                 :memory: or env LATADE_DB_PATH.

    Returns:
        pd.DataFrame with the loaded data.

    Raises:
        ValueError: If the dataset is empty after loading (DR-1 guard).
        FileNotFoundError: If CSV path does not exist.
        RuntimeError: If DuckDB query fails.
    """
    df: pd.DataFrame | None = None
    resolved = source_type

    if resolved == "auto":
        # Try CSV first, then DuckDB
        csv_path = Path(data_source)
        if csv_path.suffix.lower() in (".csv", ".tsv", ".txt") and csv_path.exists():
            resolved = "csv"
        else:
            resolved = "duckdb"

    if resolved == "csv":
        csv_path = Path(data_source)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV path does not exist: {data_source}")
        df = pd.read_csv(csv_path)

    elif resolved == "duckdb":
        # Use DuckDB via latade execute_sql (imported at call time)
        # We cannot import latade directly (it's an MCP tool), so we
        # embed the DuckDB query logic using duckdb directly if available.
        try:
            import duckdb
        except ImportError:
            # Fallback: if duckdb not in this venv, attempt using latade MCP
            raise RuntimeError(
                "DuckDB loading requires 'duckdb' package or latade MCP. "
                "Install duckdb or use CSV path instead."
            )

        resolved_db = db_path or os.environ.get("LATADE_DB_PATH") or ":memory:"
        query = sql_query or f"SELECT * FROM {_safe_ident(data_source)}"

        try:
            con = duckdb.connect(resolved_db)
            df = con.execute(query).fetchdf()
            con.close()
        except Exception as exc:
            raise RuntimeError(
                f"DuckDB query failed: {exc}. "
                f"Table='{data_source}', db_path='{resolved_db}'"
            ) from exc

    else:
        raise ValueError(f"Unknown source_type: {source_type}. Use 'csv' or 'duckdb'.")

    # --- Empty DataFrame guard (DR-1) ---
    if df is None or df.empty:
        raise ValueError(
            f"[DR-1] Dataset is empty: source='{data_source}', "
            f"source_type='{resolved}'. "
            "Cannot proceed with empty data. "
            "Check that the source has rows and the query is correct."
        )

    return df


def _safe_ident(name: str) -> str:
    """Quote an identifier safely for SQL."""
    return f'"{name}"'


def check_empty(df: pd.DataFrame, context: str = "DataFrame") -> None:
    """Check if a DataFrame is empty and raise ValueError (DR-1 guard).

    Args:
        df: DataFrame to check.
        context: Description for error message.

    Raises:
        ValueError: If df is None or empty.
    """
    if df is None or df.empty:
        raise ValueError(
            f"[DR-1] {context} is empty. "
            "Cannot proceed with empty data."
        )
