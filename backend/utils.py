"""Shared utility helpers for the Flask backend."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSICAL_RESULTS = os.path.join(BASE_DIR, "classical_models", "results")
QUANTUM_RESULTS = os.path.join(BASE_DIR, "quantum_models", "results")


class DataAccessError(Exception):
    """Raised when an expected results file is missing or malformed."""


def safe_read_csv(path: str) -> pd.DataFrame:
    """Reads a CSV file, raising a clear DataAccessError on failure."""
    if not os.path.exists(path):
        raise DataAccessError(f"Required results file not found: {path}. "
                               f"Run the corresponding training pipeline first.")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataAccessError(f"Failed to read {path}: {exc}") from exc


def df_to_records(df: pd.DataFrame) -> list:
    """Converts a DataFrame to a JSON-serializable list of dicts, replacing NaN."""
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def round_numeric(record: Dict[str, Any], digits: int = 4) -> Dict[str, Any]:
    """Rounds numeric values in a flat dict for cleaner API responses."""
    rounded = {}
    for key, value in record.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        else:
            rounded[key] = value
    return rounded
