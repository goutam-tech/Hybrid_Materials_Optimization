"""Data loading utilities for the classical ML pipeline."""

from __future__ import annotations

import gzip
import json
import logging
import os
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

DATASET_FILES: Dict[str, str] = {
    "mp_gap": "matbench_mp_gap.json.gz",
    "log_kvrh": "matbench_log_kvrh.json.gz",
    "log_gvrh": "matbench_log_gvrh.json.gz",
}


class DataLoaderError(Exception):
    """Raised when a Matbench-style dataset cannot be loaded."""


class DataLoader:
    """Loads the gzip-compressed JSON Matbench-style datasets into DataFrames."""

    def __init__(self, dataset_dir: str = DATASET_DIR) -> None:
        self.dataset_dir = dataset_dir

    def _load_single(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            raise DataLoaderError(f"Dataset file not found: {path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                records = json.load(f)
            df = pd.DataFrame.from_records(records)
            logger.info("Loaded %s -> %d rows, %d columns", filename, df.shape[0], df.shape[1])
            return df
        except (OSError, json.JSONDecodeError) as exc:
            raise DataLoaderError(f"Failed to load dataset {filename}: {exc}") from exc

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Loads all three Matbench-style datasets and returns them keyed by name."""
        frames: Dict[str, pd.DataFrame] = {}
        for key, filename in DATASET_FILES.items():
            frames[key] = self._load_single(filename)
        return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    data = loader.load_all()
    for name, frame in data.items():
        print(name, frame.shape)
