"""Data validation, cleaning, merging and train/test splitting for the unified
materials dataframe used by both the classical and quantum pipelines."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import DataLoader

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_MATERIALS = 50
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "mean_atomic_mass", "mean_electronegativity", "mean_atomic_radius",
    "valence_electrons", "density", "packing_fraction", "formation_energy",
    "num_atoms_cell", "space_group", "avg_bond_length",
]

TARGET_COLUMN = "mp_gap"


class PreprocessingError(Exception):
    """Raised when the preprocessing pipeline cannot produce a valid dataset."""


class Preprocessor:
    """Cleans, merges and prepares the Matbench-style datasets for modeling."""

    def __init__(self, random_state: int = RANDOM_STATE, n_materials: int = N_MATERIALS) -> None:
        self.random_state = random_state
        self.n_materials = n_materials
        self.scaler = StandardScaler()

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset="material_id").reset_index(drop=True)
        df = df.dropna()
        return df

    def merge_datasets(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merges the three Matbench-style frames on `material_id`, validating
        and cleaning each before the join."""
        cleaned = {name: self._clean(df) for name, df in frames.items()}

        base = cleaned["mp_gap"][["material_id", "formula"] + FEATURE_COLUMNS + ["mp_gap"]]
        kvrh = cleaned["log_kvrh"][["material_id", "log_kvrh"]]
        gvrh = cleaned["log_gvrh"][["material_id", "log_gvrh"]]

        merged = base.merge(kvrh, on="material_id", how="inner")
        merged = merged.merge(gvrh, on="material_id", how="inner")
        merged = merged.dropna().reset_index(drop=True)

        if merged.empty:
            raise PreprocessingError("Merged dataframe is empty after join/cleaning.")

        logger.info("Unified dataframe shape after merge: %s", merged.shape)
        return merged

    def select_materials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects exactly N_MATERIALS rows (deterministic, via random_state)."""
        if len(df) < self.n_materials:
            raise PreprocessingError(
                f"Not enough materials ({len(df)}) to select {self.n_materials}."
            )
        selected = df.sample(n=self.n_materials, random_state=self.random_state).reset_index(drop=True)
        return selected

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = df.copy()
        scaled[FEATURE_COLUMNS] = self.scaler.fit_transform(df[FEATURE_COLUMNS])
        return scaled

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=self.random_state
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def run(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        merged = self.merge_datasets(frames)
        selected = self.select_materials(merged)
        scaled = self.scale_features(selected)
        train_df, test_df = self.split(scaled)
        logger.info("Train size: %d | Test size: %d", len(train_df), len(test_df))
        return {
            "full": scaled,
            "train": train_df,
            "test": test_df,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    frames = loader.load_all()
    pre = Preprocessor()
    result = pre.run(frames)
    for key, frame in result.items():
        print(key, frame.shape)
