"""Newton's Method optimizer applied to the classical ensemble predictions.

For each material, the ensemble score from RF/XGB/GB is refined with a
single-variable Newton's Method search that maximizes a smooth surrogate
objective built around the ensemble prediction. This simulates a local
materials-property optimization step (e.g. tuning toward a target property
value) and produces a final ranking of all candidate materials.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target value the optimizer tries to steer the (scaled) property towards.
TARGET_VALUE = 2.0
MAX_ITER = 50
TOLERANCE = 1e-6


class NewtonOptimizationError(Exception):
    """Raised when Newton's Method fails to converge for a material."""


class NewtonOptimizer:
    """Applies Newton's Method to refine each material's ensemble score."""

    def __init__(self, target_value: float = TARGET_VALUE, max_iter: int = MAX_ITER) -> None:
        self.target_value = target_value
        self.max_iter = max_iter

    @staticmethod
    def _objective(x: float, target: float) -> float:
        """f(x) = (x - target)^2 -- minimized at x = target."""
        return (x - target) ** 2

    @staticmethod
    def _first_derivative(x: float, target: float) -> float:
        return 2.0 * (x - target)

    @staticmethod
    def _second_derivative(_x: float, _target: float) -> float:
        return 2.0

    def _newton_refine(self, x0: float) -> Tuple[float, int]:
        """Runs Newton's Method: x_{n+1} = x_n - f'(x_n) / f''(x_n)."""
        x = x0
        for i in range(self.max_iter):
            f_prime = self._first_derivative(x, self.target_value)
            f_double_prime = self._second_derivative(x, self.target_value)
            if f_double_prime == 0:
                raise NewtonOptimizationError("Zero second derivative encountered.")
            x_next = x - f_prime / f_double_prime
            if abs(x_next - x) < TOLERANCE:
                return x_next, i + 1
            x = x_next
        return x, self.max_iter

    def optimize(self, df: pd.DataFrame, score_column: str = "ensemble_score") -> pd.DataFrame:
        results = []
        start = time.perf_counter()
        for _, row in df.iterrows():
            x0 = float(row[score_column])
            refined, iters = self._newton_refine(x0)
            objective_score = -self._objective(refined, self.target_value)
            results.append({
                "material_id": row["material_id"],
                "formula": row["formula"],
                "initial_score": x0,
                "optimized_score": refined,
                "objective_score": objective_score,
                "iterations": iters,
            })
        elapsed = time.perf_counter() - start

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("objective_score", ascending=False).reset_index(drop=True)
        result_df["rank"] = np.arange(1, len(result_df) + 1)
        result_df["optimization_time"] = elapsed
        logger.info("Newton optimization completed for %d materials in %.4fs", len(df), elapsed)
        return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    demo = pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(10)],
        "formula": [f"Compound{i}" for i in range(10)],
        "ensemble_score": rng.uniform(0, 5, 10),
    })
    optimizer = NewtonOptimizer()
    out = optimizer.optimize(demo)
    print(out)
