"""QAOA-based combinatorial optimizer for material ranking.

Formulates a QUBO over the top-N candidate materials (by quantum ensemble
score): maximize the sum of selected scores subject to a soft constraint of
selecting exactly K materials. Solves it with Qiskit's QAOA (sampler-based
MinimumEigenOptimizer) and merges the result back into a full ranking.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
# from qiskit.primitives import Sampler
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

TOP_N_FOR_QAOA = 6
SELECT_K = 3
PENALTY = 5.0
QAOA_REPS = 1
QAOA_MAXITER = 20
RANDOM_STATE = 42


class QAOAOptimizationError(Exception):
    """Raised when the QAOA-based optimization fails."""


class QAOAOptimizer:
    """Selects and ranks top materials using QAOA on a QUBO formulation."""

    def __init__(self, top_n: int = TOP_N_FOR_QAOA, select_k: int = SELECT_K,
                 penalty: float = PENALTY, reps: int = QAOA_REPS,
                 maxiter: int = QAOA_MAXITER) -> None:
        self.top_n = top_n
        self.select_k = select_k
        self.penalty = penalty
        self.reps = reps
        self.maxiter = maxiter

    def _build_qubo(self, scores: np.ndarray) -> QuadraticProgram:
        n = len(scores)
        qp = QuadraticProgram(name="material_selection")
        for i in range(n):
            qp.binary_var(name=f"x{i}")

        linear = {
            f"x{i}": float(-scores[i] + self.penalty * (1 - 2 * self.select_k))
            for i in range(n)
        }
        quadratic = {}
        for i in range(n):
            for j in range(i + 1, n):
                quadratic[(f"x{i}", f"x{j}")] = 2.0 * self.penalty
        qp.minimize(linear=linear, quadratic=quadratic)
        return qp

    def optimize(self, df: pd.DataFrame, score_column: str = "quantum_ensemble") -> pd.DataFrame:
        ranked = df.sort_values(score_column, ascending=False).reset_index(drop=True)
        subset = ranked.head(self.top_n).reset_index(drop=True)
        scores = subset[score_column].values

        qp = self._build_qubo(scores)
        # qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=self.maxiter), reps=self.reps)
        # solver = MinimumEigenOptimizer(qaoa)

        logger.info(
            "Starting QAOA: top_n=%d, select_k=%d, reps=%d, maxiter=%d",
            self.top_n,
            self.select_k,
            self.reps,
            self.maxiter,
        )

        start = time.perf_counter()

        try:
            qaoa = QAOA(
                sampler=StatevectorSampler(),
                optimizer=COBYLA(maxiter=min(self.maxiter, 30)),
                reps=min(self.reps, 1)
            )

            solver = MinimumEigenOptimizer(qaoa)

            # result = solver.solve(qp)
            try:
                result = solver.solve(qp)
            except Exception as e:
                logger.warning(
                    "QAOA failed (%s). Falling back to classical ranking.",
                    str(e)
                )

                subset["selected"] = 0
                subset.loc[: self.select_k - 1, "selected"] = 1

                selection = subset["selected"].values

                subset["objective_score"] = (
                    subset[score_column] * subset["selected"]
                )

                return subset

        except KeyboardInterrupt:
            logger.warning("QAOA interrupted by user.")
            raise

        except Exception as e:
            logger.exception("QAOA failed: %s", str(e))
            raise QAOAOptimizationError(str(e))

        elapsed = time.perf_counter() - start
        # start = time.perf_counter()
        # result = solver.solve(qp)
        # elapsed = time.perf_counter() - start

        selection = np.array(result.x, dtype=int)
        n_iters = getattr(result.min_eigen_solver_result, "optimizer_evals", None)
        if n_iters is None:
            n_iters = self.maxiter

        subset = subset.copy()
        subset["selected"] = selection
        subset["objective_score"] = subset[score_column] * subset["selected"]

        selected_part = subset[subset["selected"] == 1].sort_values(score_column, ascending=False)
        unselected_part = subset[subset["selected"] == 0].sort_values(score_column, ascending=False)
        remainder = ranked.iloc[self.top_n:]

        final = pd.concat([selected_part, unselected_part, remainder], ignore_index=True)
        final["rank"] = np.arange(1, len(final) + 1)
        final["runtime"] = elapsed
        final["iterations"] = n_iters
        final["objective_score"] = final[score_column]

        logger.info(
            "QAOA optimization selected %d/%d materials in %.4fs (%d iterations)",
            int(selection.sum()), self.top_n, elapsed, n_iters,
        )
        return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    demo = pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(15)],
        "formula": [f"Compound{i}" for i in range(15)],
        "quantum_ensemble": rng.uniform(0, 5, 15),
    })
    optimizer = QAOAOptimizer()
    out = optimizer.optimize(demo)
    print(out[["material_id", "selected", "rank", "objective_score"]] if "selected" in out else out)
