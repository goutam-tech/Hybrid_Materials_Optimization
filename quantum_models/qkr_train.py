"""Quantum Kernel Regressor (QKR) implemented with Qiskit's FidelityQuantumKernel.

A ZZFeatureMap encodes the (PCA-reduced) material features into a quantum
state; pairwise state-fidelity defines a quantum kernel matrix which is fed
into a classical Kernel Ridge regressor (precomputed-kernel mode).
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

N_QUBITS = 4
RANDOM_STATE = 42
ALPHA = 0.2


class QKRRegressor:
    """Quantum-kernel-based Kernel Ridge regressor."""

    def __init__(self, n_qubits: int = N_QUBITS, random_state: int = RANDOM_STATE) -> None:
        self.n_qubits = n_qubits
        self.random_state = random_state
        self.pca = PCA(n_components=n_qubits, random_state=random_state)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
        self.kernel = FidelityQuantumKernel(feature_map=feature_map)
        self.model = KernelRidge(kernel="precomputed", alpha=ALPHA)
        self.X_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        X_pca = self.pca.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_pca)
        self.X_train_ = X_scaled

        start = time.perf_counter()
        K_train = self.kernel.evaluate(x_vec=X_scaled)
        self.model.fit(K_train, y)
        elapsed = time.perf_counter() - start
        logger.info("QKR trained in %.4fs", elapsed)
        return elapsed

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        X_pca = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)
        start = time.perf_counter()
        K_test = self.kernel.evaluate(x_vec=X_scaled, y_vec=self.X_train_)
        preds = self.model.predict(K_test)
        elapsed = time.perf_counter() - start
        return preds, elapsed

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.append("../classical_models")
    from data_loader import DataLoader  # noqa: E402
    from preprocessing import Preprocessor, FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

    loader = DataLoader()
    frames = loader.load_all()
    pre = Preprocessor()
    splits = pre.run(frames)

    qkr = QKRRegressor()
    train_time = qkr.fit(splits["train"][FEATURE_COLUMNS].values, splits["train"][TARGET_COLUMN].values)
    preds, pred_time = qkr.predict(splits["test"][FEATURE_COLUMNS].values)
    print(qkr.evaluate(splits["test"][TARGET_COLUMN].values, preds))
