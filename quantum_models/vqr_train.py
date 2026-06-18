"""Variational Quantum Regressor (VQR) implemented with PennyLane.

Differs from the QNN model by using a StronglyEntanglingLayers ansatz over
an angle-embedded feature map, with a separate observable (sum of PauliZ
expectation values) used as the regression output. Trained with PennyLane's
gradient-descent optimizer.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

N_QUBITS = 4
N_LAYERS = 2
N_EPOCHS = 25
LEARNING_RATE = 0.2
RANDOM_STATE = 42


class VQRRegressor:
    """Variational Quantum Regressor using StronglyEntanglingLayers."""

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
                 random_state: int = RANDOM_STATE) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state
        self.pca = PCA(n_components=n_qubits, random_state=random_state)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = None
        self.bias = pnp.array(0.0, requires_grad=True)
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        @qml.qnode(self.dev)
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0) + qml.PauliZ(2))

        self._circuit = circuit

    def _scale_inputs(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
        rng = np.where((self.x_max - self.x_min) == 0, 1.0, self.x_max - self.x_min)
        return (X - self.x_min) / rng * np.pi

    def _scale_targets(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.y_min, self.y_max = y.min(), y.max()
        rng = (self.y_max - self.y_min) if (self.y_max - self.y_min) != 0 else 1.0
        return 2 * (y - self.y_min) / rng - 1

    def _unscale_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        rng = (self.y_max - self.y_min) if (self.y_max - self.y_min) != 0 else 1.0
        return (y_scaled + 1) / 2 * rng + self.y_min

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        np.random.seed(self.random_state)
        X_pca = self.pca.fit_transform(X)
        X_scaled = self._scale_inputs(X_pca, fit=True)
        y_scaled = self._scale_targets(y, fit=True)

        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        self.weights = pnp.array(np.random.uniform(0, 2 * np.pi, shape), requires_grad=True)

        opt = qml.GradientDescentOptimizer(stepsize=LEARNING_RATE)

        def cost(weights, bias):
            preds = pnp.stack([self._circuit(weights, x) + bias for x in X_scaled])
            return pnp.mean((preds - y_scaled) ** 2)

        start = time.perf_counter()
        for epoch in range(N_EPOCHS):
            self.weights, self.bias = opt.step(lambda w, b: cost(w, b), self.weights, self.bias)
            if (epoch + 1) % 10 == 0:
                logger.info("VQR epoch %d/%d - loss=%.6f", epoch + 1, N_EPOCHS, cost(self.weights, self.bias))
        elapsed = time.perf_counter() - start
        logger.info("VQR trained in %.4fs", elapsed)
        return elapsed

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        X_pca = self.pca.transform(X)
        X_scaled = self._scale_inputs(X_pca, fit=False)
        start = time.perf_counter()
        preds_scaled = np.array([float(self._circuit(self.weights, x) + self.bias) for x in X_scaled])
        elapsed = time.perf_counter() - start
        preds = self._unscale_targets(preds_scaled)
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

    vqr = VQRRegressor()
    train_time = vqr.fit(splits["train"][FEATURE_COLUMNS].values, splits["train"][TARGET_COLUMN].values)
    preds, pred_time = vqr.predict(splits["test"][FEATURE_COLUMNS].values)
    print(vqr.evaluate(splits["test"][TARGET_COLUMN].values, preds))
