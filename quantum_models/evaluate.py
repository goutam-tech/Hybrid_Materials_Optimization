"""Orchestrates the full quantum pipeline: load -> preprocess -> train QNN,
VQR, QKR -> quantum ensemble -> QAOA optimize -> persist CSV outputs."""

from __future__ import annotations

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "classical_models"))

from data_loader import DataLoader  # noqa: E402
from preprocessing import Preprocessor, FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

from qnn_train import QNNRegressor  # noqa: E402
from vqr_train import VQRRegressor  # noqa: E402
from qkr_train import QKRRegressor  # noqa: E402
from qaoa_optimizer import QAOAOptimizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class QuantumPipeline:
    """End-to-end quantum ML + QAOA optimization pipeline."""

    def __init__(self) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()

    def run(self) -> None:
        logger.info("Loading datasets...")
        frames = self.loader.load_all()
        splits = self.preprocessor.run(frames)
        train_df, test_df, full_df = splits["train"], splits["test"], splits["full"]

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df[TARGET_COLUMN].values
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df[TARGET_COLUMN].values
        X_full = full_df[FEATURE_COLUMNS].values

        metrics_rows = []

        # --- QNN ---
        qnn = QNNRegressor()
        qnn_train_time = qnn.fit(X_train, y_train)
        qnn_test_preds, qnn_pred_time_test = qnn.predict(X_test)
        qnn_full_preds, qnn_pred_time = qnn.predict(X_full)
        qnn_metrics = qnn.evaluate(y_test, qnn_test_preds)
        metrics_rows.append({"model": "QNN", **qnn_metrics,
                              "training_time": qnn_train_time, "prediction_time": qnn_pred_time})

        # --- VQR ---
        vqr = VQRRegressor()
        vqr_train_time = vqr.fit(X_train, y_train)
        vqr_test_preds, _ = vqr.predict(X_test)
        vqr_full_preds, vqr_pred_time = vqr.predict(X_full)
        vqr_metrics = vqr.evaluate(y_test, vqr_test_preds)
        metrics_rows.append({"model": "VQR", **vqr_metrics,
                              "training_time": vqr_train_time, "prediction_time": vqr_pred_time})

        # --- QKR ---
        qkr = QKRRegressor()
        qkr_train_time = qkr.fit(X_train, y_train)
        qkr_test_preds, _ = qkr.predict(X_test)
        qkr_full_preds, qkr_pred_time = qkr.predict(X_full)
        qkr_metrics = qkr.evaluate(y_test, qkr_test_preds)
        metrics_rows.append({"model": "QKR", **qkr_metrics,
                              "training_time": qkr_train_time, "prediction_time": qkr_pred_time})

        # --- Quantum ensemble + predictions table ---
        quantum_ensemble = (qnn_full_preds + vqr_full_preds + qkr_full_preds) / 3.0
        predictions_df = pd.DataFrame({
            "material_id": full_df["material_id"],
            "formula": full_df["formula"],
            "actual": full_df[TARGET_COLUMN],
            "qnn_prediction": qnn_full_preds,
            "vqr_prediction": vqr_full_preds,
            "qkr_prediction": qkr_full_preds,
            "quantum_ensemble": quantum_ensemble,
        })
        predictions_df.to_csv(os.path.join(RESULTS_DIR, "quantum_predictions.csv"), index=False)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(RESULTS_DIR, "quantum_metrics.csv"), index=False)

        # --- QAOA optimization on the quantum ensemble scores ---
        optimizer = QAOAOptimizer()
        optimized_df = optimizer.optimize(predictions_df, score_column="quantum_ensemble")
        optimized_df.to_csv(os.path.join(RESULTS_DIR, "quantum_optimized.csv"), index=False)

        best_model_row = metrics_df.loc[metrics_df["R2"].idxmax()]
        best_material_row = optimized_df.iloc[0]

        logger.info("Best quantum model: %s (R2=%.4f)", best_model_row["model"], best_model_row["R2"])
        logger.info("Best quantum material: %s (objective=%.4f)",
                    best_material_row["material_id"], best_material_row["objective_score"])
        logger.info("Quantum pipeline complete. Results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    try:
        logger.info("Starting Quantum Pipeline...")
        QuantumPipeline().run()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pipeline failed: %s", str(e))
        sys.exit(1)