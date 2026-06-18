"""Orchestrates the full classical pipeline: load -> preprocess -> train all
three models -> ensemble -> Newton optimize -> persist CSV/PKL outputs."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from data_loader import DataLoader
from preprocessing import Preprocessor, TARGET_COLUMN
from train_rf import RandomForestTrainer
from train_xgboost import XGBoostTrainer
from train_gradient_boost import GradientBoostTrainer
from optimize_newton import NewtonOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class ClassicalPipeline:
    """End-to-end classical ML + optimization pipeline."""

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

        metrics_rows = []
        pred_frames = []

        # --- Random Forest ---
        rf = RandomForestTrainer()
        rf_train_time = rf.train(train_df)
        rf_preds, rf_pred_time = rf.predict(full_df)
        rf_test_preds, _ = rf.predict(test_df)
        rf_metrics = rf.evaluate(test_df, rf_test_preds)
        rf.save(os.path.join(MODELS_DIR, "rf.pkl"))
        metrics_rows.append({"model": "Random Forest", **rf_metrics,
                              "training_time": rf_train_time, "prediction_time": rf_pred_time})

        # --- XGBoost ---
        xg = XGBoostTrainer()
        xg_train_time = xg.train(train_df)
        xg_preds, xg_pred_time = xg.predict(full_df)
        xg_test_preds, _ = xg.predict(test_df)
        xg_metrics = xg.evaluate(test_df, xg_test_preds)
        xg.save(os.path.join(MODELS_DIR, "xgb.pkl"))
        metrics_rows.append({"model": "XGBoost", **xg_metrics,
                              "training_time": xg_train_time, "prediction_time": xg_pred_time})

        # --- Gradient Boosting ---
        gb = GradientBoostTrainer()
        gb_train_time = gb.train(train_df)
        gb_preds, gb_pred_time = gb.predict(full_df)
        gb_test_preds, _ = gb.predict(test_df)
        gb_metrics = gb.evaluate(test_df, gb_test_preds)
        gb.save(os.path.join(MODELS_DIR, "gb.pkl"))
        metrics_rows.append({"model": "Gradient Boosting", **gb_metrics,
                              "training_time": gb_train_time, "prediction_time": gb_pred_time})

        # --- Ensemble + predictions table ---
        ensemble_score = (rf_preds + xg_preds + gb_preds) / 3.0
        predictions_df = pd.DataFrame({
            "material_id": full_df["material_id"],
            "formula": full_df["formula"],
            "actual": full_df[TARGET_COLUMN],
            "rf_prediction": rf_preds,
            "xgb_prediction": xg_preds,
            "gb_prediction": gb_preds,
            "ensemble_score": ensemble_score,
        })
        predictions_df.to_csv(os.path.join(RESULTS_DIR, "classical_predictions.csv"), index=False)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(RESULTS_DIR, "classical_metrics.csv"), index=False)

        # --- Newton optimization on the ensemble scores ---
        optimizer = NewtonOptimizer()
        optimized_df = optimizer.optimize(predictions_df, score_column="ensemble_score")
        optimized_df.to_csv(os.path.join(RESULTS_DIR, "classical_optimized.csv"), index=False)

        best_model_row = metrics_df.loc[metrics_df["R2"].idxmax()]
        best_material_row = optimized_df.iloc[0]

        logger.info("Best classical model: %s (R2=%.4f)", best_model_row["model"], best_model_row["R2"])
        logger.info("Best classical material: %s (objective=%.4f)",
                    best_material_row["material_id"], best_material_row["objective_score"])
        logger.info("Classical pipeline complete. Results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    ClassicalPipeline().run()
