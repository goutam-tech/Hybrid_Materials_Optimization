"""Trains an XGBoost Regressor on the unified materials dataset."""

from __future__ import annotations

import logging
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Wraps training, prediction and metric computation for XGBoost."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self.random_state = random_state

    def train(self, train_df: pd.DataFrame) -> float:
        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df[TARGET_COLUMN].values
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        logger.info("XGBoost trained in %.4fs", elapsed)
        return elapsed

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        X = df[FEATURE_COLUMNS].values
        start = time.perf_counter()
        preds = self.model.predict(X)
        elapsed = time.perf_counter() - start
        return preds, elapsed

    def evaluate(self, test_df: pd.DataFrame, preds: np.ndarray) -> Dict[str, float]:
        y_true = test_df[TARGET_COLUMN].values
        return {
            "MAE": float(mean_absolute_error(y_true, preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, preds))),
            "R2": float(r2_score(y_true, preds)),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        logger.info("Saved XGBoost model to %s", path)


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    from data_loader import DataLoader
    from preprocessing import Preprocessor

    loader = DataLoader()
    frames = loader.load_all()
    pre = Preprocessor()
    splits = pre.run(frames)

    trainer = XGBoostTrainer()
    train_time = trainer.train(splits["train"])
    preds, pred_time = trainer.predict(splits["test"])
    metrics = trainer.evaluate(splits["test"], preds)
    metrics["Training Time"] = train_time
    metrics["Prediction Time"] = pred_time
    print(metrics)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(out_dir, exist_ok=True)
    trainer.save(os.path.join(out_dir, "xgb.pkl"))
