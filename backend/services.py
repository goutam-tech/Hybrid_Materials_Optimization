"""Service layer: reads pipeline result CSVs and assembles API-ready payloads."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import pandas as pd

from utils import CLASSICAL_RESULTS, QUANTUM_RESULTS, DataAccessError, df_to_records, round_numeric

logger = logging.getLogger(__name__)


class ResultsService:
    """Provides structured access to classical/quantum metrics and rankings."""

    # ---- Classical ----

    def get_classical_metrics(self) -> List[Dict[str, Any]]:
        df = self._read(os.path.join(CLASSICAL_RESULTS, "classical_metrics.csv"))
        return [round_numeric(r) for r in df_to_records(df)]

    def get_classical_predictions(self) -> List[Dict[str, Any]]:
        df = self._read(os.path.join(CLASSICAL_RESULTS, "classical_predictions.csv"))
        return [round_numeric(r) for r in df_to_records(df)]

    def get_classical_optimized(self) -> List[Dict[str, Any]]:
        df = self._read(os.path.join(CLASSICAL_RESULTS, "classical_optimized.csv"))
        return [round_numeric(r) for r in df_to_records(df)]

    # ---- Quantum ----

    def get_quantum_metrics(self) -> List[Dict[str, Any]]:
        df = self._read(os.path.join(QUANTUM_RESULTS, "quantum_metrics.csv"))
        return [round_numeric(r) for r in df_to_records(df)]

    def get_quantum_predictions(self) -> List[Dict[str, Any]]:
        df = self._read(os.path.join(QUANTUM_RESULTS, "quantum_predictions.csv"))
        return [round_numeric(r) for r in df_to_records(df)]

    def get_quantum_optimized(self) -> List[Dict[str, Any]]:
        df = self._read(os.path.join(QUANTUM_RESULTS, "quantum_optimized.csv"))
        return [round_numeric(r) for r in df_to_records(df)]

    # ---- Aggregations ----

    def get_classical_optimizer_summary(self) -> Dict[str, Any]:
        df = self._read(os.path.join(CLASSICAL_RESULTS, "classical_optimized.csv"))
        best = df.sort_values("objective_score", ascending=False).iloc[0]
        return {
            "optimizer": "Newton's Method",
            "best_material": best["material_id"],
            "formula": best.get("formula"),
            "objective_score": round(float(best["objective_score"]), 4),
            "runtime": round(float(df["optimization_time"].iloc[0]), 6),
            "iterations": int(best["iterations"]),
            "materials_ranked": len(df),
        }

    def get_quantum_optimizer_summary(self) -> Dict[str, Any]:
        df = self._read(os.path.join(QUANTUM_RESULTS, "quantum_optimized.csv"))
        best = df.sort_values("objective_score", ascending=False).iloc[0]
        return {
            "optimizer": "QAOA",
            "best_material": best["material_id"],
            "formula": best.get("formula"),
            "objective_score": round(float(best["objective_score"]), 4),
            "runtime": round(float(df["runtime"].iloc[0]), 6),
            "iterations": int(df["iterations"].iloc[0]),
            "materials_ranked": len(df),
        }

    def get_recommendations(self, top_n: int = 10) -> Dict[str, Any]:
        classical_df = self._read(os.path.join(CLASSICAL_RESULTS, "classical_optimized.csv"))
        quantum_df = self._read(os.path.join(QUANTUM_RESULTS, "quantum_optimized.csv"))

        classical_top = classical_df.sort_values("rank").head(top_n)
        quantum_top = quantum_df.sort_values("rank").head(top_n)

        best_classical = classical_top.iloc[0]
        best_quantum = quantum_top.iloc[0]

        final_pick = best_quantum if best_quantum["objective_score"] >= best_classical["objective_score"] \
            else best_classical

        top_materials = []
        for _, row in classical_top.iterrows():
            top_materials.append({
                "material_id": row["material_id"],
                "formula": row.get("formula"),
                "predicted_score": round(float(row["initial_score"]), 4),
                "optimized_score": round(float(row["optimized_score"]), 4),
                "rank": int(row["rank"]),
                "source": "classical",
            })

        return {
            "best_classical_material": {
                "material_id": best_classical["material_id"],
                "formula": best_classical.get("formula"),
                "objective_score": round(float(best_classical["objective_score"]), 4),
            },
            "best_quantum_material": {
                "material_id": best_quantum["material_id"],
                "formula": best_quantum.get("formula"),
                "objective_score": round(float(best_quantum["objective_score"]), 4),
            },
            "final_recommended_material": {
                "material_id": final_pick["material_id"],
                "formula": final_pick.get("formula"),
                "objective_score": round(float(final_pick["objective_score"]), 4),
            },
            "top_materials": top_materials,
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        classical_metrics = pd.DataFrame(self.get_classical_metrics())
        quantum_metrics = pd.DataFrame(self.get_quantum_metrics())
        classical_opt = self.get_classical_optimizer_summary()
        quantum_opt = self.get_quantum_optimizer_summary()

        best_classical_model = classical_metrics.loc[classical_metrics["R2"].idxmax(), "model"]
        best_quantum_model = quantum_metrics.loc[quantum_metrics["R2"].idxmax(), "model"]

        best_optimizer = "QAOA" if quantum_opt["objective_score"] >= classical_opt["objective_score"] \
            else "Newton's Method"
        best_material = quantum_opt["best_material"] if best_optimizer == "QAOA" \
            else classical_opt["best_material"]

        full_classical = self._read(os.path.join(CLASSICAL_RESULTS, "classical_predictions.csv"))

        return {
            "total_materials": len(full_classical),
            "total_models": len(classical_metrics) + len(quantum_metrics),
            "best_classical_model": best_classical_model,
            "best_quantum_model": best_quantum_model,
            "best_optimizer": best_optimizer,
            "best_material": best_material,
            "classical_metrics": classical_metrics.to_dict(orient="records"),
            "quantum_metrics": quantum_metrics.to_dict(orient="records"),
            "classical_optimizer": classical_opt,
            "quantum_optimizer": quantum_opt,
        }

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataAccessError(f"Failed reading {path}: {exc}") from exc
        except FileNotFoundError as exc:
            raise DataAccessError(
                f"{path} not found. Run classical_models/evaluate.py and "
                f"quantum_models/evaluate.py first."
            ) from exc
