"""Generates the consolidated exports/results.xlsx workbook containing all
classical metrics, quantum metrics, optimizer results, top materials and the
final recommendations. Uses openpyxl via pandas' ExcelWriter."""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "backend"))

from services import ResultsService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPORT_PATH = os.path.join(BASE_DIR, "exports", "results.xlsx")


def build_workbook() -> None:
    service = ResultsService()

    classical_metrics = pd.DataFrame(service.get_classical_metrics())
    quantum_metrics = pd.DataFrame(service.get_quantum_metrics())
    classical_optimized = pd.DataFrame(service.get_classical_optimized())
    quantum_optimized = pd.DataFrame(service.get_quantum_optimized())

    optimizer_summary = pd.DataFrame([
        service.get_classical_optimizer_summary(),
        service.get_quantum_optimizer_summary(),
    ])

    recommendations = service.get_recommendations()
    top_materials = pd.DataFrame(recommendations["top_materials"])
    summary_rows = pd.DataFrame([
        {"category": "Best Classical Material", **recommendations["best_classical_material"]},
        {"category": "Best Quantum Material", **recommendations["best_quantum_material"]},
        {"category": "Final Recommended Material", **recommendations["final_recommended_material"]},
    ])

    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    with pd.ExcelWriter(EXPORT_PATH, engine="openpyxl") as writer:
        classical_metrics.to_excel(writer, sheet_name="Classical Metrics", index=False)
        quantum_metrics.to_excel(writer, sheet_name="Quantum Metrics", index=False)
        optimizer_summary.to_excel(writer, sheet_name="Optimizer Metrics", index=False)
        classical_optimized.to_excel(writer, sheet_name="Classical Ranking", index=False)
        quantum_optimized.to_excel(writer, sheet_name="Quantum Ranking", index=False)
        top_materials.to_excel(writer, sheet_name="Top Materials", index=False)
        summary_rows.to_excel(writer, sheet_name="Recommendations", index=False)

    logger.info("Workbook written to %s", EXPORT_PATH)


if __name__ == "__main__":
    build_workbook()
