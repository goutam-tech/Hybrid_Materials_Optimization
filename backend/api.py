"""REST API blueprint exposing classical/quantum metrics, optimizers and
recommendations to the dashboard frontend."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify

from services import ResultsService
from utils import DataAccessError

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")
service = ResultsService()


def _handle(callable_fn):
    try:
        data = callable_fn()
        return jsonify({"status": "success", "data": data})
    except DataAccessError as exc:
        logger.error("Data access error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error")
        return jsonify({"status": "error", "message": "Internal server error", "detail": str(exc)}), 500


@api_bp.route("/classical-metrics", methods=["GET"])
def classical_metrics():
    return _handle(service.get_classical_metrics)


@api_bp.route("/classical-predictions", methods=["GET"])
def classical_predictions():
    return _handle(service.get_classical_predictions)


@api_bp.route("/quantum-metrics", methods=["GET"])
def quantum_metrics():
    return _handle(service.get_quantum_metrics)


@api_bp.route("/quantum-predictions", methods=["GET"])
def quantum_predictions():
    return _handle(service.get_quantum_predictions)


@api_bp.route("/classical-optimizer", methods=["GET"])
def classical_optimizer():
    return _handle(service.get_classical_optimizer_summary)


@api_bp.route("/quantum-optimizer", methods=["GET"])
def quantum_optimizer():
    return _handle(service.get_quantum_optimizer_summary)


@api_bp.route("/recommendations", methods=["GET"])
def recommendations():
    return _handle(service.get_recommendations)


@api_bp.route("/dashboard-summary", methods=["GET"])
def dashboard_summary():
    return _handle(service.get_dashboard_summary)
