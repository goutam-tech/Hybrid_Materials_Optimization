"""Flask application entry point for the Hybrid Materials Optimization System."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, send_from_directory

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:  # pragma: no cover
    HAS_CORS = False

from api import api_bp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path="")

    if HAS_CORS:
        CORS(app)

    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return send_from_directory(DASHBOARD_DIR, "index.html")

    @app.route("/<path:page>")
    def dashboard_pages(page: str):
        full_path = os.path.join(DASHBOARD_DIR, page)
        if os.path.exists(full_path):
            return send_from_directory(DASHBOARD_DIR, page)
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"status": "error", "message": "Resource not found"}), 404

    @app.errorhandler(500)
    def server_error(_error):
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    logger.info("Flask app initialized. Dashboard served from %s", DASHBOARD_DIR)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
