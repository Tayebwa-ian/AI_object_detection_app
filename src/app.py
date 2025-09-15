#!/usr/bin/python3
"""Application initialization module.

Creates and configures the Flask application, mounts Swagger UI using the
`docs/swagger.json` Swagger 2.0 specification, registers API resources and
initializes the database session.

Usage:
    export OBJ_DETECT_USE_SQLITE=1
    export OBJ_DETECT_SQLITE_FILE=./data/obj_detect_dev.db
    python -m src.app
"""
import os
import json
from datetime import datetime
from flask import Flask, jsonify, make_response, send_from_directory
from flask_cors import CORS
from flasgger import Swagger
from os import getenv

from src import storage
from src.app_api import register_api

# location of Swagger JSON file
SWAGGER_JSON_REL = os.path.join(os.path.dirname(__file__), ".", "docs", "swagger_template.json")


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config: Optional dict to override Flask configuration.

    Returns:
        Flask app instance with routes and Swagger configured.
    """
    # set static_folder to docs so Swagger UI can fetch swagger.json if needed
    static_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    app = Flask(__name__, static_folder=static_dir)

    # Allow CORS for API endpoints (tune origins for production)
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/metrics": {"origins": "*"}})

    # Apply optional configuration overrides
    if config:
        app.config.update(config)

    # Ensure DB engine and session are ready
    storage.database.reload()

    # Load Swagger JSON for flasgger
    swagger_spec = None
    try:
        with open(os.path.abspath(SWAGGER_JSON_REL), "r", encoding="utf-8") as f:
            swagger_spec = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load swagger.json: {e}")
        swagger_spec = None

    if swagger_spec:
        Swagger(app, template=swagger_spec)
    else:
        # minimal Swagger config if file missing
        Swagger(app)

    # Register API resources
    register_api(app)

    # Serve the raw Swagger JSON at /docs/swagger.json
    @app.route("/docs/swagger.json")
    def swagger_json():
        """Serve the Swagger 2.0 JSON file used by Swagger UI."""
        return send_from_directory(app.static_folder, "swagger.json")

    # Health endpoint
    @app.route("/health")
    def health_check():
        """
        Health check endpoint.
        ---
        tags:
          - System
        responses:
          200:
            description: API health status
        """
        ok = "connected" if storage.database.session else "disconnected"
        return jsonify({
            "status": "healthy",
            "message": "AI Object Counting API is running",
            "database": ok,
            "timestamp": datetime.now().isoformat() + "Z"
        })

    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        """Return JSON 404 response."""
        return make_response(jsonify({"error": "Resource (endpoint) not found"}), 404)

    @app.errorhandler(400)
    def bad_request(e):
        """Return JSON 400 response."""
        return make_response(jsonify({"error": "Bad request"}), 400)

    # Ensure session closed on teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        try:
            storage.database.close()
        except Exception:
            pass

    return app


if __name__ == "__main__":
    host = getenv("OBJ_DETECT_API_HOST", "0.0.0.0")
    port = int(getenv("OBJ_DETECT_API_PORT", "5000"))
    debug = getenv("OBJ_DETECT_DEBUG", "1") == "1"
    app = create_app()
    app.run(host=host, port=port, debug=debug, threaded=True)
