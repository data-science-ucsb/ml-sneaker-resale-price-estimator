"""Flask application factory for the sneaker price estimator API.

Run with (see the `api` Makefile target):

    flask --app sneakerml.api.app run --port 5000
"""

from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from sneakerml.api import db as db_module
from sneakerml.api.routes import bp as api_bp
from sneakerml.config import DB_PATH, MODELS_DIR
from sneakerml.catalog import DEFAULT_CATALOG_PATH
from sneakerml.predict import PricePredictor

#: Origin the Vite dev server (Task 7's frontend) runs on.
FRONTEND_ORIGIN = "http://localhost:5173"


def create_app(config: dict | None = None) -> Flask:
    """Build and return the Flask app.

    `config`, if given, may override `DB_PATH`, `MODELS_DIR` and/or
    `CATALOG_PATH` -- this is how tests get an isolated, tmp_path-backed
    app per test instead of touching the real database/models on disk.
    The `PricePredictor` is loaded exactly once, here, at app-creation
    time (not per-request): loading it means deserializing three joblib
    models plus the catalog, which is too expensive to repeat on every
    request.
    """
    app = Flask(__name__)
    app.config["DB_PATH"] = DB_PATH
    app.config["MODELS_DIR"] = MODELS_DIR
    app.config["CATALOG_PATH"] = DEFAULT_CATALOG_PATH
    if config:
        app.config.update(config)

    CORS(app, origins=[FRONTEND_ORIGIN])

    app.predictor = PricePredictor.load(
        models_dir=app.config["MODELS_DIR"],
        catalog_path=app.config["CATALOG_PATH"],
    )

    app.teardown_appcontext(db_module.close_db)
    app.register_blueprint(api_bp)

    return app
