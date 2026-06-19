from __future__ import annotations

from pathlib import Path

from flask import Flask

from webapp.config import Config
from webapp.routes.main import main_bp
from webapp.routes.predictions import predictions_bp
from webapp.services.classifier import NoOpClassifierService
from webapp.services.inference import InferenceService
from webapp.services.persistence import FileSystemPredictionRepository, InMemoryPredictionRepository


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    if app.config["ENABLE_PERSISTENCE"]:
        repository = FileSystemPredictionRepository(app.config["OUTPUT_DIR"])
    else:
        repository = InMemoryPredictionRepository()

    classifier_service = NoOpClassifierService(enabled=app.config["ENABLE_CLASSIFIER"])
    inference_service = InferenceService(app.config["OUTPUT_DIR"], classifier_service)

    app.extensions["prediction_repository"] = repository
    app.extensions["classifier_service"] = classifier_service
    app.extensions["inference_service"] = inference_service

    app.register_blueprint(main_bp)
    app.register_blueprint(predictions_bp)

    return app
