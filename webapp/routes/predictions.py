from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, current_app, render_template, request, send_file

predictions_bp = Blueprint("predictions", __name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _get_repository():
    return current_app.extensions["prediction_repository"]


def _get_prediction(prediction_id: str | None):
    repository = _get_repository()
    if prediction_id:
        return repository.get_prediction(prediction_id)
    latest = repository.list_predictions(limit=1)
    return latest[0] if latest else None


def _resolve_artifact(prediction: dict, artifact_path: str) -> Path:
    allowed = set(prediction.get("artifacts", []))
    if artifact_path not in allowed:
        abort(404)
    output_dir = Path(prediction["output_dir"]).resolve()
    full_path = (output_dir / artifact_path).resolve()
    try:
        full_path.relative_to(output_dir)
    except ValueError:
        abort(404)
    if not full_path.exists() or not full_path.is_file():
        abort(404)
    return full_path


@predictions_bp.get("/predictions")
def show_predictions():
    prediction_id = request.args.get("prediction_id")
    prediction = _get_prediction(prediction_id)

    if prediction is None:
        return render_template("predictions.html", prediction=None, artifacts=[])

    image_artifacts = [a for a in prediction.get("artifacts", []) if Path(a).suffix.lower() in IMAGE_EXTENSIONS]

    return render_template(
        "predictions.html",
        prediction=prediction,
        artifacts=prediction.get("artifacts", []),
        image_artifacts=image_artifacts,
    )


@predictions_bp.get("/predictions/<prediction_id>/artifact/<path:artifact_path>")
def get_artifact(prediction_id: str, artifact_path: str):
    prediction = _get_prediction(prediction_id)
    if prediction is None:
        abort(404)
    return send_file(_resolve_artifact(prediction, artifact_path))


@predictions_bp.get("/predictions/<prediction_id>/download/<path:artifact_path>")
def download_artifact(prediction_id: str, artifact_path: str):
    prediction = _get_prediction(prediction_id)
    if prediction is None:
        abort(404)
    return send_file(_resolve_artifact(prediction, artifact_path), as_attachment=True)


@predictions_bp.get("/predictions/<prediction_id>/upload")
def get_uploaded_image(prediction_id: str):
    prediction = _get_prediction(prediction_id)
    if prediction is None:
        abort(404)
    upload_dir = Path(current_app.config["UPLOAD_DIR"]).resolve()
    upload_path = Path(prediction["upload_path"]).resolve()
    try:
        upload_path.relative_to(upload_dir)
    except ValueError:
        abort(404)
    if not upload_path.exists() or not upload_path.is_file():
        abort(404)
    return send_file(upload_path)
