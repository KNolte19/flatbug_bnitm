from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, current_app, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

main_bp = Blueprint("main", __name__)


def _allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in current_app.config["ALLOWED_EXTENSIONS"]


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.post("/")
def upload_and_predict():
    file = request.files.get("image")
    if file is None or file.filename is None or file.filename == "":
        return render_template("error.html", message="Please select an image to upload."), 400

    if not _allowed_file(file.filename):
        return render_template("error.html", message="Unsupported file type."), 400

    upload_id = str(uuid4())
    safe_name = secure_filename(file.filename)
    upload_dir = Path(current_app.config["UPLOAD_DIR"]) / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / safe_name
    file.save(upload_path)

    repository = current_app.extensions["prediction_repository"]
    inference_service = current_app.extensions["inference_service"]

    upload_meta = {
        "upload_id": upload_id,
        "filename": safe_name,
        "upload_path": str(upload_path),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    repository.save_upload(upload_meta)

    try:
        prediction = inference_service.run_inference(str(upload_path))
    except (FileNotFoundError, RuntimeError, ValueError, OSError):
        current_app.logger.exception("Inference failed for upload %s", upload_id)
        return render_template("error.html", message="Inference failed. Please try again."), 500

    prediction["filename"] = safe_name
    repository.save_prediction(prediction)

    return redirect(url_for("predictions.show_predictions", prediction_id=prediction["prediction_id"]))
