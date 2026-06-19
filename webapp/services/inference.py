from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from bin.fb_predict import predict

from webapp.services.classifier import ClassifierService


class InferenceService:
    def __init__(self, output_dir: str, classifier_service: ClassifierService, model_weights: str = "flat_bug_M.pt") -> None:
        self.output_dir = Path(output_dir)
        self.classifier_service = classifier_service
        self.model_weights = model_weights

    def run_inference(self, upload_path: str) -> dict[str, Any]:
        prediction_id = str(uuid4())
        job_output_dir = self.output_dir / prediction_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        predict(
            input=upload_path,
            output_dir=str(job_output_dir),
            model_weights=self.model_weights,
            id=prediction_id,
            no_compiled_coco=False,
            verbose=False,
        )

        artifacts = sorted(
            path.relative_to(job_output_dir).as_posix()
            for path in job_output_dir.rglob("*")
            if path.is_file()
        )

        metadata_artifact = next((a for a in artifacts if a.endswith(".json") and Path(a).name.startswith("metadata_")), None)
        overview_artifacts = [a for a in artifacts if Path(a).name.startswith("overview_") and a.endswith(".jpg")]
        crop_artifacts = [a for a in artifacts if Path(a).name.startswith("crop_")]
        compiled_coco_artifact = next((a for a in artifacts if Path(a).name == "coco_instances.json"), None)

        metadata_summary = {}
        if metadata_artifact:
            metadata_path = job_output_dir / metadata_artifact
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            confs = metadata.get("confs", []) or []
            metadata_summary = {
                "image_width": metadata.get("image_width"),
                "image_height": metadata.get("image_height"),
                "detections": len(confs),
                "max_confidence": max(confs) if confs else None,
                "prediction_identifier": metadata.get("identifier"),
            }

        # Future CNN hook: pass extracted segment data once classifier integration is enabled.
        classifier_results = []
        if getattr(self.classifier_service, "enabled", False):
            classifier_results = self.classifier_service.classify_segments([])

        return {
            "prediction_id": prediction_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "upload_path": upload_path,
            "output_dir": str(job_output_dir),
            "artifacts": artifacts,
            "metadata_artifact": metadata_artifact,
            "overview_artifacts": overview_artifacts,
            "crop_artifacts": crop_artifacts,
            "compiled_coco_artifact": compiled_coco_artifact,
            "metadata_summary": metadata_summary,
            "classifier_results": classifier_results,
        }
