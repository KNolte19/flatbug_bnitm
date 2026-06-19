from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PredictionRepository(ABC):
    @abstractmethod
    def save_upload(self, upload_meta: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_prediction(self, prediction_meta: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class InMemoryPredictionRepository(PredictionRepository):
    def __init__(self) -> None:
        self._uploads: list[dict[str, Any]] = []
        self._predictions: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def save_upload(self, upload_meta: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._uploads.append(upload_meta)
        return upload_meta

    def save_prediction(self, prediction_meta: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._predictions.append(prediction_meta)
        return prediction_meta

    def list_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._predictions))[:limit]

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self._lock:
            for prediction in reversed(self._predictions):
                if prediction.get("prediction_id") == prediction_id:
                    return prediction
        return None


class FileSystemPredictionRepository(PredictionRepository):
    def __init__(self, output_dir: str) -> None:
        self._index_path = Path(output_dir) / "prediction_registry.json"
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read_state(self) -> dict[str, list[dict[str, Any]]]:
        if not self._index_path.exists():
            return {"uploads": [], "predictions": []}
        with self._index_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_state(self, state: dict[str, list[dict[str, Any]]]) -> None:
        with self._index_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def save_upload(self, upload_meta: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
            state.setdefault("uploads", []).append(upload_meta)
            self._write_state(state)
        return upload_meta

    def save_prediction(self, prediction_meta: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
            state.setdefault("predictions", []).append(prediction_meta)
            self._write_state(state)
        return prediction_meta

    def list_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            state = self._read_state()
            predictions = state.get("predictions", [])
            return list(reversed(predictions))[:limit]

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._read_state()
            for prediction in reversed(state.get("predictions", [])):
                if prediction.get("prediction_id") == prediction_id:
                    return prediction
        return None
