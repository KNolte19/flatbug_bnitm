import os


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    PORT = int(os.getenv("PORT", "8000"))
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/outputs")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH")) if os.getenv("MAX_CONTENT_LENGTH") else None
    ENABLE_CLASSIFIER = _get_bool("ENABLE_CLASSIFIER", False)
    ENABLE_PERSISTENCE = _get_bool("ENABLE_PERSISTENCE", False)
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
