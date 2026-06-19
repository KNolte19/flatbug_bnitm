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
    MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "flat_bug_M.pt")
    _MAX_CONTENT_LENGTH_RAW = os.getenv("MAX_CONTENT_LENGTH")
    try:
        MAX_CONTENT_LENGTH = int(_MAX_CONTENT_LENGTH_RAW) if _MAX_CONTENT_LENGTH_RAW else None
    except ValueError:
        MAX_CONTENT_LENGTH = None
    ENABLE_CLASSIFIER = _get_bool("ENABLE_CLASSIFIER", False)
    ENABLE_PERSISTENCE = _get_bool("ENABLE_PERSISTENCE", False)
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
