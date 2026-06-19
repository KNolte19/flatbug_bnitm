import os

bind = f"0.0.0.0:{os.getenv('PORT', '10002')}"
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
accesslog = "-"
errorlog = "-"
