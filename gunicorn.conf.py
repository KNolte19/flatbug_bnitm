import os

bind = "0.0.0.0:10002"
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
accesslog = "-"
errorlog = "-"
