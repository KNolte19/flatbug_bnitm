FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    UPLOAD_DIR=/tmp/uploads \
    OUTPUT_DIR=/tmp/outputs \
    MODEL_WEIGHTS=flat_bug_M.pt \
    ENABLE_CLASSIFIER=false \
    ENABLE_PERSISTENCE=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "webapp.wsgi:app"]
