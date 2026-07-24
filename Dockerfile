# Multi-agent assistant — Gradio web UI
# Build:  docker build -t multi-agents .
# Run:    docker run --rm -p 7860:7860 --env-file .env \
#             -v multi-agents-chroma:/app/chroma_db multi-agents

FROM python:3.12-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc files in the image
# - PYTHONUNBUFFERED: logs flush straight to the container output
# - GRADIO_SERVER_NAME: bind to all interfaces so the port is reachable from the host
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Persisted Chroma store lives here; mount a volume to keep it across runs.
VOLUME ["/app/chroma_db"]

EXPOSE 7860

CMD ["python3", "app.py"]
