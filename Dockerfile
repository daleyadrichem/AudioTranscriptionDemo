FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# System dependencies
# - ffmpeg: audio conversion support
# - libsndfile1: required by soundfile
# - curl + unzip: required by download_vosk_model.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir -U pip uv

# Copy dependency metadata first for better layer caching
COPY pyproject.toml README.md /app/

# Copy app source
COPY src /app/src

# Copy model download script
COPY download_vosk_model.sh /app/download_vosk_model.sh
RUN chmod +x /app/download_vosk_model.sh

# Install project with API + Whisper extras
RUN uv sync --extra api --extra whisper

# Download Vosk model into /app/models
RUN MODELS_DIR=/app/models /app/download_vosk_model.sh vosk-model-small-en-us-0.15

# Pre-download Whisper base model
RUN uv run python -c "import whisper; whisper.load_model('base')"

ENV WHISPER_MODEL=base
ENV VOSK_MODEL_PATH=/app/models/vosk-model-small-en-us-0.15

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "audio_transcription_demo.api:app", "--host", "0.0.0.0", "--port", "8000"]