FROM python:3.11-slim

# System deps:
# - ffmpeg: for MP3/format conversion
# - libsndfile1: required by soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md /app/
COPY src /app/src

# Install the package + API deps.
# If you created an `api` extra: use `.[api]`
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir ".[api,whisper]"

# 🔽 PRE-DOWNLOAD WHISPER MODEL
RUN python - <<EOF
import whisper
whisper.load_model("base")
EOF

# If you want Whisper/SpeechBrain inside the image, you can instead do:
# RUN pip install --no-cache-dir ".[api,whisper,speechbrain]"

ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=base

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "audio_transcription_demo.api:app", "--host", "0.0.0.0", "--port", "8000"]
