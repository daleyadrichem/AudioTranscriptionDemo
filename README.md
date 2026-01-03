````markdown
# audio_transcription_demo

A modular, workshop-friendly speech-to-text (STT) demo application with clean architecture and fully local backends.

The project is designed so it can be:
- run **interactively** from the terminal (for workshops),
- exposed as a **FastAPI service**,
- deployed as a **Docker container**, making it easy to integrate later into
  a multi-agent or microservice-based system.

---

## What this project demonstrates

### Recognizer backends (models)
Selectable at runtime:
- **Vosk** – lightweight, fast, fully offline
- **Whisper** – higher accuracy, heavier (CPU or GPU)
- **SpeechBrain** – research-oriented ASR toolkit

All recognizers implement the same `SpeechRecognizer` interface.

### Audio sources
Selectable at runtime:
- **File source** – transcribe recorded audio (MP3, WAV, etc.)
- **Microphone source** – record short snippets from the mic (CLI mode only)

Sources implement a shared `AudioSource` interface.

### Use cases
Selectable at runtime:
- **Transcribe one recording**
- **Live transcription (repeat takes)**
- **Meeting minutes** (transcription + local heuristic summary)

Use cases orchestrate source → recognizer → output and are independent of the
underlying model or input method.

---

## Repository structure

```text
project-root/
  pyproject.toml
  README.md
  Dockerfile
  src/
    audio_transcription_demo/
      main.py          # interactive CLI
      api.py           # FastAPI server
      utils.py

      recognizers/     # models (vosk / whisper / speechbrain)
      sources/         # audio sources (file / microphone)
      use_cases/       # demo scenarios
````

---

## Requirements (local run)

### Python

* Python **3.10+**

### System dependencies

* **ffmpeg** (required for MP3 and general audio conversion)
* **PortAudio** (required only for microphone usage via `sounddevice`)

### Optional hardware

* Microphone (for mic source)
* GPU is optional; CPU-only works for all backends

---

## Installation (local)

### Using uv (recommended)

```bash
uv sync
```

Optional backends:

```bash
uv sync --extra whisper
uv sync --extra speechbrain
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

With extras:

```bash
pip install -e ".[whisper]"
pip install -e ".[speechbrain]"
```

---

## Vosk model setup (required for Vosk)

1. Download a Vosk model (e.g. `vosk-model-small-en-us-0.15`)
2. Extract it
3. Set environment variable:

```bash
export VOSK_MODEL_PATH="/absolute/path/to/vosk-model-small-en-us-0.15"
```

Windows PowerShell:

```powershell
$env:VOSK_MODEL_PATH = "C:\absolute\path\to\vosk-model-small-en-us-0.15"
```

---

## Running the interactive CLI

```bash
audio_transcription_demo
```

or

```bash
python -m audio_transcription_demo.main
```

The CLI will guide you through:

1. recognizer selection
2. audio source selection
3. use case selection

---

## Running as a FastAPI service (local)

Install API dependencies:

```bash
uv sync --extra api
```

Run:

```bash
uvicorn audio_transcription_demo.api:app --host 0.0.0.0 --port 8000
```

OpenAPI docs:

* [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker usage

The project includes a `Dockerfile` that runs the **FastAPI server**.

### What the container does (important)

* Runs **API mode only** (no microphone support)
* Uses **file upload** for audio input
* Requires a **Vosk model directory** to be mounted if using Vosk
* Exposes port **8000**

This design is intentional and container-safe.

---

### Build the Docker image

From the project root:

```bash
docker build -t audio_transcription_demo:latest .
```

---

### Run the Docker container (Vosk example)

You must:

* mount the Vosk model directory
* set `VOSK_MODEL_PATH` inside the container

```bash
docker run --rm -p 8000:8000 \
  -e VOSK_MODEL_PATH=/models/vosk \
  -v /absolute/path/to/vosk-model-small-en-us-0.15:/models/vosk \
  audio_transcription_demo:latest
```

The API will be available at:

* [http://localhost:8000](http://localhost:8000)

---

### Using Whisper or SpeechBrain in Docker

If you want Whisper or SpeechBrain inside the container:

1. Install extras in the Dockerfile:

   ```dockerfile
   RUN pip install --no-cache-dir ".[api,whisper,speechbrain]"
   ```
2. Rebuild the image:

   ```bash
   docker build -t audio_transcription_demo:latest .
   ```

Notes:

* These backends increase image size and startup time.
* CPU-only execution works; GPU support requires additional setup.

---

## API endpoints (Docker or local)

### Health check

```bash
curl http://localhost:8000/health
```

### Transcribe an audio file

```bash
curl -X POST "http://localhost:8000/transcribe?recognizer=vosk" \
  -F "file=@/path/to/audio.mp3"
```

### Generate meeting minutes

```bash
curl -X POST "http://localhost:8000/meeting-minutes?recognizer=vosk&max_sentences=6" \
  -F "file=@/path/to/meeting.mp3"
```

---

## Docker + multi-agent systems

This container is designed to be:

* **stateless** (aside from model cache)
* **HTTP-driven**
* easy to wire into:

  * agent frameworks
  * workflow engines
  * message queues
  * orchestrators (Docker Compose, Kubernetes)

Typical pattern:

* one container per recognizer backend
* agents send audio → receive transcript / summary
* downstream agents consume structured text

---

## Notes & best practices

* For live workshops: keep **Vosk + file source** as a fallback
* For accuracy demos: compare Vosk vs Whisper on the same audio
* For containers: avoid microphone input; use uploads or shared volumes
* For production: pre-warm recognizer instances (already done via caching)

---

## License

Specify your preferred license in `pyproject.toml` (MIT is a common default).