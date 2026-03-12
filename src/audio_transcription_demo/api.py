from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.audio_transcription_demo.recognizers.recognizer_base import TranscriptChunk
from src.audio_transcription_demo.recognizers.recognizer_factory import RecognizerFactory
from src.audio_transcription_demo.sources.file_audio_source import FileAudioSource
from src.audio_transcription_demo.sources.source_factory import SourceFactory


class HealthResponse(BaseModel):
    """
    Health endpoint response model.

    Parameters
    ----------
    status : str
        Service status string.
    service : str
        Service name.
    """

    status: str = Field(...)
    service: str = Field(...)


class RecognizerInfo(BaseModel):
    """
    Recognizer metadata returned by the API.

    Parameters
    ----------
    name : str
        Recognizer backend name.
    models : list[str]
        Supported model identifiers.
    supports_file : bool
        Whether file transcription is supported.
    supports_stream : bool
        Whether streaming transcription is supported.
    requires_model_path : bool
        Whether this recognizer requires a local model path.
    """

    name: str = Field(...)
    models: list[str] = Field(default_factory=list)
    supports_file: bool = Field(...)
    supports_stream: bool = Field(...)
    requires_model_path: bool = Field(...)


class FileTranscriptionResponse(BaseModel):
    """
    Single-file transcription response.

    Parameters
    ----------
    filename : str
        Original uploaded filename.
    recognizer : str
        Recognizer backend used.
    text : str
        Final transcription text.
    """

    filename: str = Field(...)
    recognizer: str = Field(...)
    text: str = Field(...)


class MultiFileTranscriptionItem(BaseModel):
    """
    Per-file response item for batch transcription.

    Parameters
    ----------
    filename : str
        Original uploaded filename.
    recognizer : str
        Recognizer backend used.
    text : str | None
        Final transcription text if successful.
    error : str | None
        Error message if transcription failed.
    """

    filename: str = Field(...)
    recognizer: str = Field(...)
    text: str | None = Field(default=None)
    error: str | None = Field(default=None)


class MultiFileTranscriptionResponse(BaseModel):
    """
    Multi-file transcription response.

    Parameters
    ----------
    items : list[MultiFileTranscriptionItem]
        Results for each uploaded file.
    """

    items: list[MultiFileTranscriptionItem] = Field(default_factory=list)


class _PushAudioInputStream:
    """
    File-like PCM stream backed by externally pushed audio chunks.

    This is used by the WebSocket endpoint so browser or desktop clients
    can stream microphone audio into the API server.

    Notes
    -----
    The recognizers in this repo expect a binary stream with a ``read()``
    method for incremental transcription.
    """

    def __init__(self) -> None:
        """
        Initialize the push stream.
        """
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._closed = False

    def write(self, data: bytes) -> int:
        """
        Push a new audio chunk into the stream.

        Parameters
        ----------
        data : bytes
            Raw PCM audio bytes.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        RuntimeError
            If the stream is already closed.
        TypeError
            If ``data`` is not bytes-like.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed stream.")

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Audio chunk must be bytes-like.")

        payload = bytes(data)
        if payload:
            self._queue.put(payload)
        return len(payload)

    def read(self, size: int = -1) -> bytes:
        """
        Read the next available audio chunk.

        Parameters
        ----------
        size : int, default=-1
            Requested size in bytes. Ignored because chunks are returned
            in producer-pushed units.

        Returns
        -------
        bytes
            Next audio chunk, or ``b""`` when the stream is closed and
            no more audio is available.
        """
        item = self._queue.get()
        if item is None:
            self._closed = True
            return b""
        return item

    def close(self) -> None:
        """
        Close the stream.

        After closing, consumers will eventually receive ``b""``.
        """
        if self._closed:
            return
        self._queue.put(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application lifecycle.

    Parameters
    ----------
    app : FastAPI
        Application instance.

    Yields
    ------
    None
        Control back to FastAPI while the app is running.
    """
    app.state.service_name = "audio-transcription-demo-api"
    yield


app = FastAPI(
    title="Audio Transcription Demo API",
    version="0.1.0",
    description=(
        "API for file transcription and live streaming transcription "
        "using the recognizers from this repository."
    ),
    lifespan=lifespan,
)


def _create_recognizer(
    recognizer_name: str,
    *,
    sample_rate: int,
    whisper_model: str,
    whisper_language: str | None,
    vosk_model_path: str | None,
):
    """
    Create a recognizer instance from request parameters.

    Parameters
    ----------
    recognizer_name : str
        Recognizer backend name.
    sample_rate : int
        Audio sample rate.
    whisper_model : str
        Whisper model name.
    whisper_language : str | None
        Optional Whisper language override.
    vosk_model_path : str | None
        Optional Vosk model path.

    Returns
    -------
    SpeechRecognizer
        Initialized recognizer instance.

    Raises
    ------
    HTTPException
        If recognizer creation fails.
    """
    try:
        return RecognizerFactory.create(
            recognizer_name,
            sample_rate=sample_rate,
            whisper_model=whisper_model,
            whisper_language=whisper_language,
            vosk_model_path=vosk_model_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _save_upload_to_tempfile(upload: UploadFile) -> Path:
    """
    Persist an uploaded file to a temporary path.

    Parameters
    ----------
    upload : UploadFile
        Uploaded file received by FastAPI.

    Returns
    -------
    pathlib.Path
        Path to the saved temporary file.
    """
    suffix = Path(upload.filename or "audio.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = upload.file.read()
        tmp.write(content)
        return Path(tmp.name)


def _chunk_to_dict(chunk: TranscriptChunk) -> dict[str, Any]:
    """
    Convert a transcript chunk to a JSON-serializable dictionary.

    Parameters
    ----------
    chunk : TranscriptChunk
        Transcript chunk emitted by a recognizer.

    Returns
    -------
    dict[str, Any]
        Serialized chunk.
    """
    return {
        "text": chunk.text,
        "is_final": chunk.is_final,
        "start": chunk.start,
        "end": chunk.end,
        "meta": chunk.meta,
    }


def _iter_sse_events(chunks: Iterator[TranscriptChunk]) -> Iterator[str]:
    """
    Convert transcript chunks into Server-Sent Events.

    Parameters
    ----------
    chunks : Iterator[TranscriptChunk]
        Chunk iterator returned by a streaming recognizer.

    Yields
    ------
    str
        SSE-formatted event payloads.
    """
    try:
        for chunk in chunks:
            payload = json.dumps(_chunk_to_dict(chunk), ensure_ascii=False)
            yield f"data: {payload}\n\n"
    except Exception as exc:
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """
    Return API health information.

    Returns
    -------
    HealthResponse
        Service health response.
    """
    return HealthResponse(
        status="ok",
        service=app.state.service_name,
    )


@app.get("/sources", tags=["metadata"])
def list_sources() -> dict[str, Any]:
    """
    List supported audio sources.

    Returns
    -------
    dict[str, Any]
        Source metadata.
    """
    return SourceFactory.list_sources()


@app.get("/recognizers", response_model=list[RecognizerInfo], tags=["metadata"])
def list_recognizers() -> list[RecognizerInfo]:
    """
    List recognizer backends and their capabilities.

    Returns
    -------
    list[RecognizerInfo]
        Recognizer metadata.
    """
    model_map = RecognizerFactory.list_models()
    items: list[RecognizerInfo] = []

    for name, models in model_map.items():
        items.append(
            RecognizerInfo(
                name=name,
                models=models,
                supports_file=True,
                supports_stream=True,
                requires_model_path=name == "vosk",
            )
        )

    return items


@app.post(
    "/transcriptions/file",
    response_model=FileTranscriptionResponse,
    tags=["transcriptions"],
)
def transcribe_file(
    file: UploadFile = File(...),
    recognizer: str = Form("whisper"),
    sample_rate: int = Form(16_000),
    whisper_model: str = Form("base"),
    whisper_language: str | None = Form(None),
    vosk_model_path: str | None = Form(None),
) -> FileTranscriptionResponse:
    """
    Transcribe one uploaded audio file.

    Parameters
    ----------
    file : UploadFile
        Uploaded audio file.
    recognizer : str, default="whisper"
        Recognizer backend name.
    sample_rate : int, default=16000
        Audio sample rate.
    whisper_model : str, default="base"
        Whisper model identifier.
    whisper_language : str | None, default=None
        Optional Whisper language override.
    vosk_model_path : str | None, default=None
        Optional Vosk model path.

    Returns
    -------
    FileTranscriptionResponse
        Final transcription result.
    """
    recognizer_instance = _create_recognizer(
        recognizer,
        sample_rate=sample_rate,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        vosk_model_path=vosk_model_path,
    )
    source = FileAudioSource(sample_rate=sample_rate)

    temp_path = _save_upload_to_tempfile(file)
    try:
        text = source.transcribe(
            recognizer_instance,
            source=temp_path,
            stream=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return FileTranscriptionResponse(
        filename=file.filename or temp_path.name,
        recognizer=recognizer_instance.label,
        text=str(text),
    )


@app.post(
    "/transcriptions/files",
    response_model=MultiFileTranscriptionResponse,
    tags=["transcriptions"],
)
def transcribe_files(
    files: list[UploadFile] = File(...),
    recognizer: str = Form("whisper"),
    sample_rate: int = Form(16_000),
    whisper_model: str = Form("base"),
    whisper_language: str | None = Form(None),
    vosk_model_path: str | None = Form(None),
) -> MultiFileTranscriptionResponse:
    """
    Transcribe multiple uploaded audio files.

    Parameters
    ----------
    files : list[UploadFile]
        Uploaded audio files.
    recognizer : str, default="whisper"
        Recognizer backend name.
    sample_rate : int, default=16000
        Audio sample rate.
    whisper_model : str, default="base"
        Whisper model identifier.
    whisper_language : str | None, default=None
        Optional Whisper language override.
    vosk_model_path : str | None, default=None
        Optional Vosk model path.

    Returns
    -------
    MultiFileTranscriptionResponse
        Batch transcription results.
    """
    recognizer_instance = _create_recognizer(
        recognizer,
        sample_rate=sample_rate,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        vosk_model_path=vosk_model_path,
    )
    source = FileAudioSource(sample_rate=sample_rate)

    items: list[MultiFileTranscriptionItem] = []

    for upload in files:
        temp_path = _save_upload_to_tempfile(upload)
        try:
            text = source.transcribe(
                recognizer_instance,
                source=temp_path,
                stream=False,
            )
            items.append(
                MultiFileTranscriptionItem(
                    filename=upload.filename or temp_path.name,
                    recognizer=recognizer_instance.label,
                    text=str(text),
                )
            )
        except Exception as exc:
            items.append(
                MultiFileTranscriptionItem(
                    filename=upload.filename or temp_path.name,
                    recognizer=recognizer_instance.label,
                    error=str(exc),
                )
            )
        finally:
            temp_path.unlink(missing_ok=True)

    return MultiFileTranscriptionResponse(items=items)


@app.get("/transcriptions/file/stream", tags=["transcriptions"])
def transcribe_file_stream(
    path: str = Query(..., description="Path to a local audio file on the server"),
    recognizer: str = Query("whisper"),
    sample_rate: int = Query(16_000),
    chunk_size: int = Query(4000),
    whisper_model: str = Query("base"),
    whisper_language: str | None = Query(None),
    vosk_model_path: str | None = Query(None),
) -> StreamingResponse:
    """
    Stream transcription events for a server-side audio file.

    Parameters
    ----------
    path : str
        Path to an audio file on the server.
    recognizer : str, default="whisper"
        Recognizer backend name.
    sample_rate : int, default=16000
        Audio sample rate.
    chunk_size : int, default=4000
        Stream read chunk size.
    whisper_model : str, default="base"
        Whisper model identifier.
    whisper_language : str | None, default=None
        Optional Whisper language override.
    vosk_model_path : str | None, default=None
        Optional Vosk model path.

    Returns
    -------
    StreamingResponse
        SSE stream of transcript chunks.
    """
    recognizer_instance = _create_recognizer(
        recognizer,
        sample_rate=sample_rate,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        vosk_model_path=vosk_model_path,
    )
    source = FileAudioSource(sample_rate=sample_rate)

    try:
        chunks = source.transcribe(
            recognizer_instance,
            source=Path(path),
            stream=True,
            chunk_size=chunk_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        _iter_sse_events(chunks),
        media_type="text/event-stream",
    )


@app.websocket("/transcriptions/stream")
async def stream_transcription(websocket: WebSocket) -> None:
    """
    Receive raw PCM audio over WebSocket and stream transcription results back.

    Protocol
    --------
    The client should:

    1. Connect to this endpoint
    2. Send one JSON text message with configuration
    3. Send binary PCM audio chunks
    4. Send a final text message ``{"event": "end"}`` when done

    Initial configuration message
    ------------------------------
    {
        "recognizer": "whisper",
        "sample_rate": 16000,
        "chunk_size": 4000,
        "whisper_model": "base",
        "whisper_language": null,
        "vosk_model_path": "/models/vosk-model-en-us"
    }

    Outgoing messages
    -----------------
    The server sends JSON text frames shaped like:

    {
        "event": "transcript",
        "data": {
            "text": "...",
            "is_final": false,
            "start": null,
            "end": null,
            "meta": {}
        }
    }

    Parameters
    ----------
    websocket : WebSocket
        FastAPI WebSocket connection.
    """
    await websocket.accept()

    try:
        config = await websocket.receive_json()
    except Exception:
        await websocket.send_json(
            {"event": "error", "detail": "Expected initial JSON configuration message."}
        )
        await websocket.close(code=1003)
        return

    recognizer_name = str(config.get("recognizer", "whisper"))
    sample_rate = int(config.get("sample_rate", 16_000))
    chunk_size = int(config.get("chunk_size", 4000))
    whisper_model = str(config.get("whisper_model", "base"))
    whisper_language = config.get("whisper_language")
    vosk_model_path = config.get("vosk_model_path")

    try:
        recognizer_instance = _create_recognizer(
            recognizer_name,
            sample_rate=sample_rate,
            whisper_model=whisper_model,
            whisper_language=whisper_language,
            vosk_model_path=vosk_model_path,
        )
    except HTTPException as exc:
        await websocket.send_json({"event": "error", "detail": exc.detail})
        await websocket.close(code=1003)
        return

    push_stream = _PushAudioInputStream()
    output_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _worker() -> None:
        """
        Run the blocking recognizer stream in a background thread.

        This bridges the synchronous recognizer iterator into the async
        WebSocket handler.
        """
        try:
            for chunk in recognizer_instance.transcribe_stream(
                push_stream,
                chunk_size=chunk_size,
            ):
                asyncio.run_coroutine_threadsafe(
                    output_queue.put(
                        {
                            "event": "transcript",
                            "data": _chunk_to_dict(chunk),
                        }
                    ),
                    loop,
                )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                output_queue.put(
                    {
                        "event": "error",
                        "detail": str(exc),
                    }
                ),
                loop,
            )
        finally:
            asyncio.run_coroutine_threadsafe(
                output_queue.put({"event": "done"}),
                loop,
            )

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    async def _sender() -> None:
        """
        Forward recognizer results back to the client.

        Sends
        -----
        JSON messages
            Transcript, error, and completion events.
        """
        while True:
            message = await output_queue.get()
            await websocket.send_json(message)
            if message.get("event") in {"error", "done"}:
                break

    sender_task = asyncio.create_task(_sender())

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                push_stream.write(message["bytes"])
                continue

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"event": "error", "detail": "Invalid JSON text message."}
                    )
                    continue

                if payload.get("event") == "end":
                    push_stream.close()
                    break
                continue

            if message.get("type") == "websocket.disconnect":
                push_stream.close()
                break

    except Exception as exc:
        push_stream.close()
        await websocket.send_json({"event": "error", "detail": str(exc)})
    finally:
        push_stream.close()
        with contextlib.suppress(Exception):
            await sender_task
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    """
    Return a small root document.

    Returns
    -------
    JSONResponse
        Root endpoint response.
    """
    return JSONResponse(
        {
            "name": "Audio Transcription Demo API",
            "docs": "/docs",
            "health": "/health",
        }
    )