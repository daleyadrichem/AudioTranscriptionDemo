from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .recognizers.base import SpeechRecognizer
from .recognizers.factory import create_recognizer
from .utils import AudioProcessingError, BackendNotAvailableError, format_meeting_minutes


RecognizerName = Literal["vosk", "whisper", "speechbrain"]


class TranscriptionResponse(BaseModel):
    """
    Response model for transcription endpoints.

    Attributes
    ----------
    recognizer:
        Identifier of the recognizer backend used.
    text:
        Transcript text.
    """

    recognizer: RecognizerName
    text: str


class MeetingMinutesResponse(BaseModel):
    """
    Response model for meeting-minutes endpoint.

    Attributes
    ----------
    recognizer:
        Identifier of the recognizer backend used.
    transcript:
        Transcript text.
    minutes_markdown:
        Markdown-formatted meeting minutes.
    """

    recognizer: RecognizerName
    transcript: str
    minutes_markdown: str


class HealthResponse(BaseModel):
    """
    Basic health check response.

    Attributes
    ----------
    status:
        Health status string.
    """

    status: str = Field(default="ok")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    fastapi.FastAPI
        The configured FastAPI application instance.
    """
    app = FastAPI(
        title="speech-demo API",
        description="HTTP API for modular speech-to-text demo (recognizers + use cases).",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """
        Health check endpoint.

        Returns
        -------
        HealthResponse
            Simple status response.
        """
        return HealthResponse()

    @app.get("/recognizers")
    def list_recognizers() -> dict[str, list[str]]:
        """
        List supported recognizer backend identifiers.

        Returns
        -------
        dict[str, list[str]]
            A dictionary containing the list of recognizer names.
        """
        return {"recognizers": ["vosk", "whisper", "speechbrain"]}

    @app.post("/transcribe", response_model=TranscriptionResponse)
    async def transcribe(
        file: UploadFile = File(...),
        recognizer: RecognizerName = Query("vosk"),
    ) -> TranscriptionResponse:
        """
        Transcribe an uploaded audio file (mp3/wav/m4a/...).

        Parameters
        ----------
        file:
            Uploaded audio file. Any format supported by ffmpeg is acceptable.
        recognizer:
            Recognizer backend name ("vosk", "whisper", "speechbrain").

        Returns
        -------
        TranscriptionResponse
            Transcript text and recognizer used.

        Raises
        ------
        HTTPException
            - 400 if the upload is invalid
            - 500 for transcription errors or backend initialization failures
        """
        tmp_path = await _save_upload_to_temp(file)
        try:
            stt = _get_recognizer_cached(recognizer)
            text = stt.transcribe_any(tmp_path)
            return TranscriptionResponse(recognizer=recognizer, text=text)
        except BackendNotAvailableError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except (FileNotFoundError, AudioProcessingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
        finally:
            _safe_unlink(tmp_path)

    @app.post("/meeting-minutes", response_model=MeetingMinutesResponse)
    async def meeting_minutes(
        file: UploadFile = File(...),
        recognizer: RecognizerName = Query("vosk"),
        max_sentences: int = Query(6, ge=1, le=20),
    ) -> MeetingMinutesResponse:
        """
        Create meeting minutes from an uploaded meeting recording.

        Steps:
        1) Transcribe audio using the chosen recognizer.
        2) Produce Markdown minutes using a local heuristic summarizer.

        Parameters
        ----------
        file:
            Uploaded audio file.
        recognizer:
            Recognizer backend name ("vosk", "whisper", "speechbrain").
        max_sentences:
            Number of key sentences to include in the minutes.

        Returns
        -------
        MeetingMinutesResponse
            Transcript and meeting minutes in Markdown.

        Raises
        ------
        HTTPException
            - 400 for invalid input or conversion errors
            - 500 for transcription failures or backend initialization issues
        """
        tmp_path = await _save_upload_to_temp(file)
        try:
            stt = _get_recognizer_cached(recognizer)
            transcript = stt.transcribe_any(tmp_path)
            minutes_md = format_meeting_minutes(transcript, max_sentences=max_sentences)
            return MeetingMinutesResponse(
                recognizer=recognizer,
                transcript=transcript,
                minutes_markdown=minutes_md,
            )
        except BackendNotAvailableError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except (FileNotFoundError, AudioProcessingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Meeting minutes failed: {exc}") from exc
        finally:
            _safe_unlink(tmp_path)

    return app


app = create_app()


@lru_cache(maxsize=8)
def _get_recognizer_cached(name: RecognizerName) -> SpeechRecognizer:
    """
    Create and cache recognizer instances.

    Caching matters for Whisper/SpeechBrain because loading models can be slow.

    Parameters
    ----------
    name:
        Recognizer backend name.

    Returns
    -------
    SpeechRecognizer
        Initialized recognizer instance.

    Raises
    ------
    BackendNotAvailableError
        If required dependency for that backend is not installed.
    RuntimeError
        If backend initialization fails.
    """
    return create_recognizer(name)


async def _save_upload_to_temp(file: UploadFile) -> Path:
    """
    Save an uploaded file to a temporary path on disk.

    Parameters
    ----------
    file:
        FastAPI uploaded file.

    Returns
    -------
    pathlib.Path
        Path to the saved file.

    Raises
    ------
    HTTPException
        If the upload has no filename or cannot be written.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    suffix = Path(file.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        tmp.write(content)

    return tmp_path


def _safe_unlink(path: Path) -> None:
    """
    Best-effort deletion of a file.

    Parameters
    ----------
    path:
        File path to delete.
    """
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
