from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .recognizers.base import SpeechRecognizer
from .recognizers.factory import create_recognizer
from .utils import AudioProcessingError, BackendNotAvailableError, format_meeting_minutes


RecognizerName = Literal["vosk", "whisper", "speechbrain"]


class HealthResponse(BaseModel):
    """
    Response model for health checks.

    Attributes
    ----------
    status:
        Health status string.
    """

    status: str = Field(default="ok")


class TranscriptionResponse(BaseModel):
    """
    Response model for transcription use case.

    Attributes
    ----------
    recognizer:
        Recognizer backend identifier used for transcription.
    transcript:
        Transcript text.
    """

    recognizer: RecognizerName
    transcript: str


class LiveTranscribeChunkResponse(BaseModel):
    """
    Response model for the "live transcribe" API use case.

    Notes
    -----
    True streaming is typically done via WebSockets. For simplicity, this API
    supports a "chunked" live demo: clients repeatedly upload short audio snippets
    and get back incremental transcripts.

    Attributes
    ----------
    recognizer:
        Recognizer backend identifier used.
    transcript:
        Transcript text for this single uploaded chunk.
    """

    recognizer: RecognizerName
    transcript: str


class MeetingMinutesResponse(BaseModel):
    """
    Response model for meeting-minutes use case.

    Attributes
    ----------
    recognizer:
        Recognizer backend identifier used.
    transcript:
        Transcript text.
    minutes_markdown:
        Markdown-formatted meeting minutes generated from transcript.
    """

    recognizer: RecognizerName
    transcript: str
    minutes_markdown: str


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
        description=(
            "Upload an audio file in Swagger UI and run speech-to-text use cases.\n\n"
            "Open **/docs** for the interactive Swagger page."
        ),
        version="0.1.0",
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """
        Redirect root URL to Swagger UI.

        Returns
        -------
        fastapi.responses.RedirectResponse
            Redirect response to `/docs`.
        """
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """
        Health check endpoint.

        Returns
        -------
        HealthResponse
            Basic health response.
        """
        return HealthResponse()

    @app.get("/recognizers", tags=["system"])
    def list_recognizers() -> dict[str, list[str]]:
        """
        List supported recognizer backend identifiers.

        Returns
        -------
        dict[str, list[str]]
            Dictionary containing available recognizer names.
        """
        return {"recognizers": ["vosk", "whisper", "speechbrain"]}

    # --- Use-case endpoints (Swagger-friendly) --------------------------------

    @app.post(
        "/use-cases/transcribe",
        response_model=TranscriptionResponse,
        tags=["use-cases"],
        summary="Use case: Transcribe a single audio file",
        description=(
            "Upload an audio file (mp3/wav/m4a/etc.) and get a transcript.\n\n"
            "In Swagger UI, click **Try it out** and browse for a file."
        ),
    )
    async def use_case_transcribe(
        file: UploadFile = File(..., description="Audio file to transcribe."),
        recognizer: RecognizerName = Query(
            "whisper",
            description="Recognizer backend to use.",
        ),
    ) -> TranscriptionResponse:
        """
        Transcribe one uploaded audio file.

        Parameters
        ----------
        file:
            Uploaded audio file.
        recognizer:
            Recognizer backend identifier.

        Returns
        -------
        TranscriptionResponse
            Transcript and backend used.

        Raises
        ------
        HTTPException
            If upload is invalid or transcription fails.
        """
        tmp_path = await _save_upload_to_temp(file)
        try:
            stt = _get_recognizer_cached(recognizer)
            transcript = stt.transcribe_any(tmp_path)
            return TranscriptionResponse(recognizer=recognizer, transcript=transcript)
        except BackendNotAvailableError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except (FileNotFoundError, AudioProcessingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
        finally:
            _safe_unlink(tmp_path)

    @app.post(
        "/use-cases/live-transcribe",
        response_model=LiveTranscribeChunkResponse,
        tags=["use-cases"],
        summary="Use case: Live transcription (chunked uploads)",
        description=(
            "This is an API-friendly variant of the CLI live transcription:\n"
            "upload a short audio chunk and get back a transcript for that chunk.\n\n"
            "For real streaming, you would typically add WebSockets."
        ),
    )
    async def use_case_live_transcribe_chunk(
        file: UploadFile = File(..., description="Short audio snippet to transcribe."),
        recognizer: RecognizerName = Query(
            "whisper",
            description="Recognizer backend to use.",
        ),
    ) -> LiveTranscribeChunkResponse:
        """
        Transcribe a short snippet (one chunk) and return its transcript.

        Parameters
        ----------
        file:
            Uploaded audio snippet.
        recognizer:
            Recognizer backend identifier.

        Returns
        -------
        LiveTranscribeChunkResponse
            Transcript for the chunk and backend used.

        Raises
        ------
        HTTPException
            If upload is invalid or transcription fails.
        """
        tmp_path = await _save_upload_to_temp(file)
        try:
            stt = _get_recognizer_cached(recognizer)
            transcript = stt.transcribe_any(tmp_path)
            return LiveTranscribeChunkResponse(recognizer=recognizer, transcript=transcript)
        except BackendNotAvailableError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except (FileNotFoundError, AudioProcessingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Live transcription failed: {exc}") from exc
        finally:
            _safe_unlink(tmp_path)

    @app.post(
        "/use-cases/meeting-minutes",
        response_model=MeetingMinutesResponse,
        tags=["use-cases"],
        summary="Use case: Meeting minutes (transcribe + local summary)",
        description=(
            "Upload a meeting recording and get:\n"
            "1) transcript\n"
            "2) simple Markdown meeting minutes (heuristic, local-only)."
        ),
    )
    async def use_case_meeting_minutes(
        file: UploadFile = File(..., description="Meeting audio file."),
        recognizer: RecognizerName = Query(
            "whisper",
            description="Recognizer backend to use.",
        ),
        max_sentences: int = Query(
            6,
            ge=1,
            le=20,
            description="Number of key sentences to include in minutes.",
        ),
    ) -> MeetingMinutesResponse:
        """
        Create meeting minutes from an uploaded meeting recording.

        Parameters
        ----------
        file:
            Uploaded meeting audio file.
        recognizer:
            Recognizer backend identifier.
        max_sentences:
            Number of key sentences in the minutes.

        Returns
        -------
        MeetingMinutesResponse
            Transcript and Markdown minutes.

        Raises
        ------
        HTTPException
            If upload is invalid or processing fails.
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

    Caching avoids repeated heavy model loads (important for Whisper/SpeechBrain).

    Parameters
    ----------
    name:
        Recognizer backend identifier.

    Returns
    -------
    SpeechRecognizer
        Initialized recognizer instance.

    Raises
    ------
    BackendNotAvailableError
        If dependency for the backend is missing.
    RuntimeError
        If initialization fails.
    """
    return create_recognizer(name)


async def _save_upload_to_temp(file: UploadFile) -> Path:
    """
    Save an uploaded file to a temporary path on disk.

    Parameters
    ----------
    file:
        Uploaded file.

    Returns
    -------
    pathlib.Path
        Path to the saved temporary file.

    Raises
    ------
    HTTPException
        If the upload is empty or cannot be written.
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
    Best-effort deletion of a temporary file.

    Parameters
    ----------
    path:
        Path to delete.
    """
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
