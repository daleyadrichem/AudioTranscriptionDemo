from __future__ import annotations

from typing import Literal

from audio_transcription_demo.utils import BackendNotAvailableError
from audio_transcription_demo.recognizers.base import SpeechRecognizer
from audio_transcription_demo.recognizers.speechbrain import SpeechBrainSpeechRecognizer
from audio_transcription_demo.recognizers.vosk import VoskSpeechRecognizer
from audio_transcription_demo.recognizers.whisper import WhisperSpeechRecognizer

RecognizerName = Literal["vosk", "whisper", "speechbrain"]


def create_recognizer(name: str) -> SpeechRecognizer:
    """
    Create a speech recognizer backend by name.

    Parameters
    ----------
    name:
        Backend name. Supported values are: "vosk", "whisper", "speechbrain".

    Returns
    -------
    SpeechRecognizer
        A concrete recognizer implementation.

    Raises
    ------
    ValueError
        If `name` is not a supported backend identifier.
    BackendNotAvailableError
        If the backend's optional dependency is missing.
    """
    key = name.strip().lower()

    try:
        if key == "vosk":
            return VoskSpeechRecognizer()
        if key == "whisper":
            return WhisperSpeechRecognizer()
        if key == "speechbrain":
            return SpeechBrainSpeechRecognizer()
    except ImportError as exc:
        raise BackendNotAvailableError(
            f"Backend {key!r} requested but required package is not installed."
        ) from exc

    raise ValueError("Unknown recognizer. Use one of: vosk, whisper, speechbrain.")
