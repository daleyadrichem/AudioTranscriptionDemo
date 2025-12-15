from __future__ import annotations

from typing import Literal, Optional

from pathlib import Path

from audio_transcription_demo.sources.base import AudioSource
from audio_transcription_demo.sources.file import FileAudioSource
from audio_transcription_demo.sources.microphone import MicrophoneAudioSource

SourceName = Literal["file", "microphone"]


def create_source(
    name: str,
    *,
    sample_rate: int,
    default_file_path: Optional[Path] = None,
) -> AudioSource:
    """
    Create an audio source by name.

    Parameters
    ----------
    name:
        Source name. Supported values are: "file", "microphone".
    sample_rate:
        Sample rate used for conversion/recording in Hz.
    default_file_path:
        Optional default path for file-based source.

    Returns
    -------
    AudioSource
        A concrete source implementation.

    Raises
    ------
    ValueError
        If `name` is not a supported source identifier.
    """
    key = name.strip().lower()
    if key == "file":
        return FileAudioSource(sample_rate=sample_rate, default_path=default_file_path)
    if key == "microphone":
        return MicrophoneAudioSource(sample_rate=sample_rate)
    raise ValueError("Unknown source. Use one of: file, microphone.")
