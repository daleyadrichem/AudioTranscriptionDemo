from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal

from src.audio_transcription_demo.sources.file_audio_source import FileAudioSource
from src.audio_transcription_demo.sources.microphone_audio_source import (
    MicrophoneAudioSource,
)
from src.audio_transcription_demo.sources.source_base import AudioSource

SourceName = Literal["file", "microphone"]


class SourceFactory:
    """
    Factory for creating audio source backends.

    This factory provides:

    - Creation of audio source instances
    - Listing supported audio source types
    """

    @staticmethod
    def create(
        name: str,
        *,
        sample_rate: int,
        default_file_path: Path | None = None,
    ) -> AudioSource:
        """
        Create an audio source instance.

        Parameters
        ----------
        name : str
            Source name. Supported values are:

            - ``"file"``
            - ``"microphone"``

        sample_rate : int
            Sample rate used for conversion or capture.

        default_file_path : pathlib.Path | None, default=None
            Optional default file path used by the file source.

        Returns
        -------
        AudioSource
            Initialized audio source.

        Raises
        ------
        ValueError
            If an unknown source name is provided.
        """
        key = name.strip().lower()

        if key == "file":
            return FileAudioSource(
                sample_rate=sample_rate,
                default_path=default_file_path,
            )

        if key == "microphone":
            return MicrophoneAudioSource(sample_rate=sample_rate)

        raise ValueError(f"Unknown source: {name}")

    @staticmethod
    def list_sources() -> Dict[str, List[str]]:
        """
        List supported audio source types.

        Returns
        -------
        dict[str, list[str]]
            Mapping of source categories to supported implementations.

        Notes
        -----
        This method can be used by CLI or UI layers to populate source
        selection menus.
        """
        return {
            "sources": [
                "file",
                "microphone",
            ]
        }