from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator


@dataclass(slots=True)
class TranscriptChunk:
    """
    Container for a piece of transcription output.

    Parameters
    ----------
    text : str
        Transcribed text.
    is_final : bool, default=False
        Whether the chunk represents finalized transcription.
    start : float | None, default=None
        Start timestamp in seconds.
    end : float | None, default=None
        End timestamp in seconds.
    meta : dict, default={}
        Additional metadata produced by the recognizer.
    """

    text: str
    is_final: bool = False
    start: float | None = None
    end: float | None = None
    meta: dict = field(default_factory=dict)


class SpeechRecognizer(ABC):
    """
    Abstract interface for speech recognition backends.

    Implementations typically support:

    - Full file transcription
    - Incremental streaming transcription
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Name of the recognizer backend.

        Returns
        -------
        str
            Backend identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe_file(self, path: str | Path) -> str:
        """
        Transcribe a full audio file.

        Parameters
        ----------
        path : str | Path
            Path to the audio file.

        Returns
        -------
        str
            Final transcription text.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe_stream(
        self,
        stream: BinaryIO,
        *,
        chunk_size: int = 4000,
    ) -> Iterator[TranscriptChunk]:
        """
        Perform incremental transcription on an audio stream.

        Parameters
        ----------
        stream : BinaryIO
            Stream of raw audio bytes.
        chunk_size : int, default=4000
            Number of bytes read per iteration.

        Yields
        ------
        TranscriptChunk
            Partial or finalized transcription results.

        Notes
        -----
        Expected audio format:

        - mono
        - 16-bit PCM
        - correct sample rate for the recognizer
        """
        raise NotImplementedError