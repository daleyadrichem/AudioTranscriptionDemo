from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO, Iterator

from audio_transcription_demo.utils import TempPath
from src.audio_transcription_demo.recognizers.recognizer_base import (
    SpeechRecognizer,
    TranscriptChunk,
)


class AudioSource(ABC):
    """
    Abstract interface for audio sources.

    An audio source is responsible for obtaining audio input from the user
    and passing it to a recognizer either as a file or as a stream.

    Notes
    -----
    Implementations may support one or both of these workflows:

    - file-based transcription
    - streaming transcription
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Return a human-readable label for the source.

        Returns
        -------
        str
            Source label.
        """
        raise NotImplementedError

    @abstractmethod
    def get_audio(self, *args, **kwargs) -> TempPath:
        """
        Obtain audio input and return it as a temporary file.

        Returns
        -------
        TempPath
            Temporary path to audio data.

        Raises
        ------
        RuntimeError
            Raised when audio acquisition fails.
        NotImplementedError
            Raised when the source does not support file-based access.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe(
        self,
        recognizer: SpeechRecognizer,
        *,
        stream: bool = False,
        chunk_size: int = 4000,
        audio_stream: BinaryIO | None = None,
        **kwargs,
    ) -> str | Iterator[TranscriptChunk]:
        """
        Transcribe audio using the provided recognizer.

        Parameters
        ----------
        recognizer : SpeechRecognizer
            Recognizer used for transcription.
        stream : bool, default=False
            Whether to use streaming transcription.
        chunk_size : int, default=4000
            Chunk size to use when streaming.
        audio_stream : BinaryIO | None, default=None
            Optional externally provided binary audio stream. This is useful
            for API or WebSocket scenarios where audio is captured outside the
            current process and pushed into the source.
        **kwargs
            Additional source-specific arguments.

        Returns
        -------
        str | Iterator[TranscriptChunk]
            Final transcription text for non-streaming mode, or an iterator
            yielding transcript chunks for streaming mode.
        """
        raise NotImplementedError