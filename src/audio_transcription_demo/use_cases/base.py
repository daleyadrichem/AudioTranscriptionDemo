from __future__ import annotations

from abc import ABC, abstractmethod

from audio_transcription_demo.recognizers.base import SpeechRecognizer
from audio_transcription_demo.sources.base import AudioSource


class UseCase(ABC):
    """
    Abstract interface for demo use cases.

    A use case is an executable demo scenario (e.g. live transcription,
    transcribe a file, meeting minutes) that orchestrates:
    - selecting/recording audio (AudioSource)
    - transcribing it (SpeechRecognizer)
    - outputting results
    """

    @property
    @abstractmethod
    def key(self) -> str:
        """
        Menu key for selecting the use case.

        Returns
        -------
        str
            Key (typically a single character).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Human-readable label for menu display.

        Returns
        -------
        str
            Label.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, *, source: AudioSource, recognizer: SpeechRecognizer) -> None:
        """
        Execute the use case.

        Parameters
        ----------
        source:
            The audio source to acquire audio from.
        recognizer:
            The recognizer backend used to transcribe audio.

        Raises
        ------
        RuntimeError
            If acquisition, transcription, or output fails.
        """
        raise NotImplementedError
