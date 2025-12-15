from __future__ import annotations

from abc import ABC, abstractmethod

from audio_transcription_demo.utils import TempPath


class AudioSource(ABC):
    """
    Abstract interface for audio sources.

    An audio source is responsible for obtaining audio input from the user
    (e.g., selecting a file or recording from a microphone) and returning it
    as a `TempPath`.

    Notes
    -----
    Many speech recognizers are simplest to use with WAV (mono, 16 kHz).
    Sources may choose to normalize audio into that form.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Human-readable label for UI menus.

        Returns
        -------
        str
            Source label.
        """
        raise NotImplementedError

    @abstractmethod
    def get_audio(self) -> TempPath:
        """
        Obtain audio input and return a path to it.

        Returns
        -------
        TempPath
            Path to audio for transcription. May require cleanup.

        Raises
        ------
        RuntimeError
            If the source fails to acquire audio (device missing, conversion fails).
        """
        raise NotImplementedError
