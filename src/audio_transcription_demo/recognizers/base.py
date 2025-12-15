from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class SpeechRecognizer(ABC):
    """
    Abstract interface for a speech-to-text recognizer.

    Concrete implementations encapsulate a specific backend (e.g., Vosk,
    Whisper, SpeechBrain), while use cases operate against this stable API.

    Attributes
    ----------
    sample_rate:
        Preferred sample rate (Hz) for microphone recording.
    """

    sample_rate: int

    @abstractmethod
    def transcribe_wav(self, wav_path: Path) -> str:
        """
        Transcribe a WAV file.

        Parameters
        ----------
        wav_path:
            Path to a WAV file. Implementations may assume mono audio and a
            preferred sample rate; if required, callers should convert first.

        Returns
        -------
        str
            Transcript text (may be empty if nothing was recognized).

        Raises
        ------
        FileNotFoundError
            If `wav_path` does not exist.
        RuntimeError
            If the backend fails internally.
        """
        raise NotImplementedError

    def transcribe_any(self, audio_path: Path) -> str:
        """
        Transcribe an arbitrary audio file.

        Parameters
        ----------
        audio_path:
            Path to an audio file (mp3/wav/m4a/etc).

        Returns
        -------
        str
            Transcript text.

        Raises
        ------
        FileNotFoundError
            If `audio_path` does not exist.
        RuntimeError
            If transcription fails.
        """
        # Default: treat it like WAV unless overridden.
        return self.transcribe_wav(audio_path)
