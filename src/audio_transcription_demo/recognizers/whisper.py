from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from audio_transcription_demo.recognizers.base import SpeechRecognizer


@dataclass
class WhisperSpeechRecognizer(SpeechRecognizer):
    """
    Whisper speech recognizer (local, heavier).

    This implementation uses `openai-whisper` (imported as `whisper`).
    Whisper can transcribe many formats directly and manages resampling
    internally.

    Parameters
    ----------
    model_name:
        Whisper model name (e.g., "tiny", "base", "small", "medium", "large").
    device:
        Optional device spec passed to whisper (e.g., "cpu", "cuda").
    sample_rate:
        Preferred microphone sample rate (recording). Whisper can resample.

    Raises
    ------
    ImportError
        If `openai-whisper` is not installed.
    RuntimeError
        If model loading fails.
    """

    model_name: str = "base"
    device: Optional[str] = None
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        import whisper  # type: ignore[import-not-found]

        self._model = whisper.load_model(self.model_name, device=self.device)

    def transcribe_wav(self, wav_path: Path) -> str:
        """
        Transcribe audio with Whisper.

        Parameters
        ----------
        wav_path:
            Path to an audio file (WAV or other supported format).

        Returns
        -------
        str
            Transcript text.

        Raises
        ------
        FileNotFoundError
            If `wav_path` does not exist.
        RuntimeError
            If Whisper transcription fails.
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        result = self._model.transcribe(str(wav_path))
        return str(result.get("text", "")).strip()

    def transcribe_any(self, audio_path: Path) -> str:
        """
        Transcribe arbitrary audio formats supported by Whisper.

        Parameters
        ----------
        audio_path:
            Path to an audio file.

        Returns
        -------
        str
            Transcript text.

        Raises
        ------
        FileNotFoundError
            If `audio_path` does not exist.
        RuntimeError
            If Whisper transcription fails.
        """
        return self.transcribe_wav(audio_path)
