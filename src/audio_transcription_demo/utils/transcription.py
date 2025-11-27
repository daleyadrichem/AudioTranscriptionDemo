"""
Speech-to-text transcription utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import whisper


class WhisperTranscriber:
    """Speech-to-text transcriber using the `openai-whisper` library.

    This class lazily loads the Whisper model on first use to avoid
    long startup times if transcription is never called.

    Parameters
    ----------
    model_name : str, optional
        Name of the Whisper model to load. Common values are
        ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``, ``"large"``.
        Smaller models are faster but less accurate. The default is
        ``"small"``.
    device : {"cpu", "cuda"}, optional
        Device on which to run the model. If ``None``, the default
        device chosen by Whisper is used.

    Examples
    --------
    >>> transcriber = WhisperTranscriber(model_name="tiny")
    >>> text = transcriber.transcribe(Path("recordings/demo.wav"))
    """

    def __init__(self, model_name: str = "small", device: Optional[str] = None) -> None:
        self.model_name: str = model_name
        self.device: Optional[str] = device
        self._model = None

    def _ensure_model_loaded(self) -> None:
        """Load the Whisper model if it is not already loaded."""
        if self._model is None:
            print(f"[WhisperTranscriber] Loading model '{self.model_name}'...")
            self._model = whisper.load_model(self.model_name, device=self.device)
            print("[WhisperTranscriber] Model loaded.")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe speech from an audio file.

        Parameters
        ----------
        audio_path : Path
            Path to the audio file to transcribe.

        Returns
        -------
        str
            Transcribed text.

        Raises
        ------
        FileNotFoundError
            If the audio file does not exist.
        RuntimeError
            If transcription fails for any reason.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_model_loaded()

        try:
            print(f"[WhisperTranscriber] Transcribing: {audio_path}")
            result = self._model.transcribe(str(audio_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to transcribe audio: {exc}") from exc

        text = result.get("text", "").strip()
        print("[WhisperTranscriber] Transcription complete.")
        return text
