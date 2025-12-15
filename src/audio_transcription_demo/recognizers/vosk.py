from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import soundfile as sf

from audio_transcription_demo.utils import AudioProcessingError, ConfigurationError, get_env_var
from audio_transcription_demo.recognizers.base import SpeechRecognizer


@dataclass
class VoskSpeechRecognizer(SpeechRecognizer):
    """
    Vosk speech recognizer (fully local).

    Vosk requires a downloaded model directory. Set it via the environment
    variable `VOSK_MODEL_PATH`.

    Parameters
    ----------
    model_path:
        Optional explicit path to a Vosk model directory. If None, reads
        from `VOSK_MODEL_PATH`.
    sample_rate:
        Sample rate used for microphone recording and expected WAV input.

    Raises
    ------
    ConfigurationError
        If the model path is missing or invalid.
    ImportError
        If the `vosk` package is not installed.
    """

    model_path: Optional[Path] = None
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        from vosk import Model  # type: ignore[import-not-found]

        if self.model_path is None:
            self.model_path = Path(get_env_var("VOSK_MODEL_PATH"))

        if not self.model_path.exists():
            raise ConfigurationError(f"Vosk model not found at: {self.model_path}")

        self._model = Model(str(self.model_path))

    def transcribe_wav(self, wav_path: Path) -> str:
        """
        Transcribe a WAV file using Vosk.

        Parameters
        ----------
        wav_path:
            Path to a WAV file. Must be mono PCM16 at `sample_rate`.

        Returns
        -------
        str
            Transcript text.

        Raises
        ------
        FileNotFoundError
            If `wav_path` does not exist.
        AudioProcessingError
            If the WAV sample rate does not match `sample_rate`.
        RuntimeError
            If Vosk fails during decoding.
        """
        from vosk import KaldiRecognizer  # type: ignore[import-not-found]

        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        audio, sr = sf.read(wav_path, dtype="int16")
        if sr != self.sample_rate:
            raise AudioProcessingError(
                f"Vosk expects {self.sample_rate} Hz; got {sr} Hz. "
                "Convert audio before transcribing."
            )

        recognizer = KaldiRecognizer(self._model, self.sample_rate)
        recognizer.SetWords(True)

        audio_bytes = audio.tobytes()
        chunk_size = 4000
        parts: List[str] = []

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            if recognizer.AcceptWaveform(chunk):
                res = json.loads(recognizer.Result())
                text = res.get("text", "")
                if text:
                    parts.append(text)

        final = json.loads(recognizer.FinalResult())
        text = final.get("text", "")
        if text:
            parts.append(text)

        return " ".join(parts).strip()
