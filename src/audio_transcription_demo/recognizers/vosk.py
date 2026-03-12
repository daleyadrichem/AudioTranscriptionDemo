from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from .recognizer_base import SpeechRecognizer, TranscriptChunk


@dataclass(slots=True)
class VoskRecognizer(SpeechRecognizer):
    """
    Speech recognition backend using Vosk.

    Parameters
    ----------
    model_path : str | Path
        Path to the Vosk model directory.
    sample_rate : int, default=16000
        Expected sample rate of the audio input.
    """

    model_path: str | Path
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        """Load the Vosk model."""
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise RuntimeError(
                "vosk is not installed. Install it with: pip install vosk"
            ) from exc

        self._Model = Model
        self._KaldiRecognizer = KaldiRecognizer
        self._model = Model(str(self.model_path))

    @property
    def label(self) -> str:
        """Return recognizer label."""
        return "vosk"

    def _new_recognizer(self):
        """Create a new Vosk recognizer instance."""
        recognizer = self._KaldiRecognizer(self._model, self.sample_rate)
        recognizer.SetWords(True)
        return recognizer

    def transcribe_file(self, path: str | Path) -> str:
        """
        Transcribe a WAV audio file.

        Parameters
        ----------
        path : str | Path
            Path to the WAV file.

        Returns
        -------
        str
            Transcribed text.

        Raises
        ------
        ValueError
            If the WAV format does not match Vosk requirements.
        """
        path = Path(path)

        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("Vosk requires mono WAV input.")
            if wf.getframerate() != self.sample_rate:
                raise ValueError("Invalid sample rate.")
            if wf.getsampwidth() != 2:
                raise ValueError("Expected 16-bit PCM WAV input.")

            recognizer = self._new_recognizer()
            parts: list[str] = []

            while True:
                data = wf.readframes(4000)
                if not data:
                    break

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        parts.append(text)

            final_result = json.loads(recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                parts.append(final_text)

        return " ".join(parts).strip()

    def transcribe_stream(
        self,
        stream: BinaryIO,
        *,
        chunk_size: int = 4000,
    ) -> Iterator[TranscriptChunk]:
        """
        Perform streaming transcription.

        Parameters
        ----------
        stream : BinaryIO
            Raw PCM audio stream.
        chunk_size : int, default=4000
            Number of bytes read per iteration.

        Yields
        ------
        TranscriptChunk
            Partial or finalized transcription chunks.
        """
        recognizer = self._new_recognizer()

        while True:
            data = stream.read(chunk_size)
            if not data:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    yield TranscriptChunk(text=text, is_final=True)
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text:
                    yield TranscriptChunk(text=text, is_final=False)

        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()
        if final_text:
            yield TranscriptChunk(text=final_text, is_final=True)