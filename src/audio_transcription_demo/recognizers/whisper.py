from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Iterator

from .recognizer_base import SpeechRecognizer, TranscriptChunk


@dataclass(slots=True)
class WhisperRecognizer(SpeechRecognizer):
    """
    Speech recognition backend using OpenAI Whisper.

    Parameters
    ----------
    model_name : str, default="base"
        Whisper model size.
    sample_rate : int, default=16000
        Expected audio sample rate.
    language : str | None, default=None
        Optional language override.
    """

    model_name: str = "base"
    sample_rate: int = 16_000
    language: str | None = None

    def __post_init__(self) -> None:
        """Load Whisper model."""
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Install with: pip install openai-whisper"
            ) from exc

        self._whisper = whisper
        self._model = whisper.load_model(self.model_name)

    @property
    def label(self) -> str:
        """Return recognizer label."""
        return "whisper"

    def transcribe_file(self, path: str | Path) -> str:
        """
        Transcribe an audio file using Whisper.

        Parameters
        ----------
        path : str | Path
            Path to the audio file.

        Returns
        -------
        str
            Transcription result.
        """
        result = self._model.transcribe(
            str(path),
            language=self.language,
            fp16=False,
        )
        return result.get("text", "").strip()

    def transcribe_stream(
        self,
        stream: BinaryIO,
        *,
        chunk_size: int = 32000,
    ) -> Iterator[TranscriptChunk]:
        """
        Perform pseudo-streaming transcription.

        Whisper does not support true streaming, so audio is
        buffered and transcribed periodically.

        Parameters
        ----------
        stream : BinaryIO
            PCM audio stream.
        chunk_size : int, default=32000
            Number of bytes read per iteration.

        Yields
        ------
        TranscriptChunk
            Incremental transcription output.
        """
        pcm_buffer = bytearray()
        emitted_text = ""

        while True:
            data = stream.read(chunk_size)
            if not data:
                break

            pcm_buffer.extend(data)

            if len(pcm_buffer) < self.sample_rate * 2 * 2:
                continue

            text = self._transcribe_pcm_bytes(bytes(pcm_buffer)).strip()

            if text and text != emitted_text:
                emitted_text = text
                yield TranscriptChunk(text=text, is_final=False)

        final_text = self._transcribe_pcm_bytes(bytes(pcm_buffer)).strip()
        if final_text:
            yield TranscriptChunk(text=final_text, is_final=True)

    def _transcribe_pcm_bytes(self, pcm_bytes: bytes) -> str:
        """Convert PCM buffer to WAV and run Whisper."""
        wav_bytes = self._pcm_to_wav_bytes(pcm_bytes)

        with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()

            result = self._model.transcribe(
                tmp.name,
                language=self.language,
                fp16=False,
            )

        return result.get("text", "")

    def _pcm_to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        """Convert raw PCM audio to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        return buffer.getvalue()