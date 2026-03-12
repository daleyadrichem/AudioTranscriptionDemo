from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator

from src.audio_transcription_demo.sources.source_base import AudioSource
from src.audio_transcription_demo.recognizers.recognizer_base import (
    SpeechRecognizer,
    TranscriptChunk,
)

from src.audio_transcription_demo.utils.utils import TempPath, ensure_wav_mono


@dataclass
class FileAudioSource(AudioSource):
    """
    Audio source that reads from a single file on disk.

    The file is converted to a normalized WAV format before being passed
    to recognizers so that all backends receive consistent audio.

    Parameters
    ----------
    sample_rate : int, default=16000
        Target sample rate used when converting audio to WAV.
    default_path : pathlib.Path | None, default=None
        Optional fallback file path used if the user provides no input.
    """

    sample_rate: int = 16_000
    default_path: Optional[Path] = None

    @property
    def label(self) -> str:
        """
        Human-readable label used in menus.

        Returns
        -------
        str
            Source label.
        """
        return "Audio file (single)"

    def get_audio(self, source: str | Path | None = None) -> TempPath:
        """
        Resolve and normalize an audio file.

        The input file is converted to mono WAV at the configured sample
        rate so it can be used consistently by recognizers.

        Parameters
        ----------
        source : str | pathlib.Path | None, default=None
            Optional file path. If ``None`` the user will be prompted.

        Returns
        -------
        TempPath
            Temporary normalized WAV file.

        Raises
        ------
        FileNotFoundError
            If the provided file does not exist.
        RuntimeError
            If audio conversion fails.
        ValueError
            If no path is provided and no default exists.
        """
        src = self._resolve_path(source)
        wav_path = ensure_wav_mono(src, sample_rate=self.sample_rate)
        return TempPath(path=wav_path, delete_on_close=True)

    def transcribe(
        self,
        recognizer: SpeechRecognizer,
        *,
        source: str | Path | None = None,
        stream: bool = False,
        chunk_size: int = 4000,
    ) -> str | Iterator[TranscriptChunk]:
        """
        Transcribe a single audio file using the provided recognizer.

        Parameters
        ----------
        recognizer : SpeechRecognizer
            Speech recognizer implementation.
        source : str | pathlib.Path | None, default=None
            Optional file path to transcribe.
        stream : bool, default=False
            If True, perform streaming transcription.
        chunk_size : int, default=4000
            Chunk size used when streaming PCM audio.

        Returns
        -------
        str | Iterator[TranscriptChunk]
            Final transcription if ``stream=False`` or an iterator of
            transcription chunks if ``stream=True``.
        """
        audio = self.get_audio(source)

        if not stream:
            try:
                return recognizer.transcribe_file(audio.path)
            finally:
                audio.close()

        pcm_bytes = self._read_pcm_bytes(audio.path)
        audio.close()

        stream_buffer = io.BytesIO(pcm_bytes)
        return recognizer.transcribe_stream(stream_buffer, chunk_size=chunk_size)

    def _resolve_path(self, source: str | Path | None) -> Path:
        """
        Resolve the audio file path.

        Parameters
        ----------
        source : str | pathlib.Path | None
            Optional caller-provided file path.

        Returns
        -------
        pathlib.Path
            Resolved file path.

        Raises
        ------
        ValueError
            If no path is provided and no default path exists.
        """
        if source is not None:
            path = Path(source).expanduser()
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        hint = f" [{self.default_path}]" if self.default_path else ""
        raw = input(f"Enter path to audio file{hint}: ").strip()

        if raw:
            path = Path(raw).expanduser()
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        if self.default_path is not None:
            path = self.default_path.expanduser()
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        raise ValueError("No file path provided.")

    def _read_pcm_bytes(self, wav_path: Path) -> bytes:
        """
        Extract PCM bytes from a normalized WAV file.

        Parameters
        ----------
        wav_path : pathlib.Path
            Path to a WAV file.

        Returns
        -------
        bytes
            Raw PCM audio frames.

        Raises
        ------
        ValueError
            If the WAV format does not match expected parameters.
        """
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("Expected mono WAV input.")
            if wf.getframerate() != self.sample_rate:
                raise ValueError("Unexpected sample rate.")
            if wf.getsampwidth() != 2:
                raise ValueError("Expected 16-bit PCM.")

            return wf.readframes(wf.getnframes())