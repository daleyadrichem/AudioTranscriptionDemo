from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from audio_transcription_demo.utils import TempPath, ensure_wav_mono
from audio_transcription_demo.sources.base import AudioSource


@dataclass
class FileAudioSource(AudioSource):
    """
    Audio source that reads from a user-provided file path.

    Parameters
    ----------
    sample_rate:
        Target WAV sample rate to convert to.
    default_path:
        Optional default file path, useful for workshop demos.

    Notes
    -----
    This source converts input audio to mono WAV at `sample_rate` using ffmpeg,
    so that recognizers can receive consistent audio.
    """

    sample_rate: int = 16_000
    default_path: Optional[Path] = None

    @property
    def label(self) -> str:
        return "Audio file (MP3/WAV/...)"

    def get_audio(self) -> TempPath:
        """
        Prompt for an audio file and return a converted WAV path.

        Returns
        -------
        TempPath
            Temporary WAV file path (delete_on_close=True).

        Raises
        ------
        FileNotFoundError
            If the chosen file does not exist.
        RuntimeError
            If conversion fails.
        """
        hint = f" [{self.default_path}]" if self.default_path else ""
        raw = input(f"Enter path to audio file{hint}: ").strip()
        src = self.default_path if (not raw and self.default_path) else Path(raw).expanduser()

        wav_path = ensure_wav_mono(src, sample_rate=self.sample_rate)
        return TempPath(path=wav_path, delete_on_close=True)
