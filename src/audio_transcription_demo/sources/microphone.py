from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from audio_transcription_demo.utils import TempPath
from audio_transcription_demo.sources.base import AudioSource


@dataclass
class MicrophoneAudioSource(AudioSource):
    """
    Audio source that records from the default microphone.

    Parameters
    ----------
    sample_rate:
        Recording sample rate in Hz.
    default_duration_seconds:
        Default duration for each recording.

    Notes
    -----
    Records mono PCM16 WAV to a temporary file and returns it.
    """

    sample_rate: int = 16_000
    default_duration_seconds: float = 10.0

    @property
    def label(self) -> str:
        return "Microphone recording"

    def get_audio(self) -> TempPath:
        """
        Record audio from the microphone and return a temporary WAV file path.

        Returns
        -------
        TempPath
            Temporary WAV file path (delete_on_close=True).

        Raises
        ------
        RuntimeError
            If recording fails or the device is unavailable.
        """
        raw = input(
            f"Recording duration in seconds [{self.default_duration_seconds}]: "
        ).strip()

        try:
            duration = float(raw) if raw else float(self.default_duration_seconds)
            if duration <= 0:
                raise ValueError
        except ValueError as exc:
            raise RuntimeError("Invalid duration. Please enter a positive number.") from exc

        print(f"\nRecording for {duration:.1f} seconds at {self.sample_rate} Hz...\n")

        try:
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Microphone recording failed: {exc}") from exc

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        sf.write(wav_path, audio, self.sample_rate, subtype="PCM_16")
        return TempPath(path=wav_path, delete_on_close=True)
