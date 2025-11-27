"""
Audio recording utilities for microphone input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
import wave


@dataclass
class AudioConfig:
    """Configuration for audio recording.

    Parameters
    ----------
    samplerate : int
        Sampling rate in Hz.
    channels : int
        Number of audio channels (1 for mono, 2 for stereo).
    sample_width : int
        Sample width in bytes (2 for 16-bit PCM).
    """

    samplerate: int = 16_000
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM


class AudioRecorder:
    """Simple microphone audio recorder using `sounddevice`.

    Notes
    -----
    - Recording is done via an input stream callback.
    - Recorded data is stored in memory until :meth:`stop` is called.

    Examples
    --------
    >>> recorder = AudioRecorder()
    >>> recorder.start()
    >>> # ... speak into the microphone ...
    >>> audio_path = recorder.stop(Path("recordings/demo.wav"))
    """

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        """Initialize the audio recorder.

        Parameters
        ----------
        config : AudioConfig, optional
            Configuration for the recording. If ``None``, a default
            mono 16 kHz configuration is used.
        """
        self.config: AudioConfig = config or AudioConfig()
        self._stream: Optional[sd.InputStream] = None
        self._frames: List[np.ndarray] = []
        self._is_recording: bool = False

    @property
    def is_recording(self) -> bool:
        """bool: Whether recording is currently active."""
        return self._is_recording

    def _callback(self, indata: np.ndarray, _frames: int, _time, status) -> None:
        """Internal callback for the sounddevice stream."""
        if status:
            # In a production app you might want to log this
            print(f"[AudioRecorder] Stream status: {status}")
        if self._is_recording:
            # Copy data to avoid referencing the same buffer
            self._frames.append(indata.copy())

    def start(self) -> None:
        """Start recording from the default microphone.

        Raises
        ------
        RuntimeError
            If recording is already in progress.
        """
        if self._is_recording:
            raise RuntimeError("Recording is already in progress.")

        self._frames.clear()

        self._stream = sd.InputStream(
            samplerate=self.config.samplerate,
            channels=self.config.channels,
            callback=self._callback,
        )
        self._stream.start()
        self._is_recording = True
        print("[AudioRecorder] Recording started.")

    def stop(self, output_path: Path) -> Path:
        """Stop recording and write audio to a WAV file.

        Parameters
        ----------
        output_path : Path
            Destination path for the WAV file. Parent directories
            are created if needed.

        Returns
        -------
        Path
            The path to the written WAV file.

        Raises
        ------
        RuntimeError
            If no recording is in progress.
        ValueError
            If no audio frames were captured.
        """
        if not self._is_recording:
            raise RuntimeError("No recording is currently in progress.")

        self._is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            raise ValueError("No audio data captured; cannot write file.")

        audio_data = np.concatenate(self._frames, axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 [-1, 1] to int16
        int_data = np.clip(audio_data, -1.0, 1.0)
        int_data = (int_data * (2 ** (8 * self.config.sample_width - 1) - 1)).astype(
            np.int16
        )

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.config.sample_width)
            wf.setframerate(self.config.samplerate)
            wf.writeframes(int_data.tobytes())

        print(f"[AudioRecorder] Recording stopped. Saved to: {output_path}")
        return output_path
