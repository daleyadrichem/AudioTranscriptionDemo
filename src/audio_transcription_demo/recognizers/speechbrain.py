from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audio_transcription_demo.recognizers.base import SpeechRecognizer


@dataclass
class SpeechBrainSpeechRecognizer(SpeechRecognizer):
    """
    SpeechBrain speech recognizer (local, research toolkit).

    Uses a pretrained model via SpeechBrain's `EncoderDecoderASR`.

    Parameters
    ----------
    model_source:
        SpeechBrain pretrained model identifier.
    savedir:
        Local directory where the model is cached/downloaded.
    sample_rate:
        Preferred microphone sample rate (recording).

    Raises
    ------
    ImportError
        If `speechbrain` is not installed.
    RuntimeError
        If the pretrained model cannot be loaded.
    """

    model_source: str = "speechbrain/asr-transformer-transformerlm-librispeech"
    savedir: str = "pretrained_models/asr-transformer-transformerlm-librispeech"
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        from speechbrain.pretrained import EncoderDecoderASR  # type: ignore[import-not-found]

        self._asr = EncoderDecoderASR.from_hparams(
            source=self.model_source,
            savedir=self.savedir,
        )

    def transcribe_wav(self, wav_path: Path) -> str:
        """
        Transcribe audio using SpeechBrain.

        Parameters
        ----------
        wav_path:
            Path to an audio file (SpeechBrain can read common formats).

        Returns
        -------
        str
            Transcript text.

        Raises
        ------
        FileNotFoundError
            If `wav_path` does not exist.
        RuntimeError
            If SpeechBrain transcription fails.
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        return str(self._asr.transcribe_file(str(wav_path))).strip()

    def transcribe_any(self, audio_path: Path) -> str:
        """
        Transcribe arbitrary audio supported by SpeechBrain.

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
            If SpeechBrain transcription fails.
        """
        return self.transcribe_wav(audio_path)
