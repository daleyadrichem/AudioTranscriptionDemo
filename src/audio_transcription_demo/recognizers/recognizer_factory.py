from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .recognizer_base import SpeechRecognizer
from .vosk import VoskRecognizer
from .whisper import WhisperRecognizer


class RecognizerFactory:
    """
    Factory for creating speech recognizer backends.

    This factory provides:

    - Creation of recognizer instances
    - Listing supported recognizer models
    """

    @staticmethod
    def create(
        model_key: str,
        *,
        sample_rate: int = 16_000,
        whisper_model: str = "base",
        whisper_language: str | None = None,
        vosk_model_path: str | Path | None = None,
    ) -> SpeechRecognizer:
        """
        Create a speech recognizer instance.

        Parameters
        ----------
        model_key : str
            Name of the recognizer backend.
            Supported values are:

            - ``"whisper"``
            - ``"vosk"``

        sample_rate : int, default=16000
            Expected sample rate of audio input.

        whisper_model : str, default="base"
            Whisper model size.

        whisper_language : str | None, default=None
            Optional language override for Whisper.

        vosk_model_path : str | Path | None, default=None
            Path to a Vosk model directory.

        Returns
        -------
        SpeechRecognizer
            Initialized recognizer backend.

        Raises
        ------
        ValueError
            If an unknown recognizer model_key is provided.
        """

        key = model_key.strip().lower()

        if key == "whisper":
            return WhisperRecognizer(
                model_name=whisper_model,
                sample_rate=sample_rate,
                language=whisper_language,
            )

        if key == "vosk":
            if vosk_model_path is None:
                raise ValueError(
                    "vosk_model_path must be provided when creating a Vosk recognizer."
                )

            return VoskRecognizer(
                model_path=vosk_model_path,
                sample_rate=sample_rate,
            )

        raise ValueError(f"Unknown recognizer: {model_key}")

    @staticmethod
    def list_models() -> Dict[str, List[str]]:
        """
        List supported models for each recognizer backend.

        Returns
        -------
        dict[str, list[str]]
            Mapping of recognizer backend names to supported model identifiers.

        Notes
        -----
        - Whisper models correspond to the official OpenAI Whisper releases.
        - Vosk models must be downloaded separately and referenced by path.
        """

        whisper_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ]

        vosk_models = [
            "vosk-model-small-en-us",
            "vosk-model-en-us",
            "vosk-model-en-us-0.22",
            "vosk-model-small-en-us-0.15",
        ]

        return {
            "whisper": whisper_models,
            "vosk": vosk_models,
        }