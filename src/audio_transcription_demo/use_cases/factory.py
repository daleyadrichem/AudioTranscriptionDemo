from __future__ import annotations

from typing import Dict

from audio_transcription_demo.use_cases.base import UseCase
from audio_transcription_demo.use_cases.live_transcribe import LiveTranscribeUseCase
from audio_transcription_demo.use_cases.meeting_minutes import MeetingMinutesUseCase
from audio_transcription_demo.use_cases.transcribe_file import TranscribeFileUseCase


def available_use_cases() -> Dict[str, UseCase]:
    """
    Build and return available use cases keyed by menu key.

    Returns
    -------
    dict[str, UseCase]
        Use cases indexed by their `.key`.
    """
    use_cases = [
        TranscribeFileUseCase(),
        LiveTranscribeUseCase(),
        MeetingMinutesUseCase(),
    ]
    return {uc.key: uc for uc in use_cases}
