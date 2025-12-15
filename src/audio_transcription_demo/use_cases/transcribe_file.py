from __future__ import annotations

from dataclasses import dataclass

from audio_transcription_demo.recognizers.base import SpeechRecognizer
from audio_transcription_demo.sources.base import AudioSource
from audio_transcription_demo.utils import print_section_title
from audio_transcription_demo.use_cases.base import UseCase


@dataclass(frozen=True)
class TranscribeFileUseCase(UseCase):
    """
    Use case: transcribe a single audio input and print the transcript.

    Notes
    -----
    This works with any AudioSource (file or microphone), but it is most
    naturally paired with the file source.
    """

    @property
    def key(self) -> str:
        return "1"

    @property
    def label(self) -> str:
        return "Transcribe one recording (file or mic)"

    def run(self, *, source: AudioSource, recognizer: SpeechRecognizer) -> None:
        """
        Acquire audio once and transcribe it.

        Parameters
        ----------
        source:
            Audio source used to obtain the audio.
        recognizer:
            Speech recognizer used to transcribe.

        Raises
        ------
        RuntimeError
            If audio acquisition or transcription fails.
        """
        print_section_title("Use case 1: Transcribe one recording")

        temp = source.get_audio()
        try:
            text = recognizer.transcribe_wav(temp.path)
        finally:
            temp.cleanup()

        print("\n--- Transcript start ---\n")
        print(text or "[no speech recognised]")
        print("\n--- Transcript end ---\n")
