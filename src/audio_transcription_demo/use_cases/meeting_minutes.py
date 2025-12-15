from __future__ import annotations

from dataclasses import dataclass

from audio_transcription_demo.recognizers.base import SpeechRecognizer
from audio_transcription_demo.sources.base import AudioSource
from audio_transcription_demo.utils import format_meeting_minutes, print_section_title
from audio_transcription_demo.use_cases.base import UseCase


@dataclass(frozen=True)
class MeetingMinutesUseCase(UseCase):
    """
    Use case: meeting minutes.

    Steps
    -----
    1) Acquire audio (typically from file).
    2) Transcribe.
    3) Convert transcript into a basic Markdown "minutes" document (heuristic).

    Notes
    -----
    The minutes generator is intentionally local-only and heuristic (no LLM).
    """

    @property
    def key(self) -> str:
        return "3"

    @property
    def label(self) -> str:
        return "Meeting minutes (transcribe + local summary)"

    def run(self, *, source: AudioSource, recognizer: SpeechRecognizer) -> None:
        """
        Acquire audio, transcribe, and produce meeting minutes.

        Parameters
        ----------
        source:
            Audio source used to obtain the meeting recording.
        recognizer:
            Speech recognizer used to transcribe.

        Raises
        ------
        RuntimeError
            If acquisition or transcription fails.
        """
        print_section_title("Use case 3: Meeting minutes")

        temp = source.get_audio()
        try:
            transcript = recognizer.transcribe_wav(temp.path)
        finally:
            temp.cleanup()

        if not transcript.strip():
            print("\nNo speech recognised; nothing to summarise.\n")
            return

        minutes = format_meeting_minutes(transcript, max_sentences=6)

        print("\n--- Meeting minutes ---\n")
        print(minutes)
        print("\n--- End ---\n")
