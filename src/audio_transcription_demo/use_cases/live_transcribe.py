from __future__ import annotations

from dataclasses import dataclass

from audio_transcription_demo.recognizers.base import SpeechRecognizer
from audio_transcription_demo.sources.base import AudioSource
from audio_transcription_demo.utils import print_section_title
from audio_transcription_demo.use_cases.base import UseCase


@dataclass(frozen=True)
class LiveTranscribeUseCase(UseCase):
    """
    Use case: "live" transcription in repeated takes.

    The user records (or selects) multiple snippets in a loop until they quit.
    This provides a live-demo feel without requiring true streaming ASR.

    Notes
    -----
    This works best with `MicrophoneAudioSource`.
    """

    @property
    def key(self) -> str:
        return "2"

    @property
    def label(self) -> str:
        return "Live transcription (repeat takes)"

    def run(self, *, source: AudioSource, recognizer: SpeechRecognizer) -> None:
        """
        Repeatedly acquire audio and transcribe until the user quits.

        Parameters
        ----------
        source:
            Audio source used to obtain each snippet.
        recognizer:
            Speech recognizer used to transcribe.

        Raises
        ------
        RuntimeError
            If acquisition or transcription fails.
        """
        print_section_title("Use case 2: Live transcription (repeat takes)")
        print("Press Enter to record/transcribe a snippet. Type 'q' then Enter to stop.\n")

        while True:
            cmd = input("Continue? [Enter/q]: ").strip().lower()
            if cmd == "q":
                break

            temp = source.get_audio()
            try:
                text = recognizer.transcribe_wav(temp.path)
            finally:
                temp.cleanup()

            print("\n--- Snippet transcript ---\n")
            print(text or "[no speech recognised]")
            print("\n--------------------------\n")
