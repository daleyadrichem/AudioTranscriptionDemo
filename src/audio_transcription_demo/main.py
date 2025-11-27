"""
Entry point for AI demo use cases.
"""

from use_cases.audio_transcription import AudioTranscriptionUseCase


def main() -> None:
    """Run the selected AI use case."""
    use_case = AudioTranscriptionUseCase()
    use_case.run()


if __name__ == "__main__":
    main()
