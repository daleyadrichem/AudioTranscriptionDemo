from __future__ import annotations

from pathlib import Path
from typing import Optional

from audio_transcription_demo.recognizers.factory import create_recognizer
from audio_transcription_demo.sources.factory import create_source
from audio_transcription_demo.use_cases.factory import available_use_cases
from audio_transcription_demo.utils import BackendNotAvailableError, print_section_title


def _choose_recognizer_name() -> str:
    """
    Prompt the user to select a recognizer backend.

    Returns
    -------
    str
        Recognizer backend name.
    """
    while True:
        print_section_title("Select speech recognizer backend")
        print("  [1] vosk")
        print("  [2] whisper (optional extra)")
        print("  [3] speechbrain (optional extra)\n")

        choice = input("Choose backend [1/2/3]: ").strip()
        mapping = {"1": "vosk", "2": "whisper", "3": "speechbrain"}
        name = mapping.get(choice)
        if name:
            return name

        print("\nInvalid choice. Try again.\n")


def _choose_source_name() -> str:
    """
    Prompt the user to select an audio source.

    Returns
    -------
    str
        Source name.
    """
    while True:
        print_section_title("Select audio source")
        print("  [1] file")
        print("  [2] microphone\n")

        choice = input("Choose source [1/2]: ").strip()
        mapping = {"1": "file", "2": "microphone"}
        name = mapping.get(choice)
        if name:
            return name

        print("\nInvalid choice. Try again.\n")


def _optional_default_file_path() -> Optional[Path]:
    """
    Ask for an optional default audio path for the file source.

    Returns
    -------
    pathlib.Path | None
        Default path or None.
    """
    raw = input("Optional: default audio file path (Enter to skip): ").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def main() -> None:
    """
    Entry point for the interactive demo.

    Workflow
    --------
    1) Select recognizer backend (factory).
    2) Select audio source (factory).
    3) Select use case (factory).
    4) Run chosen use case using the chosen source + recognizer.
    """
    recognizer_name = _choose_recognizer_name()

    try:
        recognizer = create_recognizer(recognizer_name)
    except BackendNotAvailableError as exc:
        print(f"\nBackend not available: {exc}\n")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"\nFailed to initialize recognizer: {exc}\n")
        return

    source_name = _choose_source_name()
    default_file = _optional_default_file_path() if source_name == "file" else None

    # For recognizers like Vosk we want consistent 16kHz WAV; we use the recognizer's sample rate.
    source = create_source(
        source_name,
        sample_rate=getattr(recognizer, "sample_rate", 16_000),
        default_file_path=default_file,
    )

    use_cases = available_use_cases()

    while True:
        print_section_title("Select use case")
        for key in sorted(use_cases.keys()):
            print(f"  [{key}] {use_cases[key].label}")
        print("  [q] Quit\n")

        choice = input("Your choice: ").strip().lower()
        if choice == "q":
            print("\nGoodbye!\n")
            return

        use_case = use_cases.get(choice)
        if not use_case:
            print("\nInvalid choice. Try again.\n")
            continue

        use_case.run(source=source, recognizer=recognizer)
        input("Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
