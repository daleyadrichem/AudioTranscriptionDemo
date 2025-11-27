"""
GUI use case for microphone audio transcription.
"""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import BOTH, DISABLED, END, NORMAL, Tk, Button, Label, Text, ttk

from use_cases.base import UseCase
from utils.audio import AudioConfig, AudioRecorder
from utils.transcription import WhisperTranscriber


class AudioTranscriptionUseCase(UseCase):
    """Use case: Record microphone audio and transcribe it with Whisper.

    This use case provides a simple Tkinter GUI with:

    - A button to start/stop recording.
    - A label to show current status.
    - A text area to display the transcription result.

    Notes
    -----
    For a live demo:

    1. Click "Start Recording" and speak a short sentence.
    2. Click "Stop & Transcribe".
    3. Wait for the transcription to appear in the text box.
    """

    name: str = "Audio Transcription"

    def __init__(self) -> None:
        """Initialize the audio transcription use case."""
        self._root: Tk = Tk()
        self._root.title("AI Demo – Audio Transcription")

        self._recorder = AudioRecorder(AudioConfig())
        self._transcriber = WhisperTranscriber(model_name="small")

        self._record_button: Button
        self._status_label: Label
        self._output_text: Text
        self._progress_bar: ttk.Progressbar

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create and lay out the Tkinter widgets."""
        self._root.geometry("600x400")

        self._status_label = Label(self._root, text="Ready.", anchor="w")
        self._status_label.pack(fill=BOTH, padx=10, pady=(10, 5))

        self._record_button = Button(
            self._root,
            text="Start Recording",
            command=self._on_record_button_click,
            width=20,
        )
        self._record_button.pack(padx=10, pady=5)

        self._progress_bar = ttk.Progressbar(
            self._root, mode="indeterminate", length=200
        )
        self._progress_bar.pack(padx=10, pady=5)

        Label(self._root, text="Transcription:").pack(anchor="w", padx=10, pady=(10, 0))

        self._output_text = Text(self._root, wrap="word", height=15)
        self._output_text.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

    def _set_status(self, text: str) -> None:
        """Update the status label text."""
        self._status_label.config(text=text)
        self._root.update_idletasks()

    def _on_record_button_click(self) -> None:
        """Handle clicks on the start/stop recording button."""
        if not self._recorder.is_recording:
            self._start_recording()
        else:
            self._stop_and_transcribe()

    def _start_recording(self) -> None:
        """Start microphone recording."""
        try:
            self._recorder.start()
        except RuntimeError as exc:
            self._set_status(str(exc))
            return

        self._record_button.config(text="Stop & Transcribe")
        self._set_status("Recording... Speak now.")
        self._output_text.delete("1.0", END)

    def _stop_and_transcribe(self) -> None:
        """Stop recording and trigger transcription in a background thread."""
        # Stop recording and save to file
        audio_path = Path("recordings/demo.wav")
        try:
            wav_path = self._recorder.stop(audio_path)
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc))
            self._record_button.config(text="Start Recording")
            return

        self._set_status("Transcribing audio...")
        self._record_button.config(state=DISABLED)
        self._progress_bar.start()

        # Run transcription in a separate thread to keep the UI responsive.
        thread = threading.Thread(
            target=self._transcribe_async,
            args=(wav_path,),
            daemon=True,
        )
        thread.start()

    def _transcribe_async(self, audio_path: Path) -> None:
        """Background worker to transcribe audio and update the UI."""
        try:
            transcription = self._transcriber.transcribe(audio_path)
        except Exception as exc:  # noqa: BLE001
            transcription = f"Error during transcription: {exc}"

        # UI updates must happen on the main thread.
        self._root.after(0, self._on_transcription_complete, transcription)

    def _on_transcription_complete(self, transcription: str) -> None:
        """Handle completion of the transcription."""
        self._progress_bar.stop()
        self._record_button.config(state=NORMAL, text="Start Recording")

        self._output_text.delete("1.0", END)
        self._output_text.insert("1.0", transcription or "[No transcription produced].")

        self._set_status("Done. You can record again.")

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._root.mainloop()
