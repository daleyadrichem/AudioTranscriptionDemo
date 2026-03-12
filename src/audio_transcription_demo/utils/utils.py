from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class AudioProcessingError(RuntimeError):
    """Raised when audio conversion / processing fails."""


class BackendNotAvailableError(RuntimeError):
    """Raised when a requested optional backend dependency is unavailable."""


def get_env_var(name: str) -> str:
    """
    Read a required environment variable.

    Parameters
    ----------
    name:
        Name of the environment variable.

    Returns
    -------
    str
        The value of the environment variable.

    Raises
    ------
    ConfigurationError
        If the variable is unset or empty.
    """
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"Environment variable {name!r} is not set. "
            "Set it before running the demo."
        )
    return value


def print_section_title(title: str) -> None:
    """
    Print a terminal-friendly section header.

    Parameters
    ----------
    title:
        Title to print.
    """
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n")


def ensure_wav_mono(
    src_path: Path,
    sample_rate: int = 16_000,
) -> Path:
    """
    Convert an audio file to a mono WAV at the given sample rate using ffmpeg.

    This is used to support MP3 and other formats in a consistent way across
    recognizers.

    Parameters
    ----------
    src_path:
        Path to the input audio file (e.g., mp3, wav, m4a).
    sample_rate:
        Target sample rate in Hz.

    Returns
    -------
    Path
        Path to a temporary WAV file that the caller should delete after use.

    Raises
    ------
    FileNotFoundError
        If `src_path` does not exist.
    AudioProcessingError
        If ffmpeg is missing or the conversion fails.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Audio file not found: {src_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(out_path),
    ]

    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "ffmpeg not found. Install ffmpeg and ensure it is on PATH."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise AudioProcessingError(
            f"ffmpeg conversion failed (exit code {result.returncode}).\n{stderr}"
        )

    return out_path


@dataclass(frozen=True)
class TempPath:
    """
    Represents a path that may require cleanup.

    Parameters
    ----------
    path:
        The filesystem path.
    delete_on_close:
        If True, the caller should delete this path after use.
    """

    path: Path
    delete_on_close: bool = False

    def cleanup(self) -> None:
        """
        Delete the file if `delete_on_close` is True.

        Raises
        ------
        OSError
            If deletion fails (rare; caller may ignore if desired).
        """
        if self.delete_on_close:
            self.path.unlink(missing_ok=True)


# --- local heuristic "meeting minutes" ---------------------------------------


_STOPWORDS = {
    "the",
    "and",
    "a",
    "an",
    "of",
    "to",
    "in",
    "is",
    "it",
    "that",
    "for",
    "on",
    "with",
    "as",
    "this",
    "by",
    "at",
    "from",
    "or",
    "be",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "we",
    "you",
    "they",
    "i",
}


def _sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using simple punctuation heuristics.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    list[str]
        Sentences in order.
    """
    out: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in {".", "?", "!"}:
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
    if buf:
        s = "".join(buf).strip()
        if s:
            out.append(s)
    return out


def _sentence_score(sentence: str) -> float:
    """
    Score a sentence by counting non-stopword tokens.

    Parameters
    ----------
    sentence:
        Sentence text.

    Returns
    -------
    float
        Sentence score.
    """
    tokens = [t.strip(".,!?;:()[]").lower() for t in sentence.split()]
    tokens = [t for t in tokens if t and t not in _STOPWORDS]
    return float(len(tokens))


def format_meeting_minutes(transcript: str, max_sentences: int = 6) -> str:
    """
    Produce a simple Markdown "meeting minutes" document from a transcript.

    This is intentionally heuristic and local-only (no LLM). It selects
    high-scoring sentences as key points and mirrors them into a basic
    action-item checklist.

    Parameters
    ----------
    transcript:
        Full transcript text.
    max_sentences:
        Maximum number of key sentences to include.

    Returns
    -------
    str
        Markdown-formatted meeting minutes.
    """
    sentences = _sentence_split(transcript)
    if not sentences:
        return "No content to summarise."

    scored = [(i, s, _sentence_score(s)) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:max_sentences]
    top.sort(key=lambda x: x[0])

    highlights = [s for _, s, _ in top]
    if not highlights:
        return "No content to summarise."

    lines: List[str] = []
    lines.append("# Meeting minutes\n")
    lines.append("## Key points\n")
    for s in highlights:
        lines.append(f"- {s}")

    lines.append("\n## Action items (heuristic)\n")
    lines.append(
        "_Auto-generated from important sentences; add owners/dates manually._\n"
    )
    for s in highlights:
        lines.append(f"- [ ] (Owner?) {s}")

    return "\n".join(lines)
