from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import sounddevice as sd

from src.audio_transcription_demo.sources.source_base import AudioSource
from src.audio_transcription_demo.recognizers.recognizer_base import (
    SpeechRecognizer,
    TranscriptChunk,
)


@dataclass
class MicrophoneAudioSource(AudioSource):
    """
    Audio source for live microphone transcription.

    This source supports two input modes:

    - local device capture using ``sounddevice``
    - remote/pushed PCM audio via a binary stream

    The remote mode is useful for API deployments where audio is captured
    on a laptop, browser, or other client and streamed to a server or
    Docker container over WebSocket or HTTP.

    Parameters
    ----------
    sample_rate : int, default=16000
        Audio sample rate in Hz.
    channels : int, default=1
        Number of audio channels.
    dtype : str, default="int16"
        PCM sample dtype used by local capture.
    block_size : int, default=4000
        Number of frames captured per callback when using local capture.
    """

    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "int16"
    block_size: int = 4000

    @property
    def label(self) -> str:
        """
        Menu label for the microphone source.

        Returns
        -------
        str
            Source label.
        """
        return "Microphone"

    def get_audio(self):
        """
        Microphone source does not support returning a file.

        Raises
        ------
        NotImplementedError
            Always raised because microphone audio must be streamed.
        """
        raise NotImplementedError("Microphone audio must be streamed.")

    def create_push_stream(self) -> "_PushAudioInputStream":
        """
        Create a writable push stream for remote microphone audio.

        This is intended for API use cases where the client captures audio
        and sends raw PCM chunks to the server. The server can push those
        chunks into the returned object and pass it to ``transcribe``.

        Returns
        -------
        _PushAudioInputStream
            Writable/readable PCM stream.
        """
        return _PushAudioInputStream()

    def transcribe(
        self,
        recognizer: SpeechRecognizer,
        *,
        chunk_size: int = 4000,
        audio_stream: BinaryIO | None = None,
        stream: bool = True,
    ) -> Iterator[TranscriptChunk]:
        """
        Perform live transcription from a local or remote microphone stream.

        Parameters
        ----------
        recognizer : SpeechRecognizer
            Recognizer implementation used for transcription.
        chunk_size : int, default=4000
            Chunk size passed to the recognizer.
        audio_stream : BinaryIO | None, default=None
            Optional externally provided binary audio stream. If provided,
            local microphone capture is skipped and this stream is used
            instead. This is the mode to use in Docker/API deployments.
        stream : bool, default=True
            Included for compatibility with the base interface. Microphone
            transcription is always streaming and this argument is ignored.

        Returns
        -------
        Iterator[TranscriptChunk]
            Streaming transcription results.
        """
        if audio_stream is not None:
            return recognizer.transcribe_stream(audio_stream, chunk_size=chunk_size)

        local_stream = _LocalMicrophoneStream(
            sample_rate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            block_size=self.block_size,
        )
        return recognizer.transcribe_stream(local_stream, chunk_size=chunk_size)


class _LocalMicrophoneStream:
    """
    File-like PCM stream backed by a local microphone device.

    This class provides a ``read`` interface compatible with recognizer
    streaming APIs.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        dtype: str,
        block_size: int,
    ) -> None:
        """
        Initialize the local microphone stream.

        Parameters
        ----------
        sample_rate : int
            Audio capture sample rate.
        channels : int
            Number of channels.
        dtype : str
            Audio sample type.
        block_size : int
            Buffer size for each capture block.
        """
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._closed = False
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            blocksize=block_size,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        """
        Audio capture callback.

        Parameters
        ----------
        indata : bytes
            Recorded PCM frames.
        frames : int
            Number of frames captured.
        time_info : dict
            Timestamp metadata.
        status : object
            Stream status flags.
        """
        if self._closed:
            return
        self._queue.put(bytes(indata))

    def read(self, size: int = -1) -> bytes:
        """
        Read audio data from the stream.

        Parameters
        ----------
        size : int, default=-1
            Requested number of bytes. Ignored for live capture because
            audio is returned one capture block at a time.

        Returns
        -------
        bytes
            PCM audio chunk.
        """
        if self._closed:
            return b""
        return self._queue.get()

    def close(self) -> None:
        """
        Stop and close the audio stream.
        """
        if self._closed:
            return
        self._closed = True
        self._stream.stop()
        self._stream.close()


class _PushAudioInputStream:
    """
    File-like PCM stream that accepts externally pushed audio chunks.

    This stream is intended for remote microphone scenarios, such as a
    browser or desktop client sending PCM audio to a server over a
    WebSocket connection.

    Notes
    -----
    The producer should push raw PCM bytes and call ``close`` when the
    client stops sending audio.
    """

    def __init__(self) -> None:
        """
        Initialize the push stream.
        """
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._closed = False

    def write(self, data: bytes) -> int:
        """
        Push audio data into the stream.

        Parameters
        ----------
        data : bytes
            Raw PCM audio bytes.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        RuntimeError
            If the stream has already been closed.
        TypeError
            If ``data`` is not bytes-like.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed stream.")

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Audio chunks must be bytes-like.")

        payload = bytes(data)
        if payload:
            self._queue.put(payload)
        return len(payload)

    def read(self, size: int = -1) -> bytes:
        """
        Read audio data from the stream.

        Parameters
        ----------
        size : int, default=-1
            Requested number of bytes. Ignored because data is returned
            in pushed chunk units.

        Returns
        -------
        bytes
            Next available PCM audio chunk, or ``b""`` when the stream
            is closed and fully drained.
        """
        item = self._queue.get()
        if item is None:
            self._closed = True
            return b""
        return item

    def close(self) -> None:
        """
        Mark the stream as finished.

        After closing, readers will eventually receive ``b""`` once all
        queued audio has been consumed.
        """
        if self._closed:
            return
        self._queue.put(None)