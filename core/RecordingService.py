import sounddevice as sd
import core.settings as settings

CHANNELS = 1


class RecordingService:

    def __init__(self):

        self._stream: sd.InputStream | None = None
        self._chunks: list[bytes] = []
        self._on_chunk = None

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            print(f"[RecordingService] status: {status}", flush=True)
        self._chunks.append(indata.tobytes())
        if self._on_chunk is not None: 
            self._on_chunk(indata.tobytes())

    def start(self, on_chunk=None) -> None:
        if self._stream is not None:
            return
        self._chunks = []
        self._on_chunk = on_chunk
        self._stream = sd.InputStream(
            samplerate=settings.SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        """Stops recording, returns all captured audio as one PCM16 byte blob."""
        if self._stream is None:
            return b""
        self._stream.stop()
        self._stream.close()
        self._stream = None
        return b"".join(self._chunks)