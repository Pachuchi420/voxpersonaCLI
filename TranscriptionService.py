import asyncio
import importlib
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any


mx: Any = None

import numpy as np
import settings


USE_MLX = platform.system() == "Darwin"

if USE_MLX:
    import mlx.core as _mx
    mx = _mx
    # NOT loading the model here. from_pretrained() below builds MLX arrays
    # that get bound to whichever thread calls it -- loading it on the main
    # thread (as this used to) and later running inference from a
    # ThreadPoolExecutor worker throws "There is no Stream(...) in current
    # thread", because that worker never owned the model's stream context.
    # Loading it inside the executor's single worker thread instead (see
    # _init_mlx_thread) keeps model + all inference on one consistent thread.

# NeMo, sherpa-onnx and Gemma are all loaded lazily (only when a
# TranscriptionService actually selects that backend), not at import time --
# a run might ask for any one of them via the `backend` constructor arg, and
# eagerly loading all three regardless would waste startup time (or
# hard-crash, or trigger an unwanted multi-GB Hugging Face download) when
# only one was actually wanted.
_nemo_model = None
_sherpa_recognizer = None
_gemma_loaded = False


def _load_nemo():
    global _nemo_model
    if _nemo_model is not None:
        return _nemo_model

    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    except ImportError:
        raise RuntimeError("NeMo not installed. On Windows run: pip install nemo_toolkit[asr]")

    try:
        # FastConformer's default attention is full bidirectional self-
        # attention over the whole clip, which is ~quadratic in audio
        # length -- re-transcribing a growing utterance-so-far buffer (see
        # TranscriptionSession, non-MLX branch) gets disproportionately
        # slower the longer someone talks. Switching to a local/limited-
        # context attention window makes each call's cost roughly linear
        # instead, which is what keeps partials from stalling out on longer
        # utterances. This is a one-time, permanent switch for the process
        # (not per-session) -- there's no other transcription use of this
        # model that needs full-context accuracy.
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[128, 128],
        )
    except Exception as e:
        print(f"[TranscriptionService] Could not switch to local attention ({e}) -- "
              f"falling back to full-context attention, which will get slower on longer utterances.", flush=True)

    _nemo_model = model
    return _nemo_model


def _load_sherpa():
    """Loads a sherpa-onnx OfflineRecognizer over a Parakeet-TDT ONNX export.
    This is the A/B alternative to NeMo on Windows -- pure ONNX Runtime, no
    PyTorch/NeMo/CUDA dependency, meant to be lighter and faster to start."""
    global _sherpa_recognizer
    if _sherpa_recognizer is not None:
        return _sherpa_recognizer

    import os
    try:
        import sherpa_onnx
    except ImportError:
        raise RuntimeError("sherpa-onnx not installed. Run: pip install sherpa-onnx")

    model_dir = os.environ.get(
        "SHERPA_MODEL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sherpa_model"),
    )
    required_files = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
    missing = [name for name in required_files if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        raise RuntimeError(
            f"sherpa-onnx model files missing from {model_dir}: {missing}. Download the "
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 package from "
            "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 "
            f"and place encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx and tokens.txt in {model_dir} "
            "(or set the SHERPA_MODEL_DIR environment variable to point elsewhere)."
        )

    _sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=os.path.join(model_dir, "encoder.int8.onnx"),
        decoder=os.path.join(model_dir, "decoder.int8.onnx"),
        joiner=os.path.join(model_dir, "joiner.int8.onnx"),
        tokens=os.path.join(model_dir, "tokens.txt"),
        num_threads=2,
        sample_rate=settings.SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        # Without this, current sherpa-onnx releases fail with "'vocab_size'
        # does not exist in the metadata" -- this NeMo-exported model doesn't
        # populate the generic transducer metadata fields sherpa-onnx expects
        # by default, and needs the NeMo-specific decoder path instead. Model
        # files themselves are correct as downloaded; don't modify them (see
        # https://github.com/k2-fsa/sherpa-onnx/issues/2226).
        model_type="nemo_transducer",
    )
    return _sherpa_recognizer


def _load_gemma():
    """Loads Gemma 4 E2B via GemmaService (downloading it from Hugging Face
    on first use -- see GemmaService.ensure_loaded). Returns a truthy
    sentinel rather than the raw model/processor pair:
    TranscriptionService.model is only ever used as a "is a backend loaded"
    marker on this path (actual inference goes through
    GemmaService.transcribe_audio in _transcribe_array_sync), matching how
    self.model is used for the other non-MLX backends."""
    global _gemma_loaded
    if _gemma_loaded:
        return True

    from GemmaService import ensure_loaded
    ensure_loaded()
    _gemma_loaded = True
    return True


_mlx_thread_local = threading.local()


def _init_mlx_thread():
    from parakeet_mlx import from_pretrained
    _mlx_thread_local.model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")


class TranscriptionService:
    def __init__(self, backend: str | None = None):
        # backend is one of "mlx", "nemo", "sherpa", "gemma". Defaults to
        # platform-based auto-detection when not passed explicitly,
        # preserving prior behavior: MLX streaming on Mac, NeMo (the "NeMo
        # checkpoint") on everything else. Callers can override via
        # BackService.py's CLI flag to A/B test sherpa-onnx or Gemma 4 E2B
        # instead.
        self.backend = backend or ("mlx" if USE_MLX else "nemo")
        self.use_mlx = self.backend == "mlx"

        if self.backend == "mlx":
            self.model: Any = None
        elif self.backend == "nemo":
            self.model = _load_nemo()
        elif self.backend == "sherpa":
            self.model = _load_sherpa()
        elif self.backend == "gemma":
            self.model = _load_gemma()
        else:
            raise ValueError(f"Unknown transcription backend: {self.backend!r}")

        # Single dedicated worker thread: the MLX model is loaded here (see
        # _init_mlx_thread) so every mx call for it happens on the one
        # thread that actually owns its stream context, and max_workers=1
        # serializes GPU access so concurrent partial-transcription tasks
        # can't hit the Metal context at the same time.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            initializer=_init_mlx_thread if self.use_mlx else None,
        )
        if self.use_mlx:
            # Force the initializer (and model load) to happen now instead
            # of lazily on the first real transcribe() call, so the load
            # cost is paid once at startup, not on the first utterance.
            self._executor.submit(lambda: None).result()

    def pcm16_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return audio_i16.astype(np.float32) / 32768.0

    def start_session(self) -> "TranscriptionSession":
        # One session per utterance -- see TranscriptionSession for why this
        # replaces the old one-shot transcribe(whole_buffer_so_far) call.
        return TranscriptionSession(self)

    def _transcribe_array_sync(self, audio_f32: np.ndarray) -> str:
        """One-shot transcription of a full array. Only used for the non-MLX
        (NeMo/sherpa-onnx/Gemma) fallbacks, none of which has an add_audio()-
        style streaming API wired up here -- MLX sessions never call this,
        they use add_audio() incrementally."""
        if len(audio_f32) < settings.SAMPLE_RATE:
            return ""

        if self.backend == "sherpa":
            stream = self.model.create_stream()
            stream.accept_waveform(settings.SAMPLE_RATE, audio_f32)
            self.model.decode_stream(stream)
            result = stream.result
            # sherpa-onnx has returned either a plain string or a result
            # object with a .text attribute depending on version/example --
            # handle both rather than guessing wrong and silently breaking.
            return result.text if hasattr(result, "text") else result

        if self.backend == "gemma":
            from GemmaService import transcribe_audio
            return transcribe_audio(audio_f32, settings.SAMPLE_RATE)

        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_f32, settings.SAMPLE_RATE)
            result = self.model.transcribe([f.name])
            hypothesis = result[0] if isinstance(result, list) else result
            return hypothesis.text if hasattr(hypothesis, "text") else hypothesis


class TranscriptionSession:
    """Wraps one streaming transcription pass over a single utterance.

    parakeet-mlx's transcribe_stream() is a real incremental streaming API
    (add_audio() carries encoder cache/state across calls) -- it's meant to
    be opened once per utterance and fed only new audio each time, not
    reopened and replayed from the start on every partial like the old
    transcribe(whole_buffer_so_far) call did. That re-transcribe-from-scratch
    pattern was why latency grew with utterance length (a 3-partial, 7s
    utterance was pushing ~16s of audio through the model instead of 7s).

    All work still runs on TranscriptionService's single dedicated MLX
    worker thread (self._service._executor) to preserve the model's
    thread-affine stream context.

    The non-MLX (NeMo/sherpa-onnx/Gemma) path has no equivalent incremental-
    cache API wired up here -- NeMo exposes real cache-aware streaming, but
    only through a much lower-level, manually-threaded-cache API, not a
    simple add_audio() call, and Gemma's audio encoder has no streaming API
    at all. As a middle ground, this path re-transcribes the accumulated
    buffer-so-far on every feed() (same as the old MLX behavior), but relies
    on NeMo's module-level change_attention_model() call to a local/limited-
    context window (not applicable to sherpa-onnx or Gemma) so that cost
    stays roughly linear in buffer length instead of ~quadratic where
    possible, and on VADService's MAX_BUFFER_SECONDS cap to bound the worst
    case regardless of backend. Not as fast as true incremental streaming,
    but avoids the growing-utterance latency spikes.
    """

    def __init__(self, service: TranscriptionService):
        self._service = service
        self._transcriber = None  # entered StreamingParakeet, MLX path only
        self._chunks: list[np.ndarray] = []  # accumulated audio, non-MLX fallback only

    async def feed(self, pcm_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._service._executor, self._feed_sync, pcm_bytes)

    async def finish(self, pcm_bytes: bytes = b"") -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._service._executor, self._finish_sync, pcm_bytes)

    def _feed_sync(self, pcm_bytes: bytes) -> str:
        audio_f32 = self._service.pcm16_bytes_to_float32(pcm_bytes)

        if self._service.use_mlx:
            if self._transcriber is None:
                model = _mlx_thread_local.model
                self._transcriber = model.transcribe_stream(context_size=(256, 256))
                self._transcriber.__enter__()
            if len(audio_f32) > 0:
                self._transcriber.add_audio(mx.array(audio_f32))
            return self._transcriber.result.text
        else:
            if len(audio_f32) > 0:
                self._chunks.append(audio_f32)
            if not self._chunks:
                return ""
            # Re-transcribes the whole utterance-so-far each call (no real
            # incremental cache on this path -- see class docstring), bounded
            # by VADService's MAX_BUFFER_SECONDS cap on how much audio ever
            # accumulates here.
            full = np.concatenate(self._chunks)
            return self._service._transcribe_array_sync(full)

    def _finish_sync(self, pcm_bytes: bytes) -> str:
        text = self._feed_sync(pcm_bytes)

        if self._service.use_mlx:
            if self._transcriber is not None:
                text = self._transcriber.result.text
                self._transcriber.__exit__(None, None, None)
                self._transcriber = None
            return text
        else:
            # _feed_sync already returns the full accumulated transcript.
            self._chunks = []
            return text
