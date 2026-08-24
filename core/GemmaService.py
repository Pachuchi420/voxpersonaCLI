"""Gemma 4 E2B ASR, via native MLX.

TranscriptionService's "gemma" backend (see _load_gemma there) prompts
Gemma 4 E2B ("mlx-community/gemma-4-e2b-it-4bit") to transcribe raw audio to
text. Response generation is handled separately by core/llm.py's
Ollama-based path (already wired into TextChatScreen/AudioChatScreen) --
this module only covers ASR.

Nothing is downloaded at import time -- mlx_vlm.load() below only fetches
weights from the Hugging Face Hub the first time the model is actually
loaded (i.e. the first time a caller picks the "gemma" backend), and caches
them under ~/.cache/huggingface for every run after that.

Runs on mlx-vlm (native MLX), not transformers/PyTorch -- MLX is purpose-
built for Apple Silicon's unified memory and is both faster and far lighter
here: this 4-bit checkpoint is ~2GB vs. ~10GB for the bf16 transformers
checkpoint, which matters on lower-RAM Macs (e.g. a 16GB M2, where the
bf16 model already eats most of the unified memory budget). The
transformers/PyTorch path also needed attn_implementation="eager" to work
around a bf16-on-MPS bug (sdpa produced NaN/Inf logits -- greedy decoding
died after ~1 token); MLX's own attention kernels don't have that problem.

Threading: MLX arrays are bound to whatever thread creates them -- using
them from a different thread raises "There is no Stream(...) in current
thread" (the same constraint TranscriptionService's parakeet_mlx "mlx"
backend already works around via its own dedicated worker thread, see
_init_mlx_thread there). This module gets its own single dedicated thread
that every call -- model loading included -- is funneled through via
_executor below, regardless of which outer thread actually invoked
transcribe_audio()/ensure_loaded().
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

_model: Any = None
_processor: Any = None
_config: Any = None
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def _load_gemma_impl():
    global _model, _processor, _config
    if _model is not None:
        return _model, _processor

    with _lock:
        if _model is not None:
            return _model, _processor

        try:
            from mlx_vlm import load
        except ImportError as e:
            raise RuntimeError(
                "Gemma 4 support requires mlx-vlm. Run: pip install -U mlx-vlm"
            ) from e

        print(
            f"[GemmaService] Loading {MODEL_ID} via MLX -- first run downloads "
            "the model from Hugging Face (needs ~2GB free disk); it's cached "
            "under ~/.cache/huggingface after that.",
            flush=True,
        )

        model, processor = load(MODEL_ID)

        _model = model
        _processor = processor
        _config = model.config
        print("[GemmaService] Gemma 4 E2B (MLX) ready.", flush=True)
        return _model, _processor


def ensure_loaded():
    """Loads the model (if not already) on GemmaService's dedicated MLX
    thread and blocks until done. Safe to call eagerly, from any thread --
    e.g. TranscriptionService's "gemma" backend calls this at construction
    time to pay the load cost once upfront instead of on first use."""
    _executor.submit(_load_gemma_impl).result()


def _transcribe_audio_impl(audio_f32, sample_rate: int) -> str:
    import tempfile

    import soundfile as sf
    from mlx_vlm import apply_chat_template, generate

    model, processor = _load_gemma_impl()

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio_f32, sample_rate)

        prompt = apply_chat_template(
            processor,
            _config,
            "Transcribe this audio exactly, verbatim, in the language spoken. "
            "Output only the transcript, nothing else.",
            num_audios=1,
        )
        result = generate(
            model, processor, prompt, audio=f.name, max_tokens=256, temperature=0.0
        )

    return result.text.strip()


def transcribe_audio(audio_f32, sample_rate: int) -> str:
    """One-shot ASR: prompts Gemma 4 E2B's audio encoder to transcribe a
    float32 PCM array (mono, `sample_rate` Hz) to text. Used by
    TranscriptionService's "gemma" backend."""
    return _executor.submit(_transcribe_audio_impl, audio_f32, sample_rate).result()
