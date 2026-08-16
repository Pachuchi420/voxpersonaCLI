"""Shared Gemma 4 E2B model loader + inference helpers, via native MLX.

Gemma 4 E2B ("mlx-community/gemma-4-e2b-it-4bit") is a multimodal model
with native audio support (E2B/E4B/12B only), so one loaded instance can
serve two different roles in this pipeline:
  - ASR: TranscriptionService's "gemma" backend (see _load_gemma there)
    prompts it to transcribe raw audio to text.
  - Response generation: ResponseService prompts it (text-only) to produce
    the agent's next line given the conversation so far.

Both roles share the single process-wide instance below so selecting
"gemma" for both doesn't load the model twice. Nothing is downloaded at
import time -- mlx_vlm.load() below only fetches weights from the Hugging
Face Hub the first time the model is actually loaded (i.e. the first time a
caller picks the "gemma" backend/engine), and caches them under
~/.cache/huggingface for every run after that.

Runs on mlx-vlm (native MLX), not transformers/PyTorch -- MLX is purpose-
built for Apple Silicon's unified memory and is both faster and far lighter
here: this 4-bit checkpoint is ~2GB vs. ~10GB for the bf16 transformers
checkpoint, which matters on lower-RAM Macs (e.g. a 16GB M2, where the
bf16 model already eats most of the unified memory budget). The
transformers/PyTorch path also needed attn_implementation="eager" to work
around a bf16-on-MPS bug (sdpa produced NaN/Inf logits -- greedy decoding
died after ~1 token, and sampling crashed outright); MLX's own attention
kernels don't have that problem.

Threading: MLX arrays are bound to whatever thread creates them -- using
them from a different thread raises "There is no Stream(...) in current
thread" (the same constraint TranscriptionService's parakeet_mlx "mlx"
backend already works around via its own dedicated worker thread, see
_init_mlx_thread there). GemmaService is shared between two callers that
each run on their *own* separate thread pool (TranscriptionService's
dedicated ASR worker vs. BackService's default executor for response
generation), so it needs its own single dedicated thread that every call --
model loading included -- is funneled through via _executor below,
regardless of which outer thread actually invoked transcribe_audio()/
generate_text()/ensure_loaded().
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

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


def _generate_text_impl(system_prompt: Optional[str], conversation: list[dict], max_new_tokens: int) -> str:
    from mlx_vlm import apply_chat_template, generate

    model, processor = _load_gemma_impl()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in conversation:
        messages.append({"role": turn["role"], "content": turn["text"]})

    prompt = apply_chat_template(processor, _config, messages)
    result = generate(
        model, processor, prompt, max_tokens=max_new_tokens, temperature=0.7, top_p=0.9
    )

    return result.text.strip()


def generate_text(system_prompt: Optional[str], conversation: list[dict], max_new_tokens: int = 128) -> str:
    """Text-only generation: used by ResponseService to produce the agent's
    next line. `conversation` is a list of {"role": "user"|"assistant",
    "text": ...} turns, oldest first, ending on the participant's latest
    utterance."""
    return _executor.submit(_generate_text_impl, system_prompt, conversation, max_new_tokens).result()
