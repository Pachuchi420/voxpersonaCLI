"""Voice emotion recognition, layered onto the same VAD-segmented utterance
audio the transcription pipeline already receives (see handle_spoken_utterance
in webapp/server.py) -- no new audio capture needed, just one more inference
call on a buffer that's already in hand.

Uses iic/emotion2vec_plus_base (https://huggingface.co/emotion2vec/emotion2vec_plus_base),
a ~90M-param model purpose-built for speech emotion recognition (9 classes:
angry/disgusted/fearful/happy/neutral/other/sad/surprised/unknown), via
`funasr`. Runs CPU-only, deliberately: detect_emotion in server.py fires
this concurrently with the LLM reply and Kokoro's TTS, which are already
using the GPU, and the model is small enough that CPU inference on a few
seconds of audio is still well under a second (confirmed: ~90ms warm on
this machine).

Labels come back from funasr as "<chinese>/<english>" pairs (or the
literal "<unk>") -- _classify_sync strips that down to just the english
code before mapping to a display name.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

import core.settings as settings

MODEL_NAME = "iic/emotion2vec_plus_base"

LABEL_NAMES = {
    "angry": "Angry", "disgusted": "Disgusted", "fearful": "Fearful",
    "happy": "Happy", "neutral": "Neutral", "other": "Other",
    "sad": "Sad", "surprised": "Surprised", "unk": "Unknown",
}

_model: Any = None


def _load_model() -> None:
    global _model
    if _model is not None:
        return
    from funasr import AutoModel
    _model = AutoModel(model=MODEL_NAME, disable_update=True)


class EmotionService:
    def __init__(self):
        # One dedicated worker thread, same reasoning as
        # TranscriptionService's MLX executor: serializes access across
        # concurrent calls instead of letting them race, even though
        # this model has no real thread-affinity requirement.
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=_load_model)
        # Pays the model-load cost now instead of on the first classify()
        # call, so it doesn't show up as latency on someone's first turn.
        self._executor.submit(lambda: None).result()

    async def classify(self, pcm_bytes: bytes) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._classify_sync, pcm_bytes)

    def _classify_sync(self, pcm_bytes: bytes) -> list[dict]:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        # Too short for the model to say anything meaningful about --
        # avoids wasting a forward pass on e.g. a one-word "ok" that VAD
        # barely caught the tail of.
        if len(audio_f32) < settings.SAMPLE_RATE // 4:
            return []

        raw = _model.generate(audio_f32, granularity="utterance", extract_embedding=False, disable_pbar=True)
        result = raw[0]
        out = []
        for raw_label, score in zip(result["labels"], result["scores"]):
            code = raw_label.split("/")[-1] if "/" in raw_label else raw_label.strip("<>")
            out.append({"code": code, "label": LABEL_NAMES.get(code, code), "score": round(float(score), 4)})
        return sorted(out, key=lambda r: -r["score"])
