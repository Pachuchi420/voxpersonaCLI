"""Facial emotion recognition from webcam frames, captured periodically
while Audio mode is open (see the webcam toggle in audio.js) -- not VAD-
triggered like voice emotion, since facial expression isn't utterance-
bound the way speech is.

Two-stage pipeline, since FER models are trained on already-cropped
faces, not full frames:
  1. Face landmarking -- MediaPipe's Face Landmarker
     (core/models/face_landmarker.task, ~3.7MB) locates the face (it
     includes its own detector internally, no separate model needed for
     that) and returns all 478 3D mesh landmarks. Used two ways here:
     their bounding extent crops the face for step 2, and the full set
     is sent to the client so it can draw a detailed eyes/eyebrows/lips/
     face-oval overlay (see FACEMESH_* in audio.js) instead of just a
     box -- what was actually asked for, not a generic "6 dots" readout.
  2. Emotion classification -- dima806/facial_emotions_image_detection
     (https://huggingface.co/dima806/facial_emotions_image_detection),
     an 85.8M-param ViT fine-tuned for facial expression (7 classes:
     angry/disgust/fear/happy/neutral/sad/surprise), via `transformers`.
     Runs CPU-only for the same reason as EmotionService's voice model:
     this would otherwise compete with the LLM reply and Kokoro's TTS
     for the GPU.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "dima806/facial_emotions_image_detection"
FACE_LANDMARKER_MODEL = Path(__file__).parent / "models" / "face_landmarker.task"
# Google's own hosted copy -- fetched on first use rather than committed
# to the repo (~3.7MB), same as every other model here (Ollama pulls,
# the HF hub cache, funasr/modelscope's cache): none of those are
# checked in either, so this shouldn't be the one binary blob that is.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_face_landmarker_model() -> None:
    if FACE_LANDMARKER_MODEL.exists():
        return
    import urllib.request
    FACE_LANDMARKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FACE_LANDMARKER_MODEL.with_suffix(".task.part")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp_path)
    tmp_path.rename(FACE_LANDMARKER_MODEL)

LABEL_NAMES = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
    "neutral": "Neutral", "sad": "Sad", "surprise": "Surprise",
}

# Crop margin around the landmarks' tight bounding box -- FER models are
# typically trained on crops that include a bit of forehead/chin/ears,
# not just the exact face extent.
CROP_MARGIN = 0.25

_landmarker: Any = None
_classifier: Any = None


def _load() -> None:
    global _landmarker, _classifier
    if _landmarker is not None:
        return

    _ensure_face_landmarker_model()

    from mediapipe.tasks.python import vision, BaseOptions
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    _landmarker = vision.FaceLandmarker.create_from_options(options)

    from transformers import pipeline
    _classifier = pipeline("image-classification", model=MODEL_NAME, device="cpu")


class FacialEmotionService:
    def __init__(self):
        # One dedicated worker thread, same reasoning as the other
        # services here: serializes access across concurrent calls.
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=_load)
        # Pays the model-load cost now instead of on the first classify()
        # call, so it doesn't show up as latency on someone's first frame.
        self._executor.submit(lambda: None).result()

    async def classify(self, image_bytes: bytes) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._classify_sync, image_bytes)

    def _classify_sync(self, image_bytes: bytes) -> dict:
        import cv2
        import mediapipe as mp
        from PIL import Image

        empty = {"scores": [], "box": None, "landmarks": []}

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return empty
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _landmarker.detect(mp_image)
        if not result.face_landmarks:
            return empty

        # num_faces=1 above, so there's at most one.
        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        box_w, box_h = max_x - min_x, max_y - min_y

        x0 = max(0, int((min_x - box_w * CROP_MARGIN) * w))
        y0 = max(0, int((min_y - box_h * CROP_MARGIN) * h))
        x1 = min(w, int((max_x + box_w * CROP_MARGIN) * w))
        y1 = min(h, int((max_y + box_h * CROP_MARGIN) * h))
        face = rgb[y0:y1, x0:x1]
        if face.size == 0:
            return empty

        raw = _classifier(Image.fromarray(face), top_k=None)
        scores = [
            {"code": r["label"], "label": LABEL_NAMES.get(r["label"], r["label"]), "score": round(r["score"], 4)}
            for r in sorted(raw, key=lambda r: -r["score"])
        ]

        # Already normalized (0-1 fractions of the frame) by MediaPipe --
        # kept that way rather than converted to pixels, same reasoning
        # as the old bounding-box-only version: the client maps these
        # onto its own preview at whatever size *that's* currently
        # displayed, without either side needing to track exact capture
        # dimensions.
        box_norm = {"x": min_x, "y": min_y, "width": box_w, "height": box_h}
        landmarks_norm = [{"x": lm.x, "y": lm.y} for lm in landmarks]

        return {"scores": scores, "box": box_norm, "landmarks": landmarks_norm}
