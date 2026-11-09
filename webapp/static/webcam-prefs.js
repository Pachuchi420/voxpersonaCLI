// Client-side webcam/facial-overlay preferences -- all pure rendering/
// capture-cadence knobs, none of them gate the webcam/camera/
// classification themselves (see audio.js/server.py: the camera and
// detection always run). localStorage rather than /api/settings since
// nothing server-side needs to know about any of this.
const KEYS = {
  overlay: "voxpersona.faceOverlay",
  intervalMs: "voxpersona.webcamIntervalMs",
  maxDimension: "voxpersona.webcamMaxDimension",
};

// Also the sliders' own min/max/step and starting positions in the
// Settings tab (see settings.js) -- these are the values the feature
// shipped with, kept as the floor/defaults rather than picked arbitrarily.
export const DEFAULTS = {
  intervalMs: 2000,
  maxDimension: 480,
};

export function getFaceOverlayEnabled() {
  return localStorage.getItem(KEYS.overlay) === "true";
}

export function setFaceOverlayEnabled(enabled) {
  localStorage.setItem(KEYS.overlay, enabled ? "true" : "false");
}

// How often a frame is captured and sent for classification. Lower
// values track expression more closely but cost more CPU, competing
// with the LLM reply/TTS during generation.
export function getWebcamIntervalMs() {
  const value = parseInt(localStorage.getItem(KEYS.intervalMs), 10);
  return Number.isFinite(value) ? value : DEFAULTS.intervalMs;
}

export function setWebcamIntervalMs(ms) {
  localStorage.setItem(KEYS.intervalMs, String(ms));
}

// Longest edge a captured frame is downscaled to before sending. Lower
// values mean a smaller payload and a faster forward pass through the
// landmarker/classifier, at some cost to how small a face can still be
// reliably detected.
export function getWebcamMaxDimension() {
  const value = parseInt(localStorage.getItem(KEYS.maxDimension), 10);
  return Number.isFinite(value) ? value : DEFAULTS.maxDimension;
}

export function setWebcamMaxDimension(px) {
  localStorage.setItem(KEYS.maxDimension, String(px));
}
