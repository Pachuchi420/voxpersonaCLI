import { THEMES, getTheme, applyTheme } from "./theme.js";
import { api } from "./api.js";
import {
  getFaceOverlayEnabled, setFaceOverlayEnabled,
  getWebcamIntervalMs, setWebcamIntervalMs,
  getWebcamMaxDimension, setWebcamMaxDimension,
} from "./webcam-prefs.js";

const BACKEND_LABELS = { auto: "Auto", mlx: "MLX", sherpa: "Sherpa-ONNX", gemma: "Gemma" };
const ON_OFF = [false, true];
const ON_OFF_LABELS = { false: "Off", true: "On" };

export class Settings {
  constructor({
    themeContainer, modelContainer, pullForm, pullInput, pullStatus, transcriptionContainer,
    faceOverlayContainer, webcamIntervalSlider, webcamIntervalValue, webcamSizeSlider, webcamSizeValue,
  }) {
    this.themeContainer = themeContainer;
    this.modelContainer = modelContainer;
    this.pullForm = pullForm;
    this.pullInput = pullInput;
    this.pullStatus = pullStatus;
    this.transcriptionContainer = transcriptionContainer;
    this.faceOverlayContainer = faceOverlayContainer;
    this.webcamIntervalSlider = webcamIntervalSlider;
    this.webcamIntervalValue = webcamIntervalValue;
    this.webcamSizeSlider = webcamSizeSlider;
    this.webcamSizeValue = webcamSizeValue;

    this.pullForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const tag = this.pullInput.value.trim();
      if (tag) this._pullModel(tag);
    });

    // Live-applied as you drag, same as a volume slider -- takes effect
    // the next time the webcam's toggled on (see audio.js), not
    // mid-stream, same reasoning as the other settings here.
    this.webcamIntervalSlider.addEventListener("input", () => {
      setWebcamIntervalMs(Number(this.webcamIntervalSlider.value));
      this._updateWebcamSliderLabels();
    });
    this.webcamSizeSlider.addEventListener("input", () => {
      setWebcamMaxDimension(Number(this.webcamSizeSlider.value));
      this._updateWebcamSliderLabels();
    });
  }

  async open() {
    this._renderThemes();
    this._renderFaceOverlay();
    this._renderWebcamSliders();
    const settings = await api.getSettings();
    this._renderOptions(this.modelContainer, settings.models, settings.model, (model) => this._setModel(model), (id) => id);
    this._renderOptions(
      this.transcriptionContainer,
      settings.transcription_backends,
      settings.transcription_backend,
      (backend) => this._setTranscriptionBackend(backend),
      (id) => BACKEND_LABELS[id] || id
    );
  }

  // Pure client-side preference (see webcam-prefs.js), not part of
  // /api/settings -- nothing server-side needs to know whether the
  // overlay is drawn, only whether a frame was sent at all (which stays
  // under the Webcam button's own control, unaffected by this).
  _renderFaceOverlay() {
    this._renderOptions(
      this.faceOverlayContainer,
      ON_OFF,
      getFaceOverlayEnabled(),
      (enabled) => {
        setFaceOverlayEnabled(enabled);
        this._renderFaceOverlay();
      },
      (id) => ON_OFF_LABELS[id]
    );
  }

  _renderWebcamSliders() {
    this.webcamIntervalSlider.value = String(getWebcamIntervalMs());
    this.webcamSizeSlider.value = String(getWebcamMaxDimension());
    this._updateWebcamSliderLabels();
  }

  _updateWebcamSliderLabels() {
    this.webcamIntervalValue.textContent = `${(getWebcamIntervalMs() / 1000).toFixed(1)}s`;
    this.webcamSizeValue.textContent = `${getWebcamMaxDimension()}px`;
  }

  _renderThemes() {
    const current = getTheme();
    this.themeContainer.innerHTML = "";
    for (const theme of THEMES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "theme-swatch" + (theme.id === current ? " active" : "");
      btn.style.setProperty("--swatch-bg", theme.bg);
      btn.style.setProperty("--swatch-accent", theme.accent);
      btn.innerHTML = `<span class="swatch-circle"></span><span>${theme.label}</span>`;
      btn.addEventListener("click", () => {
        applyTheme(theme.id);
        this._renderThemes();
      });
      this.themeContainer.appendChild(btn);
    }
  }

  _renderOptions(container, options, current, onSelect, labelFor) {
    container.innerHTML = "";
    for (const id of options) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "settings-option" + (id === current ? " active" : "");
      btn.textContent = labelFor(id);
      btn.addEventListener("click", () => onSelect(id));
      container.appendChild(btn);
    }
  }

  async _setModel(model) {
    this.modelContainer.querySelectorAll(".settings-option").forEach((btn) => (btn.disabled = true));
    try {
      await fetch("/api/settings/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
    } finally {
      await this.open();
    }
  }

  // "Plug and play": type any Ollama tag (e.g. one found on ollama.com's
  // library) and it gets pulled, switched to, and shows up in the picker
  // from then on -- no code change needed to add a new model.
  async _pullModel(tag) {
    this.pullInput.disabled = true;
    this.pullForm.querySelector("button").disabled = true;
    this.pullStatus.textContent = `Pulling ${tag}… this can take a while for a large model.`;
    try {
      const response = await fetch("/api/settings/models/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: tag }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `${response.status} ${response.statusText}`);
      }
      this.pullInput.value = "";
      this.pullStatus.textContent = "";
    } catch (err) {
      this.pullStatus.textContent = String(err.message || err);
    } finally {
      this.pullInput.disabled = false;
      this.pullForm.querySelector("button").disabled = false;
      await this.open();
    }
  }

  async _setTranscriptionBackend(backend) {
    this.transcriptionContainer.querySelectorAll(".settings-option").forEach((btn) => (btn.disabled = true));
    try {
      await fetch("/api/settings/transcription-backend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend }),
      });
    } finally {
      await this.open();
    }
  }
}
