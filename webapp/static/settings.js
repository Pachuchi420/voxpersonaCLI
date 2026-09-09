import { THEMES, getTheme, applyTheme } from "./theme.js";

const MODEL_LABELS = { e2b: "E2B (faster)", e4b: "E4B (larger, slower)" };
const BACKEND_LABELS = { auto: "Auto", mlx: "MLX", sherpa: "Sherpa-ONNX", gemma: "Gemma" };

export class Settings {
  constructor({ themeContainer, modelContainer, transcriptionContainer }) {
    this.themeContainer = themeContainer;
    this.modelContainer = modelContainer;
    this.transcriptionContainer = transcriptionContainer;
  }

  async open() {
    this._renderThemes();
    const settings = await fetch("/api/settings").then((r) => r.json());
    this._renderOptions(this.modelContainer, settings.model_sizes, this._currentSize(settings.model), (size) =>
      this._setModel(size)
    );
    this._renderOptions(
      this.transcriptionContainer,
      settings.transcription_backends,
      settings.transcription_backend,
      (backend) => this._setTranscriptionBackend(backend),
      (id) => BACKEND_LABELS[id] || id
    );
  }

  _currentSize(modelTag) {
    // modelTag looks like "gemma4:e2b-mlx" or "gemma4:e2b" -- pull just
    // the size back out to match against model_sizes ("e2b"/"e4b").
    const match = modelTag.match(/gemma4:(e\d+b)/);
    return match ? match[1] : modelTag;
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

  _renderOptions(container, options, current, onSelect, labelFor = (id) => MODEL_LABELS[id] || id) {
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

  async _setModel(size) {
    this.modelContainer.querySelectorAll(".settings-option").forEach((btn) => (btn.disabled = true));
    try {
      await fetch("/api/settings/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size }),
      });
    } finally {
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
