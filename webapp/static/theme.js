const STORAGE_KEY = "voxpersona.theme";

export const THEMES = [
  { id: "paper", label: "Paper", bg: "#faf7f2", accent: "#0d7a68" },
  { id: "nightfall", label: "Nightfall", bg: "#0d1117", accent: "#4a9fb0" },
  { id: "slate", label: "Slate", bg: "#191712", accent: "#cf9a52" },
  { id: "grey", label: "Grey", bg: "#1c1c1e", accent: "#8fa4b8" },
];

export function getTheme() {
  return localStorage.getItem(STORAGE_KEY) || THEMES[0].id;
}

export function applyTheme(id) {
  document.documentElement.dataset.theme = id;
  localStorage.setItem(STORAGE_KEY, id);
}

// Applied as early as possible (called from the top of app.js, before
// other setup) so there's no flash of the default theme on load.
applyTheme(getTheme());
