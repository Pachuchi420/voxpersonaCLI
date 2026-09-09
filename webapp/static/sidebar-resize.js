const STORAGE_KEY = "voxpersona.sidebarWidth";
const MIN_WIDTH = 180;
const MAX_WIDTH = 440;

export function initSidebarResize(sidebar, handle) {
  const saved = parseInt(localStorage.getItem(STORAGE_KEY) || "", 10);
  if (saved) sidebar.style.width = `${clamp(saved)}px`;

  let dragging = false;

  handle.addEventListener("mousedown", (event) => {
    dragging = true;
    handle.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    // #sidebar is the leftmost element in the layout, so its left edge
    // sits at the viewport's left edge -- clientX is already the width.
    sidebar.style.width = `${clamp(event.clientX)}px`;
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem(STORAGE_KEY, parseInt(sidebar.style.width, 10));
  });
}

function clamp(width) {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));
}
