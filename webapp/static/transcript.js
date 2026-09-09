import { renderMarkdown } from "./markdown.js";

const NEAR_BOTTOM_PX = 40;

// KaTeX (see index.html) loads as plain <script> tags, not an ES module,
// so it attaches itself to window rather than being import-able here --
// guarded since it's still loading/attaching during the very first
// render or two.
function renderMath(el) {
  window.renderMathInElement?.(el, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
}

// Shared message-list rendering for both chat modes. A streaming reply is
// mounted once, in its final resting place in the list, and updated in
// place as tokens arrive -- there's no separate "pending" element that
// later gets swapped for a permanent one.
export class Transcript {
  constructor(container) {
    this.container = container;
  }

  clear() {
    this.container.innerHTML = "";
  }

  isNearBottom() {
    const el = this.container;
    return el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
  }

  scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  addMessage(role, label, text) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.innerHTML = `<div class="label">${label}</div><div class="content">${renderMarkdown(text)}</div>`;
    renderMath(el);
    this.container.appendChild(el);
    this.scrollToBottom();
    return el;
  }

  addThinking(label) {
    const el = document.createElement("div");
    el.className = "msg assistant thinking";
    el.innerHTML = `<div class="label">${label}</div><div class="content thinking-dots"><span>.</span><span>.</span><span>.</span></div>`;
    this.container.appendChild(el);
    this.scrollToBottom();
    return el;
  }

  updateMessage(el, role, label, text) {
    el.className = `msg ${role}`;
    el.innerHTML = `<div class="label">${label}</div><div class="content">${renderMarkdown(text)}</div>`;
    renderMath(el);
  }

  loadHistory(messages) {
    this.clear();
    for (const msg of messages) {
      if (msg.role === "user") {
        this.addMessage("user", "You", msg.content);
      } else if (msg.role === "assistant") {
        this.addMessage("assistant", "Agent", msg.content);
      }
    }
  }
}
