import { renderMarkdown } from "./markdown.js";
import { ICON_PENCIL, ICON_REFRESH } from "./icons.js";

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
//
// Renders a *path* through the server's branching chat tree (see
// core/ChatHistory.py): each node carries its id and its full sibling
// list. A node with more than one sibling gets a "< 2/3 >" navigator
// (always visible -- it's informational, not just an action) alongside
// its label; user messages get a hover-revealed edit action, assistant
// messages a hover-revealed regenerate action (matching the sidebar's
// "icons that only appear on hover so they don't obstruct the text"
// pattern). Editing/regenerating/switching branches are all handled by
// the caller (TextChat/AudioChat) via the onEdit/onRegenerate/
// onSwitchBranch callbacks -- this class only renders and reports clicks.
export class Transcript {
  constructor(container, { onEdit, onRegenerate, onSwitchBranch } = {}) {
    this.container = container;
    this.onEdit = onEdit;
    this.onRegenerate = onRegenerate;
    this.onSwitchBranch = onSwitchBranch;
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

  // Full re-render from a canonical server path (initial load via REST,
  // or a "sync" message after any edit/regenerate/switch_branch/turn).
  renderPath(path) {
    this.clear();
    for (const node of path) {
      this.container.appendChild(this._buildNode(node));
    }
    this.scrollToBottom();
  }

  // Drops every rendered message after `el`. Used when starting a new
  // turn from a point that isn't the end of the list (editing an older
  // message, regenerating an older reply) -- the old continuation is
  // about to be replaced by a new branch, so it shouldn't linger on
  // screen until the next sync gets around to removing it.
  cutAfter(el) {
    while (el.nextSibling) {
      this.container.removeChild(el.nextSibling);
    }
  }

  _buildNode(node) {
    const el = document.createElement("div");
    el.className = `msg ${node.role}`;
    el.dataset.nodeId = node.id;
    this._paint(el, node);
    return el;
  }

  _paint(el, node) {
    const label = node.role === "user" ? "You" : "Agent";
    el.className = `msg ${node.role}`;
    el.innerHTML = `
      <div class="msg-head">
        <span class="label">${label}</span>
        ${node.siblings.length > 1 ? this._navHtml(node) : ""}
      </div>
      <div class="msg-actions">
        ${node.role === "user"
          ? `<button type="button" class="row-btn edit-btn" title="Edit">${ICON_PENCIL}</button>`
          : `<button type="button" class="row-btn regen-btn" title="Regenerate">${ICON_REFRESH}</button>`}
      </div>
      <div class="content"></div>
    `;
    // Markdown source goes through the DOM, not a template string, so a
    // message containing literal <, >, or & isn't at risk of being torn
    // apart before renderMarkdown ever sees it -- renderMarkdown escapes
    // internally, but the *raw* content shouldn't pass through innerHTML
    // unescaped even transiently.
    el.querySelector(".content").innerHTML = renderMarkdown(node.content);
    renderMath(el);
    this._wireNode(el, node);
  }

  _navHtml(node) {
    const index = node.siblings.indexOf(node.id);
    const atStart = index <= 0;
    const atEnd = index >= node.siblings.length - 1;
    return `
      <span class="msg-nav">
        <button type="button" class="nav-btn" data-dir="prev" ${atStart ? "disabled" : ""}>&lsaquo;</button>
        <span class="nav-count">${index + 1}/${node.siblings.length}</span>
        <button type="button" class="nav-btn" data-dir="next" ${atEnd ? "disabled" : ""}>&rsaquo;</button>
      </span>
    `;
  }

  _wireNode(el, node) {
    el.querySelector('[data-dir="prev"]')?.addEventListener("click", () => {
      const index = node.siblings.indexOf(node.id);
      if (index > 0) this.onSwitchBranch?.(node.siblings[index - 1]);
    });
    el.querySelector('[data-dir="next"]')?.addEventListener("click", () => {
      const index = node.siblings.indexOf(node.id);
      if (index < node.siblings.length - 1) this.onSwitchBranch?.(node.siblings[index + 1]);
    });
    el.querySelector(".edit-btn")?.addEventListener("click", () => this._enterEdit(el, node));
    el.querySelector(".regen-btn")?.addEventListener("click", () => {
      // Regenerating replaces this node's own content (and drops
      // whatever followed it, if it wasn't the last node in the path)
      // with a new branch -- shown immediately rather than left stale
      // until the next sync.
      this.cutAfter(el);
      this._paintThinking(el);
      this.scrollToBottom();
      this.onRegenerate?.(node.id, el);
    });
  }

  _enterEdit(el, node) {
    el.innerHTML = `
      <div class="msg-head"><span class="label">You</span></div>
      <textarea class="edit-textarea chat-textarea"></textarea>
      <div class="edit-actions">
        <button type="button" class="edit-cancel">Cancel</button>
        <button type="button" class="edit-save send-btn">Save &amp; submit</button>
      </div>
    `;
    const textarea = el.querySelector(".edit-textarea");
    textarea.value = node.content;
    const autosize = () => {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    };
    textarea.addEventListener("input", autosize);
    autosize();
    textarea.focus();

    el.querySelector(".edit-cancel").addEventListener("click", () => this._paint(el, node));
    el.querySelector(".edit-save").addEventListener("click", () => {
      const text = textarea.value.trim();
      if (!text) return;
      // Same reasoning as regenerate above: this edit starts a new
      // branch, so whatever used to follow this message is stale now.
      this.cutAfter(el);
      this._paintLive(el, "user", text);
      this.onEdit?.(node.id, text);
    });
  }

  // -- The one currently-streaming reply, before it's a real node in any
  // synced path yet (see the header comment for why there's no separate
  // "pending" widget -- this mounts in its final spot and is replaced,
  // in place, by the canonical version once the next sync arrives). No
  // edit/regenerate/nav on it -- those need a real node id.

  _paintThinking(el) {
    el.className = "msg assistant thinking";
    el.innerHTML = `<div class="msg-head"><span class="label">Agent</span></div><div class="content thinking-dots"><span>.</span><span>.</span><span>.</span></div>`;
  }

  addThinking() {
    const el = document.createElement("div");
    this._paintThinking(el);
    this.container.appendChild(el);
    this.scrollToBottom();
    return el;
  }

  updateStreaming(el, text) {
    el.className = "msg assistant";
    el.innerHTML = `<div class="msg-head"><span class="label">Agent</span></div><div class="content"></div>`;
    el.querySelector(".content").innerHTML = renderMarkdown(text);
    renderMath(el);
  }

  // Shared by addLive and by the optimistic edit-save repaint above --
  // a plain, non-interactive bubble for content that isn't a real tree
  // node yet (no id, so no edit/nav actions are possible on it).
  _paintLive(el, role, text) {
    const label = role === "user" ? "You" : "Agent";
    el.className = `msg ${role}`;
    el.innerHTML = `<div class="msg-head"><span class="label">${label}</span></div><div class="content"></div>`;
    el.querySelector(".content").innerHTML = renderMarkdown(text);
    renderMath(el);
  }

  // Audio mode's live mic transcript, and a plain user/assistant turn
  // in text mode: painted in immediately, before it's ever a real node
  // in the tree (that only happens once the next "sync" arrives).
  addLive(role, text) {
    const el = document.createElement("div");
    this._paintLive(el, role, text);
    this.container.appendChild(el);
    this.scrollToBottom();
    return el;
  }

  updateLive(el, text) {
    el.querySelector(".content").innerHTML = renderMarkdown(text);
    renderMath(el);
  }
}
