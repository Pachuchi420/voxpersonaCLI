import { Transcript } from "./transcript.js";

export class TextChat {
  constructor(container, titleEl, sendBtn) {
    this.transcript = new Transcript(container, {
      onEdit: (nodeId, text) => this._edit(nodeId, text),
      onRegenerate: (nodeId) => this._regenerate(nodeId),
      onSwitchBranch: (nodeId) => this._switchBranch(nodeId),
    });
    this.titleEl = titleEl;
    this.sendBtn = sendBtn;
    this.chatId = null;
    this.ws = null;
    this.replyEl = null;
    this.replyText = "";
    this.onTitle = null;
    this.onEmptyStop = null;
    this.generating = false;
    this.pendingKind = null;
    this.pendingText = null;
  }

  async open(chatId, initialData) {
    this.close();
    this.chatId = chatId;
    if (initialData) {
      this.transcript.renderPath(initialData.path || []);
      this.titleEl.textContent = initialData.title || "";
    } else {
      this.transcript.clear();
      this.titleEl.textContent = "";
    }
    this._connect();
  }

  close() {
    if (this.ws) {
      this.ws.onmessage = null;
      this.ws.close();
      this.ws = null;
    }
    this.replyEl = null;
    this.replyText = "";
    this._setGenerating(false);
  }

  _setGenerating(value) {
    this.generating = value;
    this.sendBtn.textContent = value ? "Stop" : "Send";
    this.sendBtn.classList.toggle("stop", value);
  }

  _connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/text/${this.chatId}`);
    this.ws.onmessage = (event) => this._onMessage(JSON.parse(event.data));
  }

  _onMessage(msg) {
    if (msg.type === "delta") {
      if (!this.replyEl) {
        this.replyEl = this.transcript.addThinking();
        this.replyText = "";
      }
      this.replyText += msg.text;
      const wasNearBottom = this.transcript.isNearBottom();
      this.transcript.updateStreaming(this.replyEl, this.replyText);
      if (wasNearBottom) this.transcript.scrollToBottom();
    } else if (msg.type === "title") {
      this.titleEl.textContent = msg.title;
      this.onTitle?.(msg.title);
    } else if (msg.type === "done" || msg.type === "stopped") {
      // Stopped before any content streamed back: the server prunes the
      // now-childless user message it was replying to (see run_generation
      // in server.py) rather than leaving a dangling empty turn, and the
      // "sync" right behind this message will make it vanish from the
      // transcript. Without restoring it here, whatever was typed would
      // just be lost.
      if (msg.type === "stopped" && !this.replyText.trim() && this.pendingKind === "message") {
        this.onEmptyStop?.(this.pendingText);
      }
      this.replyEl = null;
      this.replyText = "";
      this.pendingKind = null;
      this.pendingText = null;
      this._setGenerating(false);
    } else if (msg.type === "sync") {
      // Canonical state after a turn completes (or an edit/regenerate/
      // switch_branch) -- replaces the streamed-in-place reply (which
      // had no real id yet) with the real thing, now with an id and
      // sibling info for the edit/regenerate/nav controls.
      this.transcript.renderPath(msg.path);
    }
  }

  send(text) {
    if (!text.trim() || !this.ws || this.ws.readyState !== WebSocket.OPEN || this.generating) return;
    // Painted immediately -- it has no real id yet, but the user should
    // see their own message the instant it's sent, not once the reply
    // starts arriving. The next "sync" replaces it with the canonical,
    // id-bearing version.
    this.transcript.addLive("user", text);
    this.replyEl = this.transcript.addThinking();
    this.replyText = "";
    this.pendingKind = "message";
    this.pendingText = text;
    this._setGenerating(true);
    this.ws.send(JSON.stringify({ type: "message", text }));
  }

  stop() {
    if (!this.generating || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "stop" }));
  }

  _edit(nodeId, text) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.generating) return;
    this.replyEl = this.transcript.addThinking();
    this.replyText = "";
    this.pendingKind = "edit";
    this.pendingText = null;
    this._setGenerating(true);
    this.ws.send(JSON.stringify({ type: "edit", node_id: nodeId, text }));
  }

  _regenerate(nodeId, el) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.generating) return;
    // transcript.js already repainted `el` in place as the thinking
    // placeholder before calling this, so it's reused as-is here rather
    // than adding a second one.
    this.replyEl = el || this.transcript.addThinking();
    this.replyText = "";
    this.pendingKind = "regenerate";
    this.pendingText = null;
    this._setGenerating(true);
    this.ws.send(JSON.stringify({ type: "regenerate", node_id: nodeId }));
  }

  _switchBranch(nodeId) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "switch_branch", node_id: nodeId }));
  }
}
