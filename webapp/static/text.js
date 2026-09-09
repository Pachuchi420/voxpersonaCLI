import { Transcript } from "./transcript.js";

export class TextChat {
  constructor(container, titleEl) {
    this.transcript = new Transcript(container);
    this.titleEl = titleEl;
    this.chatId = null;
    this.ws = null;
    this.replyEl = null;
    this.replyText = "";
    this.onTitle = null;
  }

  async open(chatId, initialData) {
    this.close();
    this.chatId = chatId;
    if (initialData) {
      this.transcript.loadHistory(initialData.messages || []);
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
  }

  _connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/text/${this.chatId}`);
    this.ws.onmessage = (event) => this._onMessage(JSON.parse(event.data));
  }

  _onMessage(msg) {
    if (msg.type === "delta") {
      this.replyText += msg.text;
      const wasNearBottom = this.transcript.isNearBottom();
      this.transcript.updateMessage(this.replyEl, "assistant", "Agent", this.replyText);
      if (wasNearBottom) this.transcript.scrollToBottom();
    } else if (msg.type === "title") {
      this.titleEl.textContent = msg.title;
      this.onTitle?.(msg.title);
    } else if (msg.type === "done") {
      this.replyEl = null;
      this.replyText = "";
    }
  }

  send(text) {
    if (!text.trim() || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.transcript.addMessage("user", "You", text);
    // Shown immediately so it's actually visible during the latency
    // before the first token arrives -- creating it lazily inside the
    // first "delta" handler meant it was replaced with real text in the
    // same synchronous call and never got a chance to render at all.
    this.replyEl = this.transcript.addThinking("Agent");
    this.replyText = "";
    this.ws.send(JSON.stringify({ type: "message", text }));
  }
}
