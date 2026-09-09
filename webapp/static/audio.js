import { Transcript } from "./transcript.js";

const SAMPLE_RATE = 16000;
const MIC_LEVEL_REFERENCE = 4000; // rms value treated as "full height"

export class AudioChat {
  constructor({ transcriptEl, titleEl, statusEl, micBarEl, muteBtn }) {
    this.transcript = new Transcript(transcriptEl);
    this.titleEl = titleEl;
    this.statusEl = statusEl;
    this.micBarEl = micBarEl;
    this.muteBtn = muteBtn;
    this.onTitle = null;

    this.chatId = null;
    this.ws = null;
    this.audioContext = null;
    this.workletNode = null;
    this.mediaStream = null;

    this.muted = false;
    this.busy = false;
    this.playQueue = [];
    this.playing = false;

    this.liveUserEl = null;
    this.replyEl = null;
    this.replyText = "";

    this.muteBtn.addEventListener("click", () => this.toggleMute());
  }

  async open(chatId, initialData) {
    await this.close();
    this.chatId = chatId;
    if (initialData) {
      this.transcript.loadHistory(initialData.messages || []);
      this.titleEl.textContent = initialData.title || "";
    } else {
      this.transcript.clear();
      this.titleEl.textContent = "";
    }
    this._resetState();
    this._connect();
    try {
      await this._startMic();
    } catch (err) {
      this.statusEl.textContent = "Microphone unavailable";
      console.error("getUserMedia failed:", err);
    }
  }

  async close() {
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
    if (this.ws) {
      this.ws.onmessage = null;
      this.ws.close();
      this.ws = null;
    }
    this.playQueue = [];
    this.playing = false;
  }

  _resetState() {
    this.liveUserEl = null;
    this.replyEl = null;
    this.replyText = "";
    this.busy = false;
    this.muted = false;
    this.muteBtn.textContent = "Mute";
    this.muteBtn.classList.remove("muted");
    this.micBarEl.style.width = "0%";
  }

  _connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/audio/${this.chatId}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        this._onMessage(JSON.parse(event.data));
      }
    };
  }

  async _startMic() {
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioContext = new AudioContext();
    await this.audioContext.audioWorklet.addModule("/pcm-worklet.js");
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-worklet", {
      processorOptions: { targetRate: SAMPLE_RATE },
    });
    this.workletNode.port.onmessage = (event) => this._onPcmChunk(event.data);
    // Deliberately not connected to audioContext.destination -- we don't
    // want to hear our own mic.
    source.connect(this.workletNode);
  }

  _onPcmChunk(buffer) {
    const pcm16 = new Int16Array(buffer);
    let sumSquares = 0;
    for (let i = 0; i < pcm16.length; i++) sumSquares += pcm16[i] * pcm16[i];
    const rms = Math.sqrt(sumSquares / pcm16.length);
    const level = Math.min(1, rms / MIC_LEVEL_REFERENCE);

    // Skipped while muted or busy (server is transcribing/thinking/
    // speaking) -- mirrors the old TUI's self.busy gate: without it the
    // mic would hear the assistant's own TTS reply and could trigger a
    // new turn from its own voice.
    if (this.muted || this.busy) {
      this.micBarEl.style.width = "0%";
      return;
    }
    this.micBarEl.style.width = `${level * 100}%`;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(buffer);
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    this.muteBtn.textContent = this.muted ? "Unmute" : "Mute";
    this.muteBtn.classList.toggle("muted", this.muted);
    if (this.muted) {
      this.micBarEl.style.width = "0%";
      this.statusEl.textContent = "Muted";
    } else if (!this.busy) {
      this.statusEl.textContent = "Listening...";
    }
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "status":
        this.busy = msg.text !== "Listening...";
        if (!this.muted) this.statusEl.textContent = msg.text;
        break;

      case "transcript_partial":
        if (!this.liveUserEl) {
          this.liveUserEl = this.transcript.addMessage("user", "You", msg.text);
        } else {
          this.transcript.updateMessage(this.liveUserEl, "user", "You", msg.text);
          this.transcript.scrollToBottom();
        }
        break;

      case "transcript_final":
        if (this.liveUserEl) {
          this.transcript.updateMessage(this.liveUserEl, "user", "You", msg.text);
        } else {
          this.transcript.addMessage("user", "You", msg.text);
        }
        this.liveUserEl = null;
        this.replyEl = this.transcript.addThinking("Agent");
        this.replyText = "";
        break;

      case "reply_delta": {
        if (!this.replyEl) {
          this.replyEl = this.transcript.addThinking("Agent");
          this.replyText = "";
        }
        this.replyText += msg.text;
        const wasNearBottom = this.transcript.isNearBottom();
        this.transcript.updateMessage(this.replyEl, "assistant", "Agent", this.replyText);
        if (wasNearBottom) this.transcript.scrollToBottom();
        break;
      }

      case "reply_done":
        this.replyEl = null;
        this.replyText = "";
        break;

      case "audio":
        this._enqueueAudio(msg.audio);
        break;

      case "title":
        this.titleEl.textContent = msg.title;
        this.onTitle?.(msg.title);
        break;
    }
  }

  // Typed input while in audio mode -- e.g. the mic isn't cooperating,
  // or you just don't want to talk. Goes through the same reply+TTS
  // pipeline as a spoken turn server-side (see server.py's "message"
  // control type), so the reply is still read aloud.
  send(text) {
    if (!text.trim() || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.transcript.addMessage("user", "You", text);
    this.replyEl = this.transcript.addThinking("Agent");
    this.replyText = "";
    this.ws.send(JSON.stringify({ type: "message", text }));
  }

  _enqueueAudio(base64) {
    this.playQueue.push(base64);
    this._playNext();
  }

  _playNext() {
    if (this.playing || this.playQueue.length === 0) return;
    this.playing = true;
    const base64 = this.playQueue.shift();
    const audio = new Audio(`data:audio/wav;base64,${base64}`);
    const advance = () => {
      this.playing = false;
      this._playNext();
    };
    audio.onended = advance;
    audio.onerror = advance;
    audio.play().catch(advance);
  }
}
