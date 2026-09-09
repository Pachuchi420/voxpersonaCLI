const SAMPLE_RATE = 16000;

// Push-to-talk dictation for the text-mode input box -- separate from
// AudioChat (audio.js): no VAD, no auto-reply, no TTS, just live partial
// transcription streamed into the textarea while the mic button is
// toggled on, via /ws/transcribe. You review/edit the result and send it
// like any typed message; nothing is sent to the model until you do.
export class Dictation {
  constructor({ button, textarea }) {
    this.button = button;
    this.textarea = textarea;
    this.active = false;
    this.ws = null;
    this.audioContext = null;
    this.workletNode = null;
    this.mediaStream = null;
    this.baseText = "";

    this.button.addEventListener("click", () => this.toggle());
  }

  toggle() {
    if (this.active) {
      this.stop();
    } else {
      this.start();
    }
  }

  async start() {
    if (this.active) return;
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.error("getUserMedia failed:", err);
      return;
    }

    this.active = true;
    this.button.classList.add("recording");
    this.baseText = this.textarea.value.trim() ? `${this.textarea.value.trim()} ` : "";

    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/transcribe`);
    this.ws.onopen = () => this.ws.send(JSON.stringify({ type: "start" }));
    this.ws.onmessage = (event) => this._onMessage(JSON.parse(event.data));

    this.audioContext = new AudioContext();
    await this.audioContext.audioWorklet.addModule("/pcm-worklet.js");
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-worklet", {
      processorOptions: { targetRate: SAMPLE_RATE },
    });
    this.workletNode.port.onmessage = (event) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(event.data);
      }
    };
    source.connect(this.workletNode);
  }

  async stop() {
    if (!this.active) return;
    this.active = false;
    this.button.classList.remove("recording");

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
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
    // The WebSocket itself closes once the server's "final" message
    // arrives (see _onMessage) so that last transcript isn't dropped.
  }

  _setText(text) {
    this.textarea.value = this.baseText + text;
    // Programmatic .value writes don't fire "input" -- dispatch it
    // manually so app.js's auto-resize handler still runs.
    this.textarea.dispatchEvent(new Event("input"));
  }

  _onMessage(msg) {
    if (msg.type === "partial") {
      this._setText(msg.text);
    } else if (msg.type === "final") {
      this._setText(msg.text);
      this.textarea.focus();
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    }
  }
}
