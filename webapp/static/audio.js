import { Transcript } from "./transcript.js";
import { FACE_MESH_GROUPS } from "./facemesh-data.js";
import { getFaceOverlayEnabled, getWebcamIntervalMs, getWebcamMaxDimension } from "./webcam-prefs.js";

const SAMPLE_RATE = 16000;
const MIC_LEVEL_REFERENCE = 4000; // rms value treated as "full height"
// Capture interval and frame size are tunable from Settings (see
// webcam-prefs.js) -- read live rather than hardcoded here, with the
// values this feature shipped with as the fallback defaults.

export class AudioChat {
  constructor({
    transcriptEl, titleEl, statusEl, micBarEl, muteBtn, sendBtn, emotionEl,
    webcamBtn, webcamWindowEl, webcamHandleEl, webcamPreviewEl, webcamOverlayEl, webcamCanvasEl,
    facialPanelEl, facialEl,
  }) {
    this.transcript = new Transcript(transcriptEl, {
      onEdit: (nodeId, text) => this._edit(nodeId, text),
      onRegenerate: (nodeId) => this._regenerate(nodeId),
      onSwitchBranch: (nodeId) => this._switchBranch(nodeId),
    });
    this.titleEl = titleEl;
    this.statusEl = statusEl;
    this.micBarEl = micBarEl;
    this.muteBtn = muteBtn;
    this.sendBtn = sendBtn;
    this.emotionEl = emotionEl;
    this.webcamBtn = webcamBtn;
    this.webcamWindowEl = webcamWindowEl;
    this.webcamPreviewEl = webcamPreviewEl;
    this.webcamOverlayEl = webcamOverlayEl;
    this.webcamCanvasEl = webcamCanvasEl;
    this.facialPanelEl = facialPanelEl;
    this.facialEl = facialEl;
    this.onTitle = null;
    this.onEmptyStop = null;

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
    this.pendingKind = null;
    this.pendingText = null;

    this.webcamStream = null;
    this.webcamInterval = null;
    // Set while waiting on a frame's "facial_emotion" response -- lets
    // _captureFrame skip a tick rather than piling up more frames behind
    // a slow one (classification competing with the LLM/TTS for CPU
    // during generation can genuinely slow it down; firing on a fixed
    // timer regardless just builds a backlog that takes a while to
    // drain, which reads as the camera "hanging").
    this.facialFrameInFlight = false;
    this.facialFrameTimeout = null;

    this.muteBtn.addEventListener("click", () => this.toggleMute());
    this.webcamBtn.addEventListener("click", () => this.toggleWebcam());
    this._wireWebcamDrag(webcamHandleEl);
    // Keeps the overlay canvas's pixel size matched to the stage's
    // *rendered* size -- both when the native CSS `resize` handle
    // changes it, and once at startup -- so the box/keypoint drawing
    // lines up with the video at any size rather than stretching.
    new ResizeObserver(() => this._syncOverlaySize()).observe(this.webcamWindowEl);
  }

  async open(chatId, initialData) {
    await this.close();
    this.chatId = chatId;
    if (initialData) {
      this.transcript.renderPath(initialData.path || []);
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
    this._stopWebcam();
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
    if (this.emotionEl) this.emotionEl.textContent = "Listening...";
    // Webcam deliberately doesn't carry over to a freshly-opened chat --
    // same reasoning as Mute resetting above, and it avoids a camera
    // silently staying on across an unrelated chat switch.
    this._stopWebcam();
    this._setGenerating(false);
  }

  // Reused for both spoken and typed turns: `busy` (driven by the
  // server's "status" messages -- Transcribing/Thinking/Speaking vs.
  // Listening) already tracks exactly "is a turn in flight," which is
  // also exactly when the send button should read Stop instead of Send.
  // Mirrored onto `generating` too, so app.js's wireInputRow can check
  // one consistent property name across both TextChat and AudioChat.
  _setGenerating(value) {
    this.generating = value;
    this.sendBtn.textContent = value ? "Stop" : "Send";
    this.sendBtn.classList.toggle("stop", value);
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
      // Doubles as a manual "I'm done talking" push-to-finalize -- in a
      // loud environment the server's VAD may never see a clean enough
      // silence to end the utterance on its own (and muting stops audio
      // from reaching it at all from this point on, so it never would).
      // No-op server-side if nothing was actually mid-utterance.
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "finalize" }));
      }
    } else if (!this.busy) {
      this.statusEl.textContent = "Listening...";
    }
  }

  async toggleWebcam() {
    if (this.webcamStream) {
      this._stopWebcam();
      return;
    }
    try {
      this.webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (err) {
      console.error("getUserMedia (video) failed:", err);
      if (this.facialEl) this.facialEl.textContent = "Camera unavailable";
      if (this.facialPanelEl) this.facialPanelEl.hidden = false;
      return;
    }
    this.webcamPreviewEl.srcObject = this.webcamStream;
    this.webcamWindowEl.hidden = false;
    this._positionWebcamWindow();
    this.webcamBtn.classList.add("active");
    if (this.facialPanelEl) this.facialPanelEl.hidden = false;
    if (this.facialEl) this.facialEl.textContent = "Starting camera...";
    this._syncOverlaySize();
    // Read once per toggle-on rather than live -- same reasoning as
    // other settings here (transcription backend, model): takes effect
    // on the next turn/toggle rather than needing cross-component
    // wiring to restart an interval mid-stream for a debug slider.
    this.webcamInterval = setInterval(() => this._captureFrame(), getWebcamIntervalMs());
  }

  // Pops out anchored to the bottom-right by default each time the
  // webcam's turned on (rather than wherever it was last dragged to).
  // Anchored just above the input row (found via the Webcam button's
  // own container) rather than the raw viewport edge, so the default
  // placement doesn't start out covering the input box/Send button.
  _positionWebcamWindow() {
    const margin = 16;
    const rect = this.webcamWindowEl.getBoundingClientRect();
    const inputRow = this.webcamBtn.closest(".input-row");
    const floor = inputRow ? inputRow.getBoundingClientRect().top : window.innerHeight;
    this.webcamWindowEl.style.left = `${window.innerWidth - rect.width - margin}px`;
    this.webcamWindowEl.style.top = `${floor - rect.height - margin}px`;
  }

  _stopWebcam() {
    if (this.webcamInterval) {
      clearInterval(this.webcamInterval);
      this.webcamInterval = null;
    }
    if (this.facialFrameTimeout) {
      clearTimeout(this.facialFrameTimeout);
      this.facialFrameTimeout = null;
    }
    this.facialFrameInFlight = false;
    if (this.webcamStream) {
      this.webcamStream.getTracks().forEach((track) => track.stop());
      this.webcamStream = null;
    }
    this.webcamPreviewEl.srcObject = null;
    this.webcamWindowEl.hidden = true;
    this.webcamBtn.classList.remove("active");
    if (this.facialPanelEl) this.facialPanelEl.hidden = true;
    this._clearFacialOverlay();
  }

  // Plain pointer-drag on the handle bar, updating the window's
  // top/left directly -- the native CSS `resize` on #webcam-window (see
  // style.css) already covers resizing, so this only needs to handle
  // moving it.
  _wireWebcamDrag(handleEl) {
    let startX, startY, startTop, startLeft;
    const onMove = (event) => {
      const dx = event.clientX - startX;
      const dy = event.clientY - startY;
      this.webcamWindowEl.style.left = `${startLeft + dx}px`;
      this.webcamWindowEl.style.top = `${startTop + dy}px`;
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
    };
    handleEl.addEventListener("pointerdown", (event) => {
      const rect = this.webcamWindowEl.getBoundingClientRect();
      startX = event.clientX;
      startY = event.clientY;
      startTop = rect.top;
      startLeft = rect.left;
      document.addEventListener("pointermove", onMove);
      document.addEventListener("pointerup", onUp);
    });
  }

  _syncOverlaySize() {
    const stage = this.webcamOverlayEl.parentElement;
    this.webcamOverlayEl.width = stage.clientWidth;
    this.webcamOverlayEl.height = stage.clientHeight;
  }

  // box/landmarks come back from the server in the *unmirrored* frame's
  // coordinate space (fractions of width/height -- see
  // FacialEmotionService.classify), but #webcam-preview is displayed
  // mirrored (see style.css) for a natural "looking in a mirror" feel.
  // Mirroring each point's x here, rather than the overlay canvas
  // itself, keeps the drawn label text readable instead of backwards.
  _drawFacialOverlay(box, landmarks, topLabel) {
    const canvas = this.webcamOverlayEl;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Pure client-side rendering preference (see webcam-prefs.js)
    // -- checked here, not by skipping the call at the call site, so
    // this stays the one place that decides whether anything gets
    // drawn regardless of who calls it.
    if (!getFaceOverlayEnabled() || !box || !landmarks.length) return;

    const toScreen = (lm) => [(1 - lm.x) * canvas.width, lm.y * canvas.height];

    // Per-feature-group lines (lips/eyes/eyebrows/face-oval/nose/irises
    // -- see facemesh-data.js) rather than a generic box+dots readout,
    // so it actually reads as "here's your mouth, here's your eyebrows."
    ctx.lineWidth = 1.5;
    for (const { color, pairs } of FACE_MESH_GROUPS) {
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (const [a, b] of pairs) {
        const [ax, ay] = toScreen(landmarks[a]);
        const [bx, by] = toScreen(landmarks[b]);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }

    if (topLabel) {
      const x = (1 - box.x - box.width) * canvas.width;
      const y = box.y * canvas.height;
      const h = box.height * canvas.height;
      ctx.font = "600 13px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textBaseline = "bottom";
      const textY = y > 18 ? y - 4 : y + h + 18;
      // A dark backing strip behind the text -- readable over both a
      // bright and a dark patch of video, which plain colored text on
      // its own often isn't.
      const metrics = ctx.measureText(topLabel);
      ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
      ctx.fillRect(x - 2, textY - 14, metrics.width + 8, 18);
      ctx.fillStyle = "#4ade80";
      ctx.fillText(topLabel, x + 2, textY);
    }
  }

  _clearFacialOverlay() {
    const canvas = this.webcamOverlayEl;
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }

  _captureFrame() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    // Skip this tick rather than sending another frame on top of one
    // still being processed -- see the constructor comment on
    // facialFrameInFlight. The timer keeps running on its normal
    // schedule either way; this just lets ticks pass through as no-ops
    // when the server's still catching up.
    if (this.facialFrameInFlight) return;
    const video = this.webcamPreviewEl;
    if (!video.videoWidth || !video.videoHeight) return; // not ready yet

    const maxDimension = getWebcamMaxDimension();
    const scale = Math.min(1, maxDimension / Math.max(video.videoWidth, video.videoHeight));
    const canvas = this.webcamCanvasEl;
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
    this.facialFrameInFlight = true;
    // Scales with the capture interval (floor 8s) rather than a fixed
    // timeout -- if someone's set a long interval, a short fixed
    // timeout would fire before the next tick even would have, making
    // the safety net pointless.
    this.facialFrameTimeout = setTimeout(() => {
      this.facialFrameInFlight = false;
    }, Math.max(8000, getWebcamIntervalMs() * 4));
    this.ws.send(JSON.stringify({ type: "video_frame", image: base64 }));
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "status":
        this.busy = msg.text !== "Listening...";
        this._setGenerating(this.busy);
        if (!this.muted) this.statusEl.textContent = msg.text;
        break;

      case "transcript_partial":
        if (!this.liveUserEl) {
          this.liveUserEl = this.transcript.addLive("user", msg.text);
        } else {
          this.transcript.updateLive(this.liveUserEl, msg.text);
          this.transcript.scrollToBottom();
        }
        break;

      case "transcript_final":
        // Not rendered as a permanent bubble here -- the live one is left
        // in place (updated to the final text) only until the next
        // "sync" replaces it with the canonical, node-id-bearing version.
        if (this.liveUserEl) {
          this.transcript.updateLive(this.liveUserEl, msg.text);
        } else {
          this.transcript.addLive("user", msg.text);
        }
        this.liveUserEl = null;
        this.replyEl = this.transcript.addThinking();
        this.replyText = "";
        break;

      case "reply_delta": {
        if (!this.replyEl) {
          this.replyEl = this.transcript.addThinking();
          this.replyText = "";
        }
        this.replyText += msg.text;
        const wasNearBottom = this.transcript.isNearBottom();
        this.transcript.updateStreaming(this.replyEl, this.replyText);
        if (wasNearBottom) this.transcript.scrollToBottom();
        break;
      }

      case "reply_done":
      case "stopped":
        // Same reasoning as text.js -- stopped before any content
        // streamed back means the server pruned the childless user
        // message it was replying to, and the trailing "sync" will make
        // it disappear from view. Only restorable for a typed turn
        // (there's no input-field text to give back for a spoken one).
        if (msg.type === "stopped" && !this.replyText.trim() && this.pendingKind === "message") {
          this.onEmptyStop?.(this.pendingText);
        }
        this.replyEl = null;
        this.replyText = "";
        this.pendingKind = null;
        this.pendingText = null;
        break;

      case "audio":
        this._enqueueAudio(msg.audio);
        break;

      case "title":
        this.titleEl.textContent = msg.title;
        this.onTitle?.(msg.title);
        break;

      case "sync":
        this.transcript.renderPath(msg.path);
        break;

      case "emotion":
        // Debug-stage display: the full score distribution (not just the
        // top label), so it's easy to eyeball whether a reading looks
        // believable against what was actually just said -- also logged
        // server-side (see detect_emotion in server.py) for the same
        // reason. Not wired into the LLM's behavior yet.
        console.log("[emotion]", msg.scores, `${msg.elapsed_ms}ms`);
        if (this.emotionEl) {
          const lines = msg.scores.map((s) => `${s.label} ${Math.round(s.score * 100)}%`);
          lines.push(`(${msg.elapsed_ms}ms)`);
          this.emotionEl.textContent = lines.join("\n");
        }
        break;

      case "facial_emotion": {
        // Always acked now, even when no face was found or
        // classification errored (see detect_face in server.py) --
        // lets _captureFrame know the previous frame is done (see
        // facialFrameInFlight) so it can send the next one, and makes
        // "no face this frame" an explicit, immediate signal rather
        // than something inferred from a staleness timeout.
        if (this.facialFrameTimeout) clearTimeout(this.facialFrameTimeout);
        this.facialFrameInFlight = false;

        console.log("[facial_emotion]", msg.scores, `${msg.elapsed_ms}ms`);
        if (this.facialEl) {
          const lines = msg.scores.length
            ? msg.scores.map((s) => `${s.label} ${Math.round(s.score * 100)}%`)
            : ["No face detected"];
          lines.push(`(${msg.elapsed_ms}ms)`);
          this.facialEl.textContent = lines.join("\n");
        }
        const top = msg.scores[0];
        this._drawFacialOverlay(msg.box, msg.landmarks, top ? `${top.label} ${Math.round(top.score * 100)}%` : null);
        break;
      }
    }
  }

  // Typed input while in audio mode -- e.g. the mic isn't cooperating,
  // or you just don't want to talk. Goes through the same reply+TTS
  // pipeline as a spoken turn server-side (see server.py's "message"
  // control type), so the reply is still read aloud.
  send(text) {
    if (!text.trim() || !this.ws || this.ws.readyState !== WebSocket.OPEN || this.busy) return;
    this.transcript.addLive("user", text);
    this.replyEl = this.transcript.addThinking();
    this.replyText = "";
    this.pendingKind = "message";
    this.pendingText = text;
    this.ws.send(JSON.stringify({ type: "message", text }));
  }

  // Interrupts whatever turn is currently in flight -- spoken or typed,
  // still generating the reply or already speaking it.
  stop() {
    if (!this.busy || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "stop" }));
  }

  _edit(nodeId, text) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.busy) return;
    this.replyEl = this.transcript.addThinking();
    this.replyText = "";
    this.pendingKind = "edit";
    this.pendingText = null;
    this.ws.send(JSON.stringify({ type: "edit", node_id: nodeId, text }));
  }

  _regenerate(nodeId, el) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.busy) return;
    this.replyEl = el || this.transcript.addThinking();
    this.replyText = "";
    this.pendingKind = "regenerate";
    this.pendingText = null;
    this.ws.send(JSON.stringify({ type: "regenerate", node_id: nodeId }));
  }

  _switchBranch(nodeId) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "switch_branch", node_id: nodeId }));
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
