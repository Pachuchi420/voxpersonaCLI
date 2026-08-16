"""Minimal Tk client for manually testing the Gemma transcription/response
pipeline end to end against a running BackService.py, without needing the
Unity front end.

Run BackService.py first (add `2` on its command line to also use Gemma for
ASR: `python BackService.py 2`; the response side is controlled by this
client, hardcoded below to "responseEngine": "gemma"), then run this file.
Speaks are captured from the default mic, streamed to the server, and
replies come back as text + audio the same way a real client would receive
them.
"""
import asyncio
import base64
import io
import json
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

SERVER_URI = "ws://localhost:8765"
SAMPLE_RATE = 16000
BLOCKSIZE = 1600  # 0.1s chunks -- server-side VAD buffers/frames internally


class BackendClient:
    """Owns the websocket connection and mic capture on a dedicated asyncio
    event loop (its own thread, since Tk needs the main thread for its own
    loop). Talks to the UI thread only through `events` (inbound) and the
    thread-safe send_* methods (outbound)."""

    def __init__(self, events: queue.Queue):
        self.events = events
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.muted = True
        self._stream: sd.InputStream | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        self._thread.start()

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main())

    async def _main(self):
        try:
            async with websockets.connect(SERVER_URI, max_size=None) as ws:
                self.ws = ws
                self._start_mic()
                self.events.put({"type": "_connected"})
                await self._send(json.dumps({
                    "type": "start_dialogue",
                    "script": {"id": None, "turns": []},
                    "settings": {
                        "ttsEngine": "kokoro",
                        "ttsVoice": "am_adam",
                        "language": "english",
                        "responseEngine": "gemma",
                    },
                }))
                async for message in ws:
                    if isinstance(message, str):
                        self.events.put(json.loads(message))
        except Exception as e:
            self.events.put({"type": "_error", "message": str(e)})
        finally:
            self._stop_mic()

    async def _send(self, payload: str | bytes):
        if self.ws is not None:
            await self.ws.send(payload)

    def _audio_callback(self, indata, frames, time_info, status):
        if self.muted or self.ws is None or self.loop is None:
            return
        pcm_bytes = indata.tobytes()
        asyncio.run_coroutine_threadsafe(self._send(pcm_bytes), self.loop)

    def _start_mic(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _stop_mic(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def send_control(self, payload: dict):
        if self.loop is not None and self.ws is not None:
            asyncio.run_coroutine_threadsafe(self._send(json.dumps(payload)), self.loop)


class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("VoxPersona Test Chat")
        root.geometry("520x520")

        self.status_var = tk.StringVar(value="Connecting...")
        ttk.Label(root, textvariable=self.status_var, foreground="gray").pack(anchor="w", padx=8, pady=(8, 0))

        ttk.Label(root, text="Live transcription:").pack(anchor="w", padx=8, pady=(8, 0))
        self.live_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.live_var, wraplength=480, foreground="#555").pack(anchor="w", padx=8)

        self.chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=20)
        self.chat.pack(fill="both", expand=True, padx=8, pady=8)
        self.chat.tag_config("you", foreground="#1a5fb4")
        self.chat.tag_config("gemma", foreground="#a51d2d")
        self.chat.tag_config("note", foreground="#999999", font=("TkDefaultFont", 9, "italic"))

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=8, pady=(0, 8))
        self.mute_btn = ttk.Button(controls, text="Mute", command=self.toggle_mute)
        self.mute_btn.pack(side="left")

        self.events: queue.Queue = queue.Queue()
        self.client = BackendClient(self.events)
        self.client.muted = True  # stay muted until "_connected" flips it on
        self._clip_buffer: list[dict] = []
        # Set in _flush_clips when a reply is about to play, so
        # transcription_done doesn't also re-listen immediately -- without
        # this the mic goes live again while the reply is still coming out
        # of the speakers and hears itself, turning every reply into a new
        # "user" turn. Cleared once _resume_after_playback actually re-
        # listens (or immediately, in toggle_mute, if the user overrides it).
        self._played_dialogue_this_turn = False
        self._resume_after_id: str | None = None

        self.client.start()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self._poll_events)

    def toggle_mute(self):
        # A pending auto-resume-after-playback (see _flush_clips) would
        # otherwise fight whatever the user just chose here.
        if self._resume_after_id is not None:
            self.root.after_cancel(self._resume_after_id)
            self._resume_after_id = None
        self._played_dialogue_this_turn = False

        muted = not self.client.muted
        self.client.muted = muted
        if muted:
            self.client.send_control({"type": "stop_listening"})
            self.mute_btn.config(text="Unmute")
            self.status_var.set("Muted")
        else:
            self.client.send_control({"type": "listen", "is_last": False})
            self.mute_btn.config(text="Mute")
            self.status_var.set("Listening...")

    def _append_chat(self, text: str, tag: str):
        self.chat.configure(state="normal")
        self.chat.insert(tk.END, text + "\n", (tag,))
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def _handle_event(self, event: dict):
        etype = event.get("type")

        if etype == "_connected":
            self.status_var.set("Listening...")
            self.client.muted = False
            self.mute_btn.config(text="Mute")
            # Unmuting only starts the mic *streaming to* the server -- the
            # server itself ignores incoming audio bytes until it gets an
            # explicit "listen" (is_listening starts False in BackService.py's
            # handle()), so without this the mic runs but nothing ever reaches
            # VAD.
            self.client.send_control({"type": "listen", "is_last": False})

        elif etype == "_error":
            self.status_var.set(f"Error: {event['message']}")
            messagebox.showerror("Connection error", event["message"])

        elif etype == "transcription_partial":
            self.live_var.set(event.get("text", ""))

        elif etype == "transcription_done":
            text = self.live_var.get()
            if text:
                self._append_chat(f"You: {text}", "you")
            self.live_var.set("")
            # dialogue_end (if any) is always received before transcription_done
            # for this turn (server sends the reply clips first), so
            # _played_dialogue_this_turn is already set by the time we get here
            # -- _resume_after_playback re-listens once the reply finishes
            # playing instead of us doing it immediately below.
            if not self._played_dialogue_this_turn and not self.client.muted:
                self.client.send_control({"type": "listen", "is_last": False})

        elif etype == "dialogue":
            self._clip_buffer.append(event)

        elif etype == "dialogue_end":
            self._flush_clips()

        elif etype == "backchannel":
            self._append_chat("[backchannel]", "note")
            self._play_base64(event.get("audio"))

        elif etype == "repeat_requested":
            self._append_chat("[repeat requested]", "note")
            self._play_base64(event.get("audio"))
            self.live_var.set("")
            if not self.client.muted:
                self.client.send_control({"type": "listen", "is_last": False})

    def _flush_clips(self):
        clips = sorted(self._clip_buffer, key=lambda c: c.get("index", 0))
        self._clip_buffer = []
        if not clips:
            return

        texts = [c.get("text", "") for c in clips if c.get("text")]
        if texts:
            self._append_chat(f"Gemma: {' '.join(texts)}", "gemma")

        total_seconds = sum(self._play_base64(clip.get("audio")) for clip in clips)
        if total_seconds > 0:
            # Mute now (mic would otherwise stay live through the reply) and
            # hold off re-listening until playback actually finishes -- see
            # _played_dialogue_this_turn's comment in __init__.
            self._played_dialogue_this_turn = True
            self.client.muted = True
            self.status_var.set("Playing reply...")
            delay_ms = int(total_seconds * 1000) + 300  # pad for output-device latency
            self._resume_after_id = self.root.after(delay_ms, self._resume_after_playback)

    def _resume_after_playback(self):
        self._resume_after_id = None
        self._played_dialogue_this_turn = False
        self.client.muted = False
        self.status_var.set("Listening...")
        self.client.send_control({"type": "listen", "is_last": False})

    def _play_base64(self, audio_b64: str | None) -> float:
        """Plays the clip; returns its duration in seconds (0.0 if nothing played)."""
        if not audio_b64:
            return 0.0
        try:
            data, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)), dtype="float32")
            sd.play(data, sr)
            return len(data) / sr
        except Exception as e:
            self._append_chat(f"[playback error: {e}]", "note")
            return 0.0

    def on_close(self):
        if self._resume_after_id is not None:
            self.root.after_cancel(self._resume_after_id)
        self.client.muted = True
        if self.client.loop is not None and self.client.ws is not None:
            asyncio.run_coroutine_threadsafe(self.client.ws.close(), self.client.loop)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()
