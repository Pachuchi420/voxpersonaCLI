"""Minimal Tk tool for testing just Gemma 4 E2B's transcription quality,
decoupled from BackService.py's full pipeline (no VAD, no TTS, no response
generation, no websocket). Click Record, speak, click Stop -- the captured
clip is transcribed via GemmaService.transcribe_audio() and shown below.
"""
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class GemmaVoiceTestApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Gemma Voice Test")
        root.geometry("480x360")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self.status_var, foreground="gray").pack(anchor="w", padx=8, pady=(8, 0))

        self.output = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=16)
        self.output.pack(fill="both", expand=True, padx=8, pady=8)

        self.record_btn = ttk.Button(root, text="Record", command=self.toggle_record)
        self.record_btn.pack(padx=8, pady=(0, 8))

        self._recording = False
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def toggle_record(self):
        if self._recording:
            self._stop_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self):
        self._chunks = []
        self._recording = True
        self.record_btn.config(text="Stop")
        self.status_var.set("Recording...")
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=self._audio_callback
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())

    def _stop_and_transcribe(self):
        self._recording = False
        self.record_btn.config(text="Record", state="disabled")
        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._chunks:
            self.status_var.set("Idle")
            self.record_btn.config(state="normal")
            return

        audio_i16 = np.concatenate(self._chunks)
        audio_f32 = audio_i16.astype(np.float32).flatten() / 32768.0
        self.status_var.set(f"Transcribing {len(audio_f32) / SAMPLE_RATE:.1f}s of audio (first call loads the model, can take a while)...")
        threading.Thread(target=self._transcribe, args=(audio_f32,), daemon=True).start()

    def _transcribe(self, audio_f32: np.ndarray):
        try:
            from GemmaService import transcribe_audio
            text = transcribe_audio(audio_f32, SAMPLE_RATE)
        except Exception as e:
            text = f"[error: {e}]"
        self.root.after(0, self._show_result, text)

    def _show_result(self, text: str):
        self.output.configure(state="normal")
        self.output.insert(tk.END, (text or "(empty)") + "\n\n")
        self.output.configure(state="disabled")
        self.output.see(tk.END)
        self.status_var.set("Idle")
        self.record_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    GemmaVoiceTestApp(root)
    root.mainloop()
