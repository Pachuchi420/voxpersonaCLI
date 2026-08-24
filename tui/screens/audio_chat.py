import asyncio
import io
import random
import re

import numpy as np
import sounddevice as sd
import soundfile as sf
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Header, RichLog, Static

from core.llm import generate_reply_stream
import core.llm as llm
from core.RecordingService import RecordingService
from core.TranscriptionService import TranscriptionService
import core.TranscriptionService as transcription
from core.VADService import VADService
from core.DialogueService import DialogueService

BAR_CHARS = " ▁▂▃▄▅▆▇█"
BAR_COUNT = 300
MIC_LEVEL_REFERENCE = 4000  # rms value treated as "full height"

# Replies here go through TTS -- markdown syntax and emoji get read aloud
# literally (kokoro has no idea "*" means emphasis, it just says "asterisk"),
# so this asks the model for plain, speakable text instead.
SYSTEM_PROMPT = (
    "You are a helpful voice assistant having a spoken conversation. Never "
    "use markdown formatting -- no asterisks, bullet points, headers, or "
    "code blocks -- and never use emoji. Your replies are converted to "
    "speech and read aloud, so write only plain, speakable text."
)

# Sentence-ending punctuation followed by whitespace -- used to flush
# complete sentences to TTS as they arrive in the stream, rather than
# waiting for the whole reply (see finish_utterance).
SENTENCE_END_RE = re.compile(r"[.!?]+\s+")


class AudioChatScreen(Screen):
    """Hands-free voice conversation: continuously listens (VAD-driven turn
    detection, same as core/BackService_ws.py's loop), transcribes each
    utterance, gets a reply from core/llm.py (Ollama), and speaks it back --
    unlike TextChatScreen's push-to-talk voice button, there's no button
    here, and unlike text mode, replies are spoken aloud, not just logged.
    """

    BINDINGS = [
        Binding("escape", "confirm_leave", show=False)
    ]

    CSS = """
    #transcript {
        border: none;
        padding: 1 2;
        width: 100%;
        scrollbar-color: $accent;
        scrollbar-background: $surface;
        background: transparent;
        scrollbar-size: 1 1;
    }

    #live_transcript {
        color: #e0f7ff;
        text-style: italic;
        padding: 0 2;
        height: auto;
    }

    #pending_reply {
        color: #0077b6;
        text-style: bold;
        padding: 0 2;
        height: auto;
    }

    #mic_level {
        height: 1;
        padding: 0 2;
        color: #4caf50;
    }

    #status {
        color: $accent;
        text-style: bold;
        content-align: center middle;
        height: 1;
        padding: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield RichLog(id="transcript", wrap=True, markup=True)
            yield Static("", id="live_transcript")
            yield Static("", id="pending_reply")
            yield Static("", id="mic_level", markup=True)
            yield Static("", id="status")

    def on_mount(self) -> None:
        self.title = "voxPersona"
        self.sub_title = "Audio mode"
        self.leave_armed = False
        self.conversation_history: list[dict] = []
        self.bar_weights = [random.uniform(0.4, 1.0) for _ in range(BAR_COUNT)]

        self.vad_service = VADService()
        self.recording_service = RecordingService()
        self.transcription_service = TranscriptionService(backend=transcription.DEFAULT_BACKEND)
        self.dialogue_service: DialogueService | None = None
        self.session = None
        # Raw PCM for the utterance in progress, reconstructed from every
        # partial_ready + the final silence_detected event's audio (each is
        # "only new bytes since the last event", so concatenating all of
        # them for one utterance reconstructs the whole thing) -- used for
        # a full-context one-shot re-transcription once the utterance ends,
        # see finish_utterance's docstring note.
        self._utterance_chunks: list[bytes] = []
        # True from the moment an utterance finishes (silence_detected) all
        # the way through transcription-finalize + generate_reply_stream() +
        # speak() -- audio is still captured but not fed to VAD while busy,
        # so the mic can't (a) hear its own TTS reply and re-trigger a new
        # "user" turn from its own voice (hit while building the equivalent
        # loop in TestChatUI.py on gemmaTesting), or (b) start a second,
        # overlapping turn while the first is still waiting on a reply.
        self.busy = False
        self._output_stream: sd.OutputStream | None = None
        # Tracked so leaving the screen mid-reply can cancel/stop everything
        # in flight instead of leaving audio playing or a request running
        # in the background after the screen's gone.
        self._current_worker = None
        self._synth_task: asyncio.Task | None = None
        self._player_task: asyncio.Task | None = None

        self.run_worker(self.ensure_model, thread=True)
        self.run_worker(self.ensure_tts, thread=True)
        self.run_worker(self.ensure_output_stream, thread=True)
        self.recording_service.start(on_chunk=self.on_pcm_chunk)

    def ensure_model(self) -> None:
        status = self.query_one("#status", Static)
        self.app.call_from_thread(status.update, f"Checking model {llm.model_name}...")
        llm.ensure_model_pulled(llm.model_name)
        self.app.call_from_thread(status.update, "Listening...")

    def ensure_tts(self) -> None:
        self.dialogue_service = DialogueService(engine="kokoro")

    def ensure_output_stream(self) -> None:
        # One persistent stream for the screen's whole lifetime, written to
        # incrementally per sentence in _play_queue, instead of calling
        # sd.play() (which opens/tears down its own stream) once per
        # sentence -- repeated stream open/close was the likely source of
        # the audible clicks/crackle at sentence boundaries. kokoro (see
        # DialogueService) always outputs mono float32 at 24000 Hz, so a
        # single fixed-format stream covers every clip this screen plays.
        self._output_stream = sd.OutputStream(
            samplerate=24000, channels=1, dtype="float32", latency="high"
        )
        self._output_stream.start()

    def action_confirm_leave(self) -> None:
        status = self.query_one("#status", Static)
        if self.leave_armed:
            self.recording_service.stop()
            self.stop_audio()
            self.app.pop_screen()
        else:
            self.leave_armed = True
            status.update("Press Esc again to exit to the main menu")

    def stop_audio(self) -> None:
        """Cancels any in-flight reply/synthesis/playback -- called on exit
        so leaving mid-reply doesn't leave TTS still talking (or an Ollama
        request still running) after the screen's gone."""
        if self._current_worker is not None:
            self._current_worker.cancel()
        if self._synth_task is not None:
            self._synth_task.cancel()
        if self._player_task is not None:
            self._player_task.cancel()
        if self._output_stream is not None:
            self._output_stream.abort()  # discards buffered audio immediately, unlike .stop()
            self._output_stream.close()
            self._output_stream = None

    def update_waveform(self, level: float) -> None:
        bars = "".join(
            BAR_CHARS[min(8, int(level * weight * 8))] for weight in self.bar_weights
        )
        self.query_one("#mic_level", Static).update(bars)

    def on_pcm_chunk(self, chunk: bytes) -> None:
        # Skipped entirely while busy (not just the VAD call below) -- every
        # call_from_thread() here is a cross-thread hop competing for the GIL
        # with sd.play()'s own output callback, right when it needs the most
        # consistent scheduling to avoid audible stutter during playback.
        if self.busy:
            return

        samples = np.frombuffer(chunk, dtype="int16")
        rms = np.sqrt(np.mean(samples.astype("float32") ** 2))
        level = min(1.0, rms / MIC_LEVEL_REFERENCE)
        self.app.call_from_thread(self.update_waveform, level)

        for event in self.vad_service.process_pcm(chunk):
            self.app.call_from_thread(self.dispatch_vad_event, event)

    def dispatch_vad_event(self, event: dict) -> None:
        if event["type"] == "partial_ready":
            self._utterance_chunks.append(event["audio"])
            if self.session is None:
                self.session = self.transcription_service.start_session()
            self.run_worker(self.feed_partial(event["audio"]))
        elif event["type"] == "silence_detected":
            self._utterance_chunks.append(event["audio"])
            # Set here, synchronously, rather than inside finish_utterance --
            # on_pcm_chunk checks self.busy before even calling process_pcm(),
            # so this has to be true before the *next* audio chunk arrives,
            # not just before finish_utterance's own awaits.
            self.busy = True
            self.query_one("#status", Static).update("Transcribing...")
            chunks, self._utterance_chunks = self._utterance_chunks, []
            self._current_worker = self.run_worker(self.finish_utterance(b"".join(chunks)))

    async def feed_partial(self, audio_bytes: bytes) -> None:
        if self.session is None:
            return
        text = await self.session.feed(audio_bytes)
        live = self.query_one("#live_transcript", Static)
        live.update(f"You: {text}" if text else "")

    async def finish_utterance(self, full_audio: bytes) -> None:
        status = self.query_one("#status", Static)
        live = self.query_one("#live_transcript", Static)
        try:
            # Close out the streaming session used only for the live partial
            # display above -- its own result is discarded here in favor of
            # a fresh one-shot pass over the *whole* utterance below, same
            # method TextChatScreen's push-to-talk flow already uses.
            # Streaming feeds parakeet-mlx a limited context window
            # (context_size=(256, 256), see TranscriptionSession) for low
            # latency; batching the complete utterance gives it full
            # context instead, which is measurably more accurate -- worth
            # the extra transcribe call since parakeet-mlx is fast/light.
            if self.session is not None:
                await self.session.finish(b"")
                self.session = None
            live.update("")

            final_session = self.transcription_service.start_session()
            text = (await final_session.finish(full_audio)).strip()
            if not text:
                return

            transcript = self.query_one("#transcript", RichLog)
            transcript.write(f"[bold #00d9ff]You (voice):[/] {text}")
            self.conversation_history.append({"role": "user", "content": text})

            status.update("Thinking...")
            pending = self.query_one("#pending_reply", Static)
            reply_text = ""
            sentence_buffer = ""

            # Two-stage pipeline so speech starts after the *first sentence*
            # instead of the whole reply, and so synthesizing sentence N+1
            # overlaps with sentence N still playing instead of leaving dead
            # air between them: generate_reply_stream() -> sentence_queue
            # (split as complete sentences arrive) -> _synthesize_queue()
            # (TTS per sentence) -> audio_queue -> _play_queue() (plays each
            # in order, only as fast as real-time).
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            audio_queue: asyncio.Queue[tuple | None] = asyncio.Queue()
            self._synth_task = asyncio.create_task(self._synthesize_queue(sentence_queue, audio_queue))
            self._player_task = asyncio.create_task(self._play_queue(audio_queue))

            async for chunk in generate_reply_stream(self.conversation_history, system_prompt=SYSTEM_PROMPT):
                reply_text += chunk
                sentence_buffer += chunk
                pending.update(f"[bold #0077b6]Agent:[/] {reply_text}")

                while (m := SENTENCE_END_RE.search(sentence_buffer)) is not None:
                    sentence, sentence_buffer = sentence_buffer[:m.end()].strip(), sentence_buffer[m.end():]
                    if sentence:
                        await sentence_queue.put(sentence)

            if sentence_buffer.strip():
                await sentence_queue.put(sentence_buffer.strip())
            await sentence_queue.put(None)

            pending.update("")
            self.conversation_history.append({"role": "assistant", "content": reply_text})
            transcript.write(f"[bold #0077b6]Agent:[/] {reply_text}")

            status.update("Speaking...")
            await self._synth_task
            await self._player_task
        finally:
            # Always clears busy/status, even if transcription, Ollama, or
            # TTS raised -- otherwise a single failed turn would permanently
            # deafen the mic instead of just skipping that one reply.
            self._synth_task = None
            self._player_task = None
            self._current_worker = None
            self.busy = False
            status.update("Listening...")

    async def _synthesize_queue(self, sentence_queue: "asyncio.Queue[str | None]", audio_queue: "asyncio.Queue[tuple | None]") -> None:
        """Consumes sentences as finish_utterance flushes them, synthesizes
        each via kokoro, and hands the decoded audio to _play_queue --
        runs concurrently with _play_queue (not awaited per-sentence by it),
        so this can be working on sentence N+1 while that's still playing
        sentence N."""
        while True:
            text = await sentence_queue.get()
            if text is None:
                await audio_queue.put(None)
                return
            if self.dialogue_service is None:
                continue
            audio = await self.dialogue_service.runTTS(text)
            if not audio["audio"]:
                continue
            data, sr = sf.read(io.BytesIO(audio["audio"]), dtype="float32")
            await audio_queue.put((data, sr))

    async def _play_queue(self, audio_queue: "asyncio.Queue[tuple | None]") -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await audio_queue.get()
            if item is None:
                return
            data, sr = item
            if self._output_stream is None or sr != 24000:
                continue
            # .write() blocks until the stream's buffer has room, so it
            # naturally paces playback in real time on its own (no manual
            # sleep needed) -- run off the event loop thread since it's a
            # blocking call, same reasoning as run_in_executor calls elsewhere.
            await loop.run_in_executor(None, self._output_stream.write, data)
