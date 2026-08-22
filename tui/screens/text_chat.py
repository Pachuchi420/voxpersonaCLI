from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header, RichLog, Input, Button
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from core.llm import generate_reply
from core.RecordingService import RecordingService
from core.TranscriptionService import TranscriptionService
import numpy as np
import random
import core.llm as llm

BAR_CHARS = " ▁▂▃▄▅▆▇█"
BAR_COUNT = 300
MIC_LEVEL_REFERENCE = 4000  # rms value treated as "full height"

class TextChatScreen(Screen): 
    BINDINGS = [
        Binding('escape', "confirm_leave", show=False)
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

    #input_area {
        height: auto;
        margin: 0 2 1 2;
    }

    #chat_input {
        width: 1fr;
    }

    #voice_button {
        width: auto;
        margin-left: 1;
    }

    #sidebar {
        width: 15%;
        border-right: solid $accent;
        padding: 1 2;
    }

    #chat_area {
        width: 1fr;
    }

    #status {
        color: $accent;
        content-align: center middle;
        height: 1;
        padding: 0 ;
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
"""





    def compose(self) -> ComposeResult: 
        yield Header()
        with Horizontal():
             with Vertical(id="sidebar"): 
                  yield Static("hello world", id="sidebar_content")
             with Vertical(id="chat_area"): 
                yield RichLog(id="transcript", wrap=True, markup=True)
                yield Static("", id="pending_reply")
                yield Static("", id="mic_level", markup=True)
                with Horizontal(id='input_area'):
                    yield Input(placeholder="Type a message...", id="chat_input")
                    yield Button('Start', id='voice_button')
                yield Static("", id="status")


    def ensure_model(self) -> None:
        status = self.query_one("#status", Static)
        self.app.call_from_thread(status.update, f"Checking model {llm.model_name}...")
        llm.ensure_model_pulled(llm.model_name)
        self.app.call_from_thread(status.update, "")

    def on_mount(self) -> None: 
        self.title = 'voxPersona'
        self.sub_title = 'New text chat'
        self.leave_armed = False
        self.conversation_history = []
        self.thinking_dots = 0
        self.thinking_timer = None
        self.is_recording = False
        self.bar_weights = [random.uniform(0.4, 1.0) for _ in range(BAR_COUNT)]
        self.transcription_service = TranscriptionService()
        self.recording_service = RecordingService()
        self.query_one("#chat_input", Input).focus()
        self.run_worker(self.ensure_model, thread=True)



    def action_confirm_leave(self) -> None: 
        status = self.query_one("#status", Static)
        if self.leave_armed: 
                self.app.pop_screen()
        else : 
            self.leave_armed = True
            status.update("Press Esc again to exit to the main menu")


    def handle_user_command(self, user_text: str) -> bool:
         commands = {
              "/clear" : self.command_clear_text,
              "/help" : self.command_help
         }

         handler = commands.get(user_text.strip())

         if handler is None: 
              return False

         handler()
         return True

         

    async def on_input_submitted(self, event: Input.Submitted) -> None: 
        # Resets leave armed if we type something
        self.leave_armed = False 
        self.query_one("#status", Static).update("")

        # Actual transcript logic
        transcript = self.query_one("#transcript", RichLog)
        user_text = event.value
        event.input.value = ""

        if not user_text.strip(): 
             return 

        if self.handle_user_command(user_text):
             return


        
        transcript.write(f"[bold #00d9ff]You:[/] {user_text}")
        self.run_worker(self.wait_for_answer(user_text))

    async def wait_for_answer(self, user_text) -> None:
        self.conversation_history.append({"role": "user", "content": user_text})

        reply_text = await generate_reply(self.conversation_history)

        self.conversation_history.append({"role": "assistant", "content": reply_text})

        transcript = self.query_one("#transcript", RichLog)
        transcript.write(f"[bold #0077b6]Agent:[/] {reply_text}")



    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'voice_button': 
             self.toggle_recording()


    async def transcribe_and_send(self, pcm_bytes: bytes) -> None:
        session = self.transcription_service.start_session()
        text = await session.finish(pcm_bytes)
        text = text.strip()

        if not text:
            return

        transcript = self.query_one("#transcript", RichLog)
        transcript.write(f"[bold #00d9ff]You (voice):[/] {text}")

        self.run_worker(self.wait_for_answer(text))

    def toggle_recording(self): 
         self.is_recording = not self.is_recording
         voice_button = self.query_one("#voice_button", Button)

         if self.is_recording:
            voice_button.label = "⏹"
            self.recording_service.start(on_chunk=self.on_pcm_chunk)
         else:
            voice_button.label = "▶"
            pcm_bytes = self.recording_service.stop()
            self.run_worker(self.transcribe_and_send(pcm_bytes))


    def update_waveform(self, level: float) -> None:
        bars = "".join(
            BAR_CHARS[min(8, int(level * weight * 8))] for weight in self.bar_weights
        )
        self.query_one("#mic_level", Static).update(bars)

    def on_pcm_chunk(self, chunk: bytes) -> None:
        samples = np.frombuffer(chunk, dtype="int16")
        rms = np.sqrt(np.mean(samples.astype("float32") ** 2))
        level = min(1.0, rms / MIC_LEVEL_REFERENCE)
        self.app.call_from_thread(self.update_waveform, level)


    # User commands 
    def command_clear_text(self) -> None: 
        transcript = self.query_one("#transcript", RichLog)
        transcript.clear()
        self.conversation_history = []

    def command_help(self) -> None:
        transcript = self.query_one("#transcript", RichLog)
        transcript.write("[bold #00d9ff]Available commands:[/] /clear, /help")  


        
  