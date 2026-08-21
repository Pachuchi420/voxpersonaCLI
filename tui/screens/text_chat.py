from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header, RichLog, Input
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from core.llm import generate_reply
import core.llm as llm

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

        #chat_input {
            margin: 0 2 1 2;
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
            padding: 0 2;
        }

        #pending_reply {
            color: #0077b6;
            text-style: bold;
            padding: 0 2;
            height: auto;
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
                yield Input(placeholder="Type a message...", id="chat_input")
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
        self.query_one("#chat_input", Input).focus()
        self.run_worker(self.ensure_model, thread=True)



    def action_confirm_leave(self) -> None: 
        status = self.query_one("#status", Static)
        if self.leave_armed: 
                self.app.pop_screen()
        else : 
            self.leave_armed = True
            status.update("Press Esc again to exit to the main menu")


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
        transcript.write(f"[bold #00d9ff]You:[/] {user_text}")
        reply_text = await generate_reply(user_text)
        transcript.write(f"[bold #0077b6]Agent:[/] {reply_text}")


        
  