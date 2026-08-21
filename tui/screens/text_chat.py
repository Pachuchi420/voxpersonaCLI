from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header, RichLog, Input
from textual.binding import Binding
from textual.containers import Vertical, Horizontal



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
    """





    def compose(self) -> ComposeResult: 
        yield Header()
        with Horizontal():
             with Vertical(id="sidebar"): 
                  yield Static("hello world", id="sidebar_content")
             with Vertical(id="chat_area"): 
                yield RichLog(id="transcript", wrap=True, markup=True)
                yield Input(placeholder="Type a message...", id="chat_input")
                yield Static("", id="status")

    def on_mount(self) -> None: 
        self.title = 'voxPersona'
        self.sub_title = 'New text chat'
        self.leave_armed = False
        self.query_one("#chat_input", Input).focus()

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
        reply_text = f"You said: {user_text}"
        transcript.write(f"[bold #0077b6]Agent:[/] {reply_text}")


        
  