from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header
from textual.widgets.option_list import Option
from textual.containers import Vertical



class AudioChatScreen(Screen): 
    def compose(self) -> ComposeResult: 
        yield Static("this is the audio screen")
  