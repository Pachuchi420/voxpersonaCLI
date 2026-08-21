from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header
from textual.widgets.option_list import Option
from textual.containers import Vertical



class TextChatScreen(Screen): 
    def compose(self) -> ComposeResult: 
        yield Header()
        yield

    def on_mount(self) -> None: 
        self.title = 'voxPersona'
        self.sub_title = 'New text chat'


  