from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual.containers import Vertical
import pyfiglet 

font = pyfiglet.Figlet(font='larry3d')
title_art = font.renderText("voxPersona")
DESCRIPTIONS = {
    "text_mode" : "Chat with voxPersona via text input.",
    "audio_mode" : "Chat with voxPersona via audio input.",
    "settings" : "Manage user settings.",
    "exit" : "Quit voxPersona."
}


class MainMenuScreen(Screen): 
    CSS = """
    #title { 
        color: $accent;
        content-align: center middle;
        height: auto;
        background: $surface
    }


    #menu {
        align: center top;
        height: 1fr;
    }

    OptionList {
        width: 30; 
        background: transparent; 
        border: none;
    }

    OptionList > .option-list--option-highlighted {
        color: $background;
        text-style: bold;
        background: $accent;
    }

    """

    def compose(self) -> ComposeResult: 
        yield Static(title_art, id="title")
        yield Static()
        with Vertical(id='menu'): 
            yield OptionList(
                Option("Text Mode", id="text_mode"),
                Option("Audio Mode", id="audio_mode"),
                Option("Settings", id="settings"),
                Option("Exit", id="exit"),
                id="mode_list"
            )
        yield Static(DESCRIPTIONS["text_mode"], id="description")

    def on_mount(self) -> None: 
        self.query_one("#mode_list").focus()

    async def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        description = self.query_one("#description", Static)
        option_id = event.option.id or ""
        description.update(DESCRIPTIONS.get(option_id, ""))


    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None: 
        option_id = event.option_id or ""

        if option_id == "text_mode": 
            from tui.screens.text_chat import TextChatScreen
            self.app.push_screen(TextChatScreen())
        elif option_id == "audio_mode": 
            from tui.screens.audio_chat import AudioChatScreen
            self.app.push_screen(AudioChatScreen())
        elif option_id == "settings": 
                from tui.screens.settings import SettingsScreen
                self.app.push_screen(SettingsScreen())
        elif option_id == "audio_mode": 
                self.app.exit()
