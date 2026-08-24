from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header, OptionList
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.containers import Vertical
import core.llm as llm
from core.llm import determine_system
import core.TranscriptionService as transcription
import pyfiglet

font = pyfiglet.Figlet(font='larry3d')
title_art = font.renderText("Settings")


class SettingsScreen(Screen):

    BINDINGS = [
        Binding('escape', "confirm_leave", show=False)
    ]

    CSS = """
        #title {
            color: $accent;
            content-align: center middle;
            height: auto;
            background: $surface
        }

        #status {
            color: $accent;
            content-align: center middle;
            height: 1;
            padding: 0 2;
        }

        OptionList {
            width: 40;
            background: transparent;
            border: none;
        }

        OptionList > .option-list--option-highlighted {
            color: $background;
            text-style: bold;
            background: $accent;
        }

        #menu {
            align: center top;
            height: 1fr;
        }

    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            yield Static(title_art, id="title")
            with Vertical(id="menu"):
                yield OptionList(
                    Option(f"Model: {llm.model_name}", id="model_toggle"),
                    Option(f"Transcription: {transcription.DEFAULT_BACKEND or 'mlx'}", id="transcription_toggle"),
                    Option("Back", id="back"),
                    id="settings_list"
                )
            yield Static("", id="status")

    def on_mount(self) -> None:
        self.title = 'voxPersona'
        self.sub_title = 'Settings'
        self.leave_armed = False
        self.query_one("#settings_list", OptionList).focus()

    def action_confirm_leave(self) -> None:
        status = self.query_one("#status", Static)
        if self.leave_armed:
            self.app.pop_screen()
        else:
            self.leave_armed = True
            status.update("Press Esc again to exit to the main menu")

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option_id or ""

        if option_id == "model_toggle":
            self.toggle_model()
        elif option_id == "transcription_toggle":
            self.toggle_transcription()
        elif option_id == "back":
            self.app.pop_screen()

    def toggle_model(self) -> None:
        suffix = "-mlx" if determine_system() else ""


        llm.model_name = f"gemma4:e2b{suffix}" if llm.model_name == f"gemma4:e4b{suffix}" else f"gemma4:e4b{suffix}"

        option_list = self.query_one("#settings_list", OptionList)
        # replace_option_prompt updates the option in place -- remove_option
        # + add_option (the previous approach) re-appends at the *end* of
        # the list instead, so the row jumps below "Back" every time you
        # toggle it and the menu visibly scrambles itself.
        option_list.replace_option_prompt("model_toggle", f"Model: {llm.model_name} {suffix}")

        self.notify(f"Switched to {llm.model_name}")

    def toggle_transcription(self) -> None:
        # None means "auto" (mlx on Darwin) -- see TranscriptionService.py.
        transcription.DEFAULT_BACKEND = "gemma" if transcription.DEFAULT_BACKEND != "gemma" else None
        label = transcription.DEFAULT_BACKEND or "mlx"

        option_list = self.query_one("#settings_list", OptionList)
        option_list.replace_option_prompt("transcription_toggle", f"Transcription: {label}")

        self.notify(f"Transcription switched to {label}")