from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, Header, OptionList
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.containers import Vertical
import core.llm as llm
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
            self._toggle_model()
        elif option_id == "back":
            self.app.pop_screen()

    def _toggle_model(self) -> None:
        llm.model_name = "gemma4:e2b" if llm.model_name == "gemma4:e4b" else "gemma4:e4b"

        option_list = self.query_one("#settings_list", OptionList)
        option_list.remove_option("model_toggle")
        option_list.add_option(Option(f"Model: {llm.model_name}", id="model_toggle"))

        self.notify(f"Switched to {llm.model_name}")