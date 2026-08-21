from textual.app import App
from tui.screens.main_menu import MainMenuScreen
from textual.theme import Theme



voxBlue_theme = Theme(
    name="voxBlue",
    primary="#00d9ff",
    secondary="#0077b6",
    accent="#00d9ff",
    background="#0a0e14",
    surface="#0d1b2a",
    panel="#0d1b2a",
    foreground="#e0f7ff",
    dark=True,
)

class ConvoTUI(App):
    def on_mount(self) -> None: 
        self.register_theme(voxBlue_theme)
        self.theme = 'voxBlue'
        self.push_screen(MainMenuScreen())





    

if __name__ == "__main__":
    ConvoTUI().run()

    

