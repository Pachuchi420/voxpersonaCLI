"""Dynamic agent-response generation.

Scripts are normally a fixed sequence of pre-authored turns (see
BackService.py's start_dialogue handler): the agent's lines are exactly what
the script author wrote, played back verbatim through DialogueService's TTS.
ResponseService is an opt-in alternative to that: when a script's settings
set "responseEngine": "gemma", the agent's next line is generated live by
Gemma 4 E2B from the conversation so far (via GemmaService.generate_text)
instead of being read off the script.

The default engine, "scripted", is a no-op wrapper that preserves the old
fixed-script behavior exactly -- BackService.py only calls generate() after
checking enabled(), so nothing about the existing scripted-playback path
changes for scripts that don't opt in.
"""
from typing import Optional

_ENGINES = ("scripted", "gemma")

DEFAULT_PERSONA = (
    "You are a helpful conversational agent taking part in a spoken dialogue "
    "study. Keep replies short, natural, and in-character for a spoken "
    "conversation."
)


class ResponseService:
    def __init__(self, engine: str = "scripted", persona: Optional[str] = None):
        if engine not in _ENGINES:
            raise ValueError(f"Unknown response engine: {engine!r}. Use one of {_ENGINES}")
        self.engine = engine
        self.persona = persona or DEFAULT_PERSONA

    def enabled(self) -> bool:
        """False for "scripted" -- callers should check this before calling
        generate() and fall back to the script's own pre-authored turns
        otherwise."""
        return self.engine == "gemma"

    def generate(self, conversation: list[dict]) -> str:
        """conversation: list of {"role": "user"|"assistant", "text": ...}
        turns, oldest first, ending on the participant's latest utterance.
        Returns the agent's next line. Only valid when enabled() is True."""
        if not self.enabled():
            raise RuntimeError("generate() called on a 'scripted' ResponseService -- check enabled() first")

        from GemmaService import generate_text
        return generate_text(self.persona, conversation)
