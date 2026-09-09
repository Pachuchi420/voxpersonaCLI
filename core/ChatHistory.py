"""File-based chat history, one JSON file per chat, kept separate per mode
(text vs. audio -- see chat_history/text/ and chat_history/audio/) so the
two conversation lists don't mix. Mirrors the app's existing lightweight
file-cache conventions (audio_cache/, backchannel_cache/) rather than
pulling in a database for what's a handful of small JSON files.
"""
import json
import time
import uuid
from pathlib import Path

HISTORY_DIR = Path(__file__).parent.parent / "chat_history"

UNTITLED = "New Chat"


def _mode_dir(mode: str) -> Path:
    d = HISTORY_DIR / mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def list_chats(mode: str) -> list[dict]:
    """Returns [{"id":, "title":, "updated_at":}, ...] sorted newest first.
    Corrupt/unreadable files are skipped rather than raised -- a bad chat
    file shouldn't take down the whole history sidebar."""
    chats = []
    for path in _mode_dir(mode).glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        chats.append({
            "id": data.get("id", path.stem),
            "title": data.get("title") or UNTITLED,
            "updated_at": data.get("updated_at", 0),
        })
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


def load_chat(mode: str, chat_id: str) -> dict:
    path = _mode_dir(mode) / f"{chat_id}.json"
    return json.loads(path.read_text())


def save_chat(mode: str, chat_id: str, title: str, messages: list[dict]) -> None:
    path = _mode_dir(mode) / f"{chat_id}.json"
    data = {
        "id": chat_id,
        "title": title or UNTITLED,
        "updated_at": time.time(),
        "messages": messages,
    }
    path.write_text(json.dumps(data, indent=2))


def rename_chat(mode: str, chat_id: str, new_title: str) -> None:
    data = load_chat(mode, chat_id)
    save_chat(mode, chat_id, new_title.strip() or UNTITLED, data.get("messages", []))


def delete_chat(mode: str, chat_id: str) -> None:
    (_mode_dir(mode) / f"{chat_id}.json").unlink(missing_ok=True)
