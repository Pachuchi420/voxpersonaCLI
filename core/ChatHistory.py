"""File-based chat history with branching support: each chat is a TREE of
message nodes, not a flat list, so editing a past message or regenerating
a reply creates a new sibling branch instead of destroying what was there
before -- the old branch stays on disk, just not active, and the UI can
switch back to it (the "< 2/3 >" navigator). Kept separate per mode (text
vs. audio -- see chat_history/text/ and chat_history/audio/) so the two
conversation lists don't mix. Mirrors the app's existing lightweight
file-cache conventions (audio_cache/, backchannel_cache/) rather than
pulling in a database for what's a handful of small JSON files.

Tree shape (also what save_chat/load_chat persist, alongside id/title/
updated_at): {"children": [id, ...], "active_child": id | None, "nodes":
{id: {"role", "content", "parent_id", "children": [id, ...],
"active_child": id | None}}}. The tree dict itself acts as the implicit
parent of the top-level messages (parent_id=None), using the exact same
children/active_child shape a node uses -- so editing/branching the very
first message in a chat works the same way as any other, no special case
needed.
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


def new_node_id() -> str:
    return uuid.uuid4().hex[:10]


def new_tree() -> dict:
    return {"children": [], "active_child": None, "nodes": {}}


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
    data = json.loads(path.read_text())
    if "nodes" not in data:
        data = _migrate_flat(data)
    return data


def _migrate_flat(data: dict) -> dict:
    """Chats saved before branching existed were a flat "messages":
    [{role, content}, ...] list -- turn that into a straight-line tree
    (no branches) the first time it's loaded under the new format.
    save_chat always writes the new shape, so this is a one-time
    per-chat upgrade, not something that runs on every load."""
    tree = new_tree()
    parent_id = None
    for msg in data.get("messages", []):
        parent_id = append_message(tree, parent_id, msg.get("role"), msg.get("content", ""))
    data["children"] = tree["children"]
    data["active_child"] = tree["active_child"]
    data["nodes"] = tree["nodes"]
    data.pop("messages", None)
    return data


def save_chat(mode: str, chat_id: str, title: str, tree: dict) -> None:
    path = _mode_dir(mode) / f"{chat_id}.json"
    data = {
        "id": chat_id,
        "title": title or UNTITLED,
        "updated_at": time.time(),
        "children": tree["children"],
        "active_child": tree["active_child"],
        "nodes": tree["nodes"],
    }
    path.write_text(json.dumps(data, indent=2))


def rename_chat(mode: str, chat_id: str, new_title: str) -> None:
    data = load_chat(mode, chat_id)
    save_chat(mode, chat_id, new_title.strip() or UNTITLED, data)


def delete_chat(mode: str, chat_id: str) -> None:
    (_mode_dir(mode) / f"{chat_id}.json").unlink(missing_ok=True)


# -- Tree operations ---------------------------------------------------
#
# All of these take a "tree" dict shaped like new_tree()'s return value
# (load_chat's return value already has the right keys, so it can be
# passed straight through). Callers are responsible for calling
# save_chat afterward -- these just mutate the tree in memory.

def _siblings_and_setter(tree: dict, parent_id: str | None):
    """Returns (sibling_id_list, set_active_fn) for whatever parent_id
    is -- None means the top level, where the tree dict itself acts as
    the implicit parent."""
    container = tree if parent_id is None else tree["nodes"][parent_id]

    def set_active(child_id: str) -> None:
        container["active_child"] = child_id

    return container["children"], set_active


def active_path(tree: dict) -> list[dict]:
    """Walks from the top down through active_child pointers, returning
    each node on the current branch with its id and its full sibling
    list (for the "< 2/3 >" navigator -- switching to a sibling just
    means sending its id back as a switch_branch request)."""
    path = []
    node_id = tree["active_child"]
    nodes = tree["nodes"]
    while node_id is not None:
        node = nodes[node_id]
        siblings, _ = _siblings_and_setter(tree, node["parent_id"])
        path.append({
            "id": node_id,
            "role": node["role"],
            "content": node["content"],
            "siblings": list(siblings),
        })
        node_id = node["active_child"]
    return path


def active_messages(tree: dict) -> list[dict]:
    """Just role/content, in the shape core/llm.py expects."""
    return [{"role": n["role"], "content": n["content"]} for n in active_path(tree)]


def path_to(tree: dict, node_id: str) -> list[dict]:
    """Messages from the root down to node_id (inclusive), in the shape
    core/llm.py expects -- used to build the LLM context when generating
    a reply that follows a specific node, which might not be the current
    active leaf (e.g. regenerating an older reply after later switching
    branches elsewhere)."""
    chain = []
    nodes = tree["nodes"]
    while node_id is not None:
        node = nodes[node_id]
        chain.append({"role": node["role"], "content": node["content"]})
        node_id = node["parent_id"]
    chain.reverse()
    return chain


def append_message(tree: dict, parent_id: str | None, role: str, content: str) -> str:
    """Adds a new node under parent_id (None = top level) and makes it
    the active branch there -- this is the normal "send a message"
    path, not branching, so there's nothing to preserve."""
    node_id = new_node_id()
    tree["nodes"][node_id] = {
        "role": role,
        "content": content,
        "parent_id": parent_id,
        "children": [],
        "active_child": None,
    }
    siblings, set_active = _siblings_and_setter(tree, parent_id)
    siblings.append(node_id)
    set_active(node_id)
    return node_id


def edit_message(tree: dict, node_id: str, new_content: str) -> str:
    """Creates a new sibling with the edited content under node_id's
    parent, and makes it the active branch. node_id and everything that
    was generated after it are untouched on disk, just no longer on the
    active path -- switch_branch can bring them back."""
    node = tree["nodes"][node_id]
    return append_message(tree, node["parent_id"], node["role"], new_content)


def remove_leaf(tree: dict, node_id: str) -> None:
    """Removes a childless node and detaches it from its parent -- used
    to retract a turn that was appended right before generation started,
    when that generation got stopped before any content came back at
    all (nothing worth keeping). Only safe on leaves; doesn't recurse."""
    node = tree["nodes"].pop(node_id)
    siblings, set_active = _siblings_and_setter(tree, node["parent_id"])
    siblings.remove(node_id)
    set_active(siblings[-1] if siblings else None)


def switch_branch(tree: dict, node_id: str) -> None:
    """Makes node_id the active child of its parent. Whatever comes
    after node_id resolves on its own via active_path -- each deeper
    branch point keeps its own active_child from whenever it was last
    the current branch, so switching back to an old branch resumes
    exactly where you left it."""
    node = tree["nodes"][node_id]
    _, set_active = _siblings_and_setter(tree, node["parent_id"])
    set_active(node_id)
