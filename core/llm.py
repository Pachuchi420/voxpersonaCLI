import httpx
import json
import time
import subprocess
import platform

ollama_url = "http://localhost:11434"



def determine_system() -> bool: 
    try: 
        import mlx.core as mx
        return mx.default_device().type == mx.DeviceType.gpu
    except ImportError: 
        return False


def model_tag(size: str) -> str:
    """size is 'e2b' or 'e4b' -- preserves the mlx/non-mlx suffix logic
    above so switching size doesn't also require the caller to know
    which platform it's running on."""
    return f"gemma4:{size}-mlx" if determine_system() else f"gemma4:{size}"


def set_model_size(size: str) -> str:
    """Mutable app-wide preference, same pattern as
    core/TranscriptionService.py's DEFAULT_BACKEND -- the settings UI
    calls this directly, and generate_reply_stream/generate_title below
    pick up the new value on their next call since they read module_name
    at call time, not at import time."""
    global model_name
    model_name = model_tag(size)
    return model_name


model_name = model_tag("e2b")


def is_llm_running() -> bool:
    try: 
        httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
        return True
    except httpx.ConnectError: 
        return False

def ensure_server_running() -> None:
    if is_llm_running():
        return
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):
        if is_llm_running():
            return
        time.sleep(0.5)
    raise RuntimeError("Ollama server didn't start in time.")


def ensure_model_pulled(model_name: str) -> None:
    if model_name not in list_installed_models():
        subprocess.run(["ollama", "pull", model_name], check=True)


def list_installed_models() -> list[str]:
    """Whatever's actually pulled, straight from Ollama -- this is what
    drives the Settings model picker, so picking up a new model is just
    `ollama pull <tag>` (or the pull_model() below), no code change."""
    response = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    return sorted(m["name"] for m in response.json().get("models", []))


def set_model(tag: str) -> str:
    """Mutable app-wide preference, same pattern as
    core/TranscriptionService.py's DEFAULT_BACKEND -- generate_reply_stream/
    generate_title below read model_name at call time, not at import time,
    so this takes effect on the next request with no other changes needed."""
    global model_name
    model_name = tag
    return model_name


def pull_model(tag: str) -> None:
    subprocess.run(["ollama", "pull", tag], check=True)

async def generate_reply_stream(messages: list[dict], system_prompt: str | None = None):
    """Yields the reply text incrementally as Ollama streams it back (NDJSON,
    one {"message": {"content": <delta>}, ...} object per line, ending with
    a final {"done": true, ...} summary object) instead of waiting for the
    whole reply -- lets callers show words appearing as they're generated,
    same as mainstream chat UIs, instead of the full text landing at once.

    system_prompt is prepended as a leading system message when given --
    e.g. AudioChatScreen passes one asking for plain, markdown/emoji-free
    text, since replies there get read aloud by TTS and things like "*" or
    an emoji get read out literally. Left out entirely by default (None)
    so callers that just display text, like TextChatScreen, are unaffected."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages
    payload = {
        "model" : model_name,
        "messages" : full_messages,
        "stream" : True,
        # Ollama's default keep_alive (5m) unloads the model between turns
        # if there's any gap -- the next request then pays ~4s+ just to
        # reload it (start the runner subprocess, load tensors) before any
        # actual generation happens. Keeping it resident for the life of a
        # conversation screen avoids that repeated cold-start cost.
        "keep_alive": "30m",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{ollama_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                content = json.loads(line).get("message", {}).get("content", "")
                if content:
                    yield content


async def generate_title(messages: list[dict]) -> str:
    """One-shot (non-streaming) call asking for a short chat title based on
    the conversation so far -- used to auto-name a new chat after its first
    exchange, same idea as mainstream chat UIs. Falls back to a truncated
    copy of the first user message if generation fails for any reason (bad
    JSON, empty content, network hiccup) -- a so-so title beats a chat the
    sidebar can't render or a naming step that can crash the conversation."""
    prompt_messages = messages + [{
        "role": "user",
        "content": (
            "Summarize this conversation in 3 to 5 words for a chat title. "
            "Respond with only the title, no punctuation, no quotes."
        ),
    }]
    payload = {
        "model": model_name,
        "messages": prompt_messages,
        "stream": False,
        "keep_alive": "30m",
    }

    fallback = next((m["content"] for m in messages if m.get("role") == "user"), "New Chat")
    fallback = fallback.strip()[:40] or "New Chat"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            title = response.json()["message"]["content"].strip().strip('"').strip("'")
        return title[:60] or fallback
    except (httpx.HTTPError, KeyError, ValueError):
        return fallback

