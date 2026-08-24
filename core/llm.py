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


model_name = "gemma4:e2b-mlx" if determine_system() else "gemma4:e2b"


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
    response = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    models = [m["name"] for m in response.json().get("models", [])]
    if model_name not in models:
        subprocess.run(["ollama", "pull", model_name], check=True)   

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

