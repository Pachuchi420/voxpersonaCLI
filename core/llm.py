import httpx 
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


model_name = "gemma4:e4b-mlx" if determine_system() else "gemma4:e4b"


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

async def generate_reply(messages: list[dict]) -> str:
    payload = {
        "model" : model_name,
        "messages" : messages,
        "stream" : False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client: 
        response = await client.post(f"{ollama_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    return data["message"]["content"]

