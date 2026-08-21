import httpx 
import time 
import subprocess

ollama_url = "http://localhost:11434"
model_name = "gemma4:e4b"


def is_ollama_running() -> bool:
    try: 
        httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
        return True
    except httpx.ConnectError: 
        return False

def ensure_server_running() -> None:
    if is_ollama_running():
        return
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):
        if is_ollama_running():
            return
        time.sleep(0.5)
    raise RuntimeError("Ollama server didn't start in time.")


def ensure_model_pulled(model_name: str) -> None:
    response = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
    models = [m["name"] for m in response.json().get("models", [])]
    if model_name not in models:
        subprocess.run(["ollama", "pull", model_name], check=True)   

async def generate_reply(user_text : str) -> str: 
    payload = {
        "model" : model_name, 
        "messages" : [{"role" : "user", "content" : user_text}],
        "stream" : False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client: 
        response = await client.post(f"{ollama_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    return data["message"]["content"]

