"""voxPersona web backend. Serves the static frontend (webapp/static/) and
exposes chat history over REST plus live text/audio chat over WebSocket,
reusing the same core/ services the TUI was built on (VADService,
TranscriptionService, DialogueService, core/llm.py, core/ChatHistory.py) --
only the transport changed: audio capture and playback now happen in the
browser instead of via sounddevice on the machine running the app, which is
what makes this deployable to a server instead of only running locally.
"""
import asyncio
import base64
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import core.ChatHistory as history
import core.llm as llm
import core.TranscriptionService as transcription
from core.DialogueService import DialogueService
from core.TranscriptionService import TranscriptionService
from core.VADService import VADService

STATIC_DIR = Path(__file__).parent / "static"

# Replies here go through TTS -- markdown syntax and emoji get read aloud
# literally (kokoro has no idea "*" means emphasis, it just says
# "asterisk"), so this asks the model for plain, speakable text instead.
AUDIO_SYSTEM_PROMPT = (
    "You are a helpful voice assistant having a spoken conversation. Never "
    "use markdown formatting -- no asterisks, bullet points, headers, or "
    "code blocks -- and never use emoji. Your replies are converted to "
    "speech and read aloud, so write only plain, speakable text."
)

# Sentence-ending punctuation followed by whitespace -- used to flush
# complete sentences to TTS as they arrive in the reply stream, rather than
# waiting for the whole reply (see _handle_utterance).
SENTENCE_END_RE = re.compile(r"[.!?]+\s+")

app = FastAPI()

# TranscriptionService (on the MLX backend) loads its ASR model onto a
# dedicated thread the moment it's constructed, and nothing in it caches
# that model at module level the way the other backends do -- constructing
# a fresh one per connection would reload the whole model, and leak a
# thread, every time someone opens the audio screen. DialogueService's
# kokoro pipeline is similarly expensive to construct. Both are shared,
# lazily-created singletons instead; VADService is cheap and *must* stay
# per-connection since its silence timers/RNN state would otherwise get
# corrupted by two people talking at once.
_transcription_service: TranscriptionService | None = None
_dialogue_service: DialogueService | None = None
_service_lock = asyncio.Lock()


async def get_transcription_service() -> TranscriptionService:
    global _transcription_service
    async with _service_lock:
        if _transcription_service is None:
            loop = asyncio.get_event_loop()
            _transcription_service = await loop.run_in_executor(
                None, lambda: TranscriptionService(backend=transcription.DEFAULT_BACKEND)
            )
    return _transcription_service


async def get_dialogue_service() -> DialogueService:
    global _dialogue_service
    async with _service_lock:
        if _dialogue_service is None:
            loop = asyncio.get_event_loop()
            _dialogue_service = await loop.run_in_executor(None, lambda: DialogueService(engine="kokoro"))
    return _dialogue_service


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, llm.ensure_server_running)
    await loop.run_in_executor(None, llm.ensure_model_pulled, llm.model_name)


# -- Chat history (REST) -------------------------------------------------

class RenameBody(BaseModel):
    title: str


@app.get("/api/chats/{mode}")
def api_list_chats(mode: str):
    return history.list_chats(mode)


@app.post("/api/chats/{mode}")
def api_new_chat(mode: str):
    # Doesn't persist anything yet -- same as the TUI, a chat only gets
    # written to disk once it has at least one message (see
    # core/ChatHistory.py's save_chat / persist-on-first-exchange pattern).
    return {"id": history.new_chat_id()}


@app.get("/api/chats/{mode}/{chat_id}")
def api_load_chat(mode: str, chat_id: str):
    try:
        return history.load_chat(mode, chat_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")


@app.patch("/api/chats/{mode}/{chat_id}")
def api_rename_chat(mode: str, chat_id: str, body: RenameBody):
    history.rename_chat(mode, chat_id, body.title)
    return {"ok": True}


@app.delete("/api/chats/{mode}/{chat_id}")
def api_delete_chat(mode: str, chat_id: str):
    history.delete_chat(mode, chat_id)
    return {"ok": True}


# -- Settings (REST) -------------------------------------------------

MODEL_SIZES = ["e2b", "e4b"]
TRANSCRIPTION_BACKENDS = ["auto", "mlx", "sherpa", "gemma"]


class ModelBody(BaseModel):
    size: str


class TranscriptionBackendBody(BaseModel):
    backend: str


@app.get("/api/settings")
def api_get_settings():
    return {
        "model": llm.model_name,
        "model_sizes": MODEL_SIZES,
        "transcription_backend": transcription.DEFAULT_BACKEND or "auto",
        "transcription_backends": TRANSCRIPTION_BACKENDS,
    }


@app.post("/api/settings/model")
async def api_set_model(body: ModelBody):
    if body.size not in MODEL_SIZES:
        raise HTTPException(status_code=400, detail=f"size must be one of {MODEL_SIZES}")
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(None, llm.set_model_size, body.size)
    await loop.run_in_executor(None, llm.ensure_model_pulled, model)
    return {"model": model}


@app.post("/api/settings/transcription-backend")
async def api_set_transcription_backend(body: TranscriptionBackendBody):
    global _transcription_service
    if body.backend not in TRANSCRIPTION_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {TRANSCRIPTION_BACKENDS}")
    transcription.DEFAULT_BACKEND = None if body.backend == "auto" else body.backend
    async with _service_lock:
        # The existing singleton has the *old* backend's model baked in --
        # drop it so the next get_transcription_service() rebuilds fresh
        # with the new one, same idea as the TUI's SettingsScreen toggle
        # constructing a new TranscriptionService for the next screen.
        _transcription_service = None
    return {"transcription_backend": body.backend}


# -- Text mode (WebSocket) ------------------------------------------------

@app.websocket("/ws/text/{chat_id}")
async def ws_text(websocket: WebSocket, chat_id: str) -> None:
    await websocket.accept()
    try:
        data = history.load_chat("text", chat_id)
        messages: list[dict] = data.get("messages", [])
        title = data.get("title")
    except FileNotFoundError:
        messages = []
        title = None

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            if payload.get("type") != "message":
                continue
            user_text = (payload.get("text") or "").strip()
            if not user_text:
                continue

            messages.append({"role": "user", "content": user_text})

            reply_text = ""
            async for chunk in llm.generate_reply_stream(messages):
                reply_text += chunk
                await websocket.send_json({"type": "delta", "text": chunk})

            messages.append({"role": "assistant", "content": reply_text})

            # First exchange of a new chat -- name it, same idea as
            # mainstream chat UIs auto-titling from the opening turn.
            if title is None:
                title = await llm.generate_title(messages)
                await websocket.send_json({"type": "title", "title": title})

            history.save_chat("text", chat_id, title, messages)
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass


# -- Push-to-talk dictation (WebSocket) ------------------------------------
#
# Separate from /ws/audio: this is the mic button in *text* mode -- no VAD,
# no auto-reply, no TTS, just live partial transcription while the button's
# held, into the text box for you to review/edit before actually sending.
# Turn-taking is manual (start/stop messages from the client) instead of
# VAD-driven since there's no hands-free conversation loop to drive here.

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    transcription_service = await get_transcription_service()
    session = None

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                if session is None:
                    continue
                text = await session.feed(message["bytes"])
                if text.strip():
                    await websocket.send_json({"type": "partial", "text": text})
                continue

            if message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "start":
                    session = transcription_service.start_session()
                elif control.get("type") == "stop":
                    if session is not None:
                        text = (await session.finish(b"")).strip()
                        session = None
                        await websocket.send_json({"type": "final", "text": text})
    except WebSocketDisconnect:
        pass


# -- Audio mode (WebSocket) ------------------------------------------------

@app.websocket("/ws/audio/{chat_id}")
async def ws_audio(websocket: WebSocket, chat_id: str) -> None:
    await websocket.accept()
    try:
        data = history.load_chat("audio", chat_id)
        messages: list[dict] = data.get("messages", [])
        title = data.get("title")
    except FileNotFoundError:
        messages = []
        title = None

    vad_service = VADService()
    transcription_service = await get_transcription_service()
    session = None
    utterance_chunks: list[bytes] = []

    async def send_status(text: str) -> None:
        await websocket.send_json({"type": "status", "text": text})

    async def respond_to(text: str) -> None:
        nonlocal title

        messages.append({"role": "user", "content": text})
        await send_status("Thinking...")

        dialogue_service = await get_dialogue_service()

        reply_text = ""
        sentence_buffer = ""
        sentence_index = 0
        spoken_yet = False

        async def speak(sentence: str) -> None:
            nonlocal sentence_index, spoken_yet
            audio = await dialogue_service.runTTS(sentence)
            if not audio["audio"]:
                return
            if not spoken_yet:
                spoken_yet = True
                await send_status("Speaking...")
            await websocket.send_json({
                "type": "audio",
                "index": sentence_index,
                "audio": base64.b64encode(audio["audio"]).decode("ascii"),
            })
            sentence_index += 1

        async for chunk in llm.generate_reply_stream(messages, system_prompt=AUDIO_SYSTEM_PROMPT):
            reply_text += chunk
            sentence_buffer += chunk
            await websocket.send_json({"type": "reply_delta", "text": chunk})

            while (m := SENTENCE_END_RE.search(sentence_buffer)) is not None:
                sentence, sentence_buffer = sentence_buffer[:m.end()].strip(), sentence_buffer[m.end():]
                if sentence:
                    await speak(sentence)

        if sentence_buffer.strip():
            await speak(sentence_buffer.strip())

        messages.append({"role": "assistant", "content": reply_text})
        await websocket.send_json({"type": "reply_done"})

        if title is None:
            title = await llm.generate_title(messages)
            await websocket.send_json({"type": "title", "title": title})

        history.save_chat("audio", chat_id, title, messages)
        await send_status("Listening...")

    async def handle_spoken_utterance(full_audio: bytes) -> None:
        final_session = transcription_service.start_session()
        text = (await final_session.finish(full_audio)).strip()
        if not text:
            await send_status("Listening...")
            return
        await websocket.send_json({"type": "transcript_final", "text": text})
        await respond_to(text)

    try:
        await send_status("Listening...")
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                chunk = message["bytes"]
                for event in vad_service.process_pcm(chunk):
                    if event["type"] == "partial_ready":
                        utterance_chunks.append(event["audio"])
                        if session is None:
                            session = transcription_service.start_session()
                        text = await session.feed(event["audio"])
                        if text.strip():
                            await websocket.send_json({"type": "transcript_partial", "text": text})
                    elif event["type"] == "silence_detected":
                        utterance_chunks.append(event["audio"])
                        if session is not None:
                            await session.finish(b"")
                            session = None
                        chunks, utterance_chunks = utterance_chunks, []
                        await send_status("Transcribing...")
                        await handle_spoken_utterance(b"".join(chunks))
            elif message.get("text") is not None:
                control = json.loads(message["text"])
                control_type = control.get("type")
                if control_type == "reset":
                    vad_service.reset()
                    utterance_chunks = []
                    session = None
                elif control_type == "message":
                    # Typed input while in audio mode -- e.g. the mic
                    # isn't cooperating, or you just don't want to talk.
                    # Skips transcription but goes through the exact same
                    # reply+TTS pipeline as a spoken turn, so the reply is
                    # still read aloud.
                    typed_text = (control.get("text") or "").strip()
                    if typed_text:
                        await respond_to(typed_text)
    except WebSocketDisconnect:
        pass


# Mounted last -- FastAPI matches routes in registration order, and this
# catch-all (html=True serves index.html for any unmatched path) would
# otherwise shadow the /api and /ws routes above.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
