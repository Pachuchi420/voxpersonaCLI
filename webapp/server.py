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
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import core.ChatHistory as history
import core.llm as llm
import core.TranscriptionService as transcription
from core.DialogueService import DialogueService
from core.EmotionService import EmotionService
from core.FacialEmotionService import FacialEmotionService
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
_emotion_service: EmotionService | None = None
_facial_emotion_service: FacialEmotionService | None = None
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


async def get_emotion_service() -> EmotionService:
    global _emotion_service
    async with _service_lock:
        if _emotion_service is None:
            loop = asyncio.get_event_loop()
            _emotion_service = await loop.run_in_executor(None, EmotionService)
    return _emotion_service


async def get_facial_emotion_service() -> FacialEmotionService:
    global _facial_emotion_service
    async with _service_lock:
        if _facial_emotion_service is None:
            loop = asyncio.get_event_loop()
            _facial_emotion_service = await loop.run_in_executor(None, FacialEmotionService)
    return _facial_emotion_service


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, llm.ensure_server_running)
    await loop.run_in_executor(None, llm.ensure_model_pulled, llm.model_name)

    # Hot-load every model-backed service now rather than lazily on
    # whichever request happens to hit it first -- without this, the
    # first utterance or webcam frame after a server restart eats the
    # full model-load cost (confirmed: ~15-20s for emotion2vec/the
    # facial pipeline) on top of actually doing the work, which reads as
    # the whole feature "hanging" rather than a one-time startup cost.
    # (Each getter serializes on _service_lock, so these run one after
    # another, not in parallel -- fine for a one-time startup cost.)
    await get_transcription_service()
    await get_dialogue_service()
    await get_emotion_service()
    await get_facial_emotion_service()


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
        tree = history.load_chat(mode, chat_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"id": chat_id, "title": tree.get("title"), "path": history.active_path(tree)}


@app.patch("/api/chats/{mode}/{chat_id}")
def api_rename_chat(mode: str, chat_id: str, body: RenameBody):
    history.rename_chat(mode, chat_id, body.title)
    return {"ok": True}


@app.delete("/api/chats/{mode}/{chat_id}")
def api_delete_chat(mode: str, chat_id: str):
    history.delete_chat(mode, chat_id)
    return {"ok": True}


# -- Settings (REST) -------------------------------------------------

TRANSCRIPTION_BACKENDS = ["auto", "mlx", "sherpa", "gemma"]


class ModelBody(BaseModel):
    model: str


class TranscriptionBackendBody(BaseModel):
    backend: str


@app.get("/api/settings")
def api_get_settings():
    return {
        "model": llm.model_name,
        # Whatever's actually installed in Ollama right now -- pull a new
        # one (see /api/settings/models/pull) and it shows up here with no
        # code change, that's the whole "plug and play" point.
        "models": llm.list_installed_models(),
        "transcription_backend": transcription.DEFAULT_BACKEND or "auto",
        "transcription_backends": TRANSCRIPTION_BACKENDS,
    }


@app.post("/api/settings/model")
def api_set_model(body: ModelBody):
    return {"model": llm.set_model(body.model)}


@app.post("/api/settings/models/pull")
async def api_pull_model(body: ModelBody):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, llm.pull_model, body.model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pull failed: {exc}")
    return {"model": llm.set_model(body.model)}


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
        tree = history.load_chat("text", chat_id)
        title = tree.get("title")
    except FileNotFoundError:
        tree = history.new_tree()
        title = None

    generation_task: asyncio.Task | None = None

    def run_task(coro) -> None:
        # Runs a turn as a background task instead of awaiting it inline,
        # specifically so the receive loop below stays free to notice a
        # "stop" message arriving mid-turn and cancel it.
        nonlocal generation_task

        async def wrapper() -> None:
            nonlocal generation_task
            try:
                await coro
            finally:
                generation_task = None

        generation_task = asyncio.create_task(wrapper())

    async def run_generation(parent_id: str) -> None:
        # Shared by every way a reply can get generated -- a plain send,
        # an edit, or a regenerate -- all of which differ only in *which*
        # node the new reply should be a child of; the streaming/
        # cancellation/persistence logic here is identical either way.
        nonlocal title
        llm_messages = history.path_to(tree, parent_id)
        reply_text = ""
        stopped = False
        try:
            async for chunk in llm.generate_reply_stream(llm_messages):
                reply_text += chunk
                await websocket.send_json({"type": "delta", "text": chunk})
        except asyncio.CancelledError:
            # Cancelling this task also cancels whatever it's awaiting --
            # here, the httpx stream reading from Ollama -- which closes
            # that connection, which is what actually stops Ollama from
            # continuing to generate tokens nobody's listening for anymore.
            stopped = True

        if reply_text.strip():
            history.append_message(tree, parent_id, "assistant", reply_text)
            if title is None:
                title = await llm.generate_title(llm_messages + [{"role": "assistant", "content": reply_text}])
                await websocket.send_json({"type": "title", "title": title})
            history.save_chat("text", chat_id, title, tree)
        elif tree["nodes"][parent_id]["role"] == "user" and not tree["nodes"][parent_id]["children"]:
            # Stopped before any reply text came back at all, and this
            # was a freshly-appended user turn (not a regenerate, which
            # points at an existing node with history before it) --
            # nothing worth keeping, retract the dangling turn too.
            history.remove_leaf(tree, parent_id)

        await websocket.send_json({"type": "stopped" if stopped else "done"})
        await websocket.send_json({"type": "sync", "path": history.active_path(tree)})

    async def handle_message(user_text: str) -> None:
        path = history.active_path(tree)
        leaf_id = path[-1]["id"] if path else None
        new_user_id = history.append_message(tree, leaf_id, "user", user_text)
        await run_generation(new_user_id)

    async def handle_edit(node_id: str, new_text: str) -> None:
        new_node_id = history.edit_message(tree, node_id, new_text)
        await run_generation(new_node_id)

    async def handle_regenerate(node_id: str) -> None:
        parent_id = tree["nodes"][node_id]["parent_id"]
        if parent_id is not None:
            await run_generation(parent_id)

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            msg_type = payload.get("type")
            node_id = payload.get("node_id")

            if msg_type == "message":
                user_text = (payload.get("text") or "").strip()
                if user_text and generation_task is None:
                    run_task(handle_message(user_text))
            elif msg_type == "edit":
                new_text = (payload.get("text") or "").strip()
                if node_id in tree["nodes"] and new_text and generation_task is None:
                    run_task(handle_edit(node_id, new_text))
            elif msg_type == "regenerate":
                if node_id in tree["nodes"] and generation_task is None:
                    run_task(handle_regenerate(node_id))
            elif msg_type == "switch_branch":
                if node_id in tree["nodes"]:
                    history.switch_branch(tree, node_id)
                    history.save_chat("text", chat_id, title or history.UNTITLED, tree)
                    await websocket.send_json({"type": "sync", "path": history.active_path(tree)})
            elif msg_type == "stop":
                if generation_task is not None:
                    generation_task.cancel()
    except WebSocketDisconnect:
        if generation_task is not None:
            generation_task.cancel()


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
        tree = history.load_chat("audio", chat_id)
        title = tree.get("title")
    except FileNotFoundError:
        tree = history.new_tree()
        title = None

    vad_service = VADService()
    transcription_service = await get_transcription_service()
    session = None
    utterance_chunks: list[bytes] = []
    # Holds strong references to fire-and-forget background tasks (see
    # detect_emotion below) -- asyncio only keeps a *weak* reference to a
    # task internally, so one with nothing else referencing it can get
    # garbage-collected mid-run, silently cutting it short. Each task
    # removes itself via its done-callback once finished.
    background_tasks: set[asyncio.Task] = set()

    async def send_status(text: str) -> None:
        await websocket.send_json({"type": "status", "text": text})

    current_task: asyncio.Task | None = None

    def run_turn(coro) -> None:
        # Runs a turn (spoken or typed) as a background task instead of
        # awaiting it inline, specifically so the receive loop below stays
        # free to notice a "stop" control message arriving mid-turn and
        # cancel it -- awaiting inline would block the loop from reading
        # anything else until the turn finished on its own. The wrapper's
        # `finally` is what current_task is None checks below rely on to
        # know a turn has ended, on every exit path (including the early
        # "transcription came back empty" return in handle_spoken_utterance).
        nonlocal current_task

        async def wrapper() -> None:
            nonlocal current_task
            try:
                await coro
            finally:
                current_task = None

        current_task = asyncio.create_task(wrapper())

    async def run_generation(parent_id: str) -> None:
        # Shared by every way a reply can get generated -- a spoken turn,
        # a typed turn, an edit, or a regenerate -- all of which differ
        # only in *which* node the new reply should be a child of.
        nonlocal title

        await send_status("Thinking...")
        dialogue_service = await get_dialogue_service()
        llm_messages = history.path_to(tree, parent_id)

        reply_text = ""
        sentence_buffer = ""
        sentence_index = 0
        spoken_yet = False
        stopped = False

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

        try:
            async for chunk in llm.generate_reply_stream(llm_messages, system_prompt=AUDIO_SYSTEM_PROMPT):
                reply_text += chunk
                sentence_buffer += chunk
                await websocket.send_json({"type": "reply_delta", "text": chunk})

                while (m := SENTENCE_END_RE.search(sentence_buffer)) is not None:
                    sentence, sentence_buffer = sentence_buffer[:m.end()].strip(), sentence_buffer[m.end():]
                    if sentence:
                        await speak(sentence)

            if sentence_buffer.strip():
                await speak(sentence_buffer.strip())
        except asyncio.CancelledError:
            # See the matching comment in ws_text's run_generation --
            # cancelling this task also cancels whatever it's currently
            # awaiting (the LLM stream, or a runTTS() call), which is what
            # actually stops work happening for a turn nobody's listening
            # for anymore.
            stopped = True

        if reply_text.strip():
            history.append_message(tree, parent_id, "assistant", reply_text)
            if title is None:
                title = await llm.generate_title(llm_messages + [{"role": "assistant", "content": reply_text}])
                await websocket.send_json({"type": "title", "title": title})
            history.save_chat("audio", chat_id, title, tree)
        elif tree["nodes"][parent_id]["role"] == "user" and not tree["nodes"][parent_id]["children"]:
            history.remove_leaf(tree, parent_id)

        await websocket.send_json({"type": "stopped" if stopped else "reply_done"})
        await websocket.send_json({"type": "sync", "path": history.active_path(tree)})
        await send_status("Listening...")

    async def respond_to(text: str) -> None:
        path = history.active_path(tree)
        leaf_id = path[-1]["id"] if path else None
        new_user_id = history.append_message(tree, leaf_id, "user", text)
        await run_generation(new_user_id)

    async def handle_edit(node_id: str, new_text: str) -> None:
        new_node_id = history.edit_message(tree, node_id, new_text)
        await run_generation(new_node_id)

    async def handle_regenerate(node_id: str) -> None:
        parent_id = tree["nodes"][node_id]["parent_id"]
        if parent_id is not None:
            await run_generation(parent_id)

    async def detect_emotion(audio: bytes) -> None:
        # Fire-and-forget, deliberately not awaited by handle_spoken_
        # utterance below -- this is a debug-stage addition, layered on
        # top of the exact same utterance buffer transcription already
        # got, and shouldn't add latency to the actual conversation turn
        # (or take the turn down with it if classification ever errors).
        # Printed server-side as well as sent over the socket so the
        # model's full score distribution -- not just the top label -- is
        # easy to eyeball against what was actually said, to get a feel
        # for whether it's worth trusting before it drives anything.
        try:
            started = time.monotonic()
            emotion_service = await get_emotion_service()
            scores = await emotion_service.classify(audio)
            elapsed_ms = round((time.monotonic() - started) * 1000)
            if not scores:
                return
            print(f"[EmotionService] {elapsed_ms}ms: {scores}", flush=True)
            await websocket.send_json({"type": "emotion", "scores": scores, "elapsed_ms": elapsed_ms})
        except Exception as exc:
            print(f"[EmotionService] classification failed: {exc}", flush=True)

    async def detect_face(image_bytes: bytes) -> None:
        # Same fire-and-forget shape as detect_emotion above -- not tied
        # to an utterance (facial expression isn't VAD-bound the way
        # speech is), just fired per captured frame (see "video_frame"
        # below), and shouldn't block the receive loop from handling
        # anything else while a frame is being classified.
        #
        # Always sends a reply, even when no face was found or
        # classification errored -- audio.js waits for this to know the
        # previous frame is done before capturing the next one, rather
        # than firing on a fixed timer regardless of whether the server
        # has caught up. Without an ack on every path, a frame that
        # found no face (or hit an error) would leave the client waiting
        # forever and stuck never sending another one.
        elapsed_ms = 0
        scores, box, landmarks = [], None, []
        try:
            started = time.monotonic()
            facial_service = await get_facial_emotion_service()
            result = await facial_service.classify(image_bytes)
            elapsed_ms = round((time.monotonic() - started) * 1000)
            scores, box, landmarks = result["scores"], result["box"], result["landmarks"]
            if scores:
                print(f"[FacialEmotionService] {elapsed_ms}ms: {scores}", flush=True)
        except Exception as exc:
            print(f"[FacialEmotionService] classification failed: {exc}", flush=True)
        await websocket.send_json({
            "type": "facial_emotion",
            "scores": scores,
            "box": box,
            "landmarks": landmarks,
            "elapsed_ms": elapsed_ms,
        })

    async def handle_spoken_utterance(full_audio: bytes) -> None:
        final_session = transcription_service.start_session()
        text = (await final_session.finish(full_audio)).strip()
        if not text:
            await send_status("Listening...")
            return
        await websocket.send_json({"type": "transcript_final", "text": text})
        emotion_task = asyncio.create_task(detect_emotion(full_audio))
        background_tasks.add(emotion_task)
        emotion_task.add_done_callback(background_tasks.discard)
        await respond_to(text)

    async def finish_utterance(final_audio: bytes) -> None:
        # Shared by the normal VAD-driven silence_detected event and the
        # manual "finalize" control below (see force_finish in
        # VADService.py) -- both end an utterance the same way, they just
        # differ in *why* it's considered over.
        nonlocal session, utterance_chunks
        utterance_chunks.append(final_audio)
        if session is not None:
            await session.finish(b"")
            session = None
        chunks, utterance_chunks = utterance_chunks, []
        await send_status("Transcribing...")
        run_turn(handle_spoken_utterance(b"".join(chunks)))

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
                        await finish_utterance(event["audio"])
            elif message.get("text") is not None:
                control = json.loads(message["text"])
                control_type = control.get("type")
                node_id = control.get("node_id")
                if control_type == "reset":
                    vad_service.reset()
                    utterance_chunks = []
                    session = None
                elif control_type == "finalize":
                    # Manual "I'm done talking" signal, sent when muting
                    # (see audio.js) -- for environments too noisy for
                    # Silero's speech/non-speech model to reliably tell
                    # background noise from true silence, so the normal
                    # 1.5s-of-silence threshold may never cross on its
                    # own. No-op if nothing was actually mid-utterance.
                    event = vad_service.force_finish()
                    if event is not None:
                        await finish_utterance(event["audio"])
                elif control_type == "message":
                    # Typed input while in audio mode -- e.g. the mic
                    # isn't cooperating, or you just don't want to talk.
                    # Skips transcription but goes through the exact same
                    # reply+TTS pipeline as a spoken turn, so the reply is
                    # still read aloud.
                    typed_text = (control.get("text") or "").strip()
                    if typed_text and current_task is None:
                        run_turn(respond_to(typed_text))
                elif control_type == "edit":
                    new_text = (control.get("text") or "").strip()
                    if node_id in tree["nodes"] and new_text and current_task is None:
                        run_turn(handle_edit(node_id, new_text))
                elif control_type == "regenerate":
                    if node_id in tree["nodes"] and current_task is None:
                        run_turn(handle_regenerate(node_id))
                elif control_type == "switch_branch":
                    if node_id in tree["nodes"]:
                        history.switch_branch(tree, node_id)
                        history.save_chat("audio", chat_id, title or history.UNTITLED, tree)
                        await websocket.send_json({"type": "sync", "path": history.active_path(tree)})
                elif control_type == "stop":
                    if current_task is not None:
                        current_task.cancel()
                elif control_type == "video_frame":
                    # Webcam toggle in audio.js -- a JPEG frame, base64-
                    # encoded the same way "audio" (TTS output) already is
                    # in the other direction. Fired off, not awaited, so a
                    # slow classification can't stall the receive loop or
                    # delay the next frame's capture. Whether the *overlay*
                    # drawn from the result shows up is a pure client-side
                    # preference (see face-overlay-pref.js) -- this always
                    # processes a frame it's sent.
                    image_b64 = control.get("image")
                    if image_b64:
                        face_task = asyncio.create_task(detect_face(base64.b64decode(image_b64)))
                        background_tasks.add(face_task)
                        face_task.add_done_callback(background_tasks.discard)
    except WebSocketDisconnect:
        if current_task is not None:
            current_task.cancel()


# Mounted last -- FastAPI matches routes in registration order, and this
# catch-all (html=True serves index.html for any unmatched path) would
# otherwise shadow the /api and /ws routes above.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
