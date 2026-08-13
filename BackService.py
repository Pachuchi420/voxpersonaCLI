import websockets
import asyncio
import json
import base64
import sys
import time
from VADService import VADService
from TranscriptionService import TranscriptionService
from DialogueService import (
    DialogueService, _all_turns_cached, load_cached_clips, _backchannel_pools_cached,
    pick_backchannel_audio, is_repeat_request, pick_repeat_response_audio,
)


# `python BackService.py 1` selects sherpa-onnx instead of the default ASR
# backend (MLX on Mac, NeMo "checkpoint" elsewhere) -- lets the two be A/B
# tested on the same Windows machine without touching code.
_TRANSCRIPTION_BACKEND = "sherpa" if len(sys.argv) > 1 and sys.argv[1] == "1" else None

vad_service = VADService()
transcription_service = TranscriptionService(backend=_TRANSCRIPTION_BACKEND)
print(f"[BackService] Transcription backend: {transcription_service.backend}", flush=True)




async def sendClips(websocket, clips):
    for clip in clips:
        audio_base64 = base64.b64encode(clip["audio"]).decode("utf-8")

        await websocket.send(json.dumps({
            "type": "dialogue",
            "audio": audio_base64,
            "index": clip["index"]
        }))

    await websocket.send(json.dumps({
        "type": "dialogue_end",
        "length": len(clips)
    }))


def _resolve_engine_and_voice(script: dict):
    """Mirrors FrontService.cs's ResolveEngine/ResolveVoice: whichever voice
    field is populated on the script decides the engine."""
    chatterbox_voice = script.get("chatterbox_voice")
    kokoro_voice = script.get("kokoro_voice")
    if chatterbox_voice:
        return "chatterbox", chatterbox_voice
    if kokoro_voice:
        return "kokoro", kokoro_voice
    return None, None


def _script_is_cached(script: dict) -> bool:
    """True only if every agent turn's clip, plus any enabled backchannel pools,
    are already on disk for this script's current engine/voice/language/text,
    i.e. no regeneration needed."""
    script_id = script.get("id")
    engine, voice = _resolve_engine_and_voice(script)
    if not script_id or not engine:
        return False

    language = (script.get("language") or "english").lower()
    if not _all_turns_cached(script_id, script.get("turns") or [], engine, voice, language):
        return False

    return _backchannel_pools_cached(
        engine, voice, language,
        script.get("affirmationUtterancesEnabled", False),
        script.get("repeatDetectionEnabled", False),
    )


async def handle(websocket):

    print("Client connected!")
    is_last = False
    dialogue_service = None
    is_listening = False
    affirm_enabled = False
    # Set from the script's start_dialogue payload; used to pick a backchannel
    # clip from the pre-generated pool when a "thinking_pause" fires mid-listen.
    current_engine = None
    current_voice = None
    current_language = "english"
    # Whole-script setting (not per-turn like affirm_enabled): whether to check
    # the participant's utterance for a "please repeat that" request.
    current_repeat_enabled = False
    # Mirrors what Unity assembles client-side from transcription_partial events,
    # so the repeat-request check has the full utterance to match against without
    # a round trip back to Unity or a second transcription pass.
    accumulated_transcript = ""
    # One TranscriptionSession per utterance -- created on "listen", fed
    # incrementally on each partial_ready, finished on silence_detected. See
    # TranscriptionService.TranscriptionSession for why this replaced
    # re-transcribing the whole buffer from scratch on every call.
    session = None
    # Most recently created handle_partial task. Tracked so silence_detected
    # can await it before finish() -- feed() calls must land on the session
    # in order (each one is a required audio increment now, not a
    # replaceable whole-buffer transcription), so this can no longer be
    # cancelled/discarded the way the old redundant-partial design could.
    pending_partial_task = None

    async def handle_partial(audio_bytes):
        # Runs as a background task -- affirmations/silence handling (see
        # "thinking_pause"/"silence_detected" below) are driven purely by
        # VAD timing and must never sit behind a transcription call.
        # feed() is now cheap (only the new audio since the last call, not
        # the whole utterance), so in practice this finishes quickly, but it
        # still stays non-blocking here since VAD timing must never depend
        # on transcription speed.
        nonlocal accumulated_transcript
        t0 = time.monotonic()
        text = await session.feed(audio_bytes)
        elapsed = time.monotonic() - t0

        if text.strip():
            print(f"[Transcribe] {elapsed:.2f}s -> {text!r}", flush=True)
            accumulated_transcript = text.strip()
            try:
                await websocket.send(json.dumps({
                    "type": "transcription_partial",
                    "text": text
                }))
            except websockets.ConnectionClosed:
                pass
        else:
            print(f"[Transcribe] {elapsed:.2f}s -> (empty)", flush=True)

    async for message in websocket:

        if isinstance(message, str):
            control = json.loads(message)
            msg_type = control.get("type")

            if msg_type == "listen":
                is_listening = True
                is_last = control.get("is_last", False)
                affirm_enabled = control.get("affirm_enabled", False)
                accumulated_transcript = ""
                vad_service.reset()
                if session is not None:
                    # Defensive -- shouldn't happen (silence_detected/
                    # stop_listening always clear it), but avoid leaking a
                    # stray streaming session's cache/attention-mode state.
                    asyncio.create_task(session.finish())
                session = transcription_service.start_session()
                pending_partial_task = None
                print("Started listening")
                continue

            if msg_type == "stop_listening":
                is_listening = False
                affirm_enabled = False
                accumulated_transcript = ""
                vad_service.reset()
                if session is not None:
                    # Cleanly release the streaming session's cache/
                    # attention-mode state instead of just dropping the
                    # reference; result is unused, this is cleanup only.
                    asyncio.create_task(session.finish())
                    session = None
                pending_partial_task = None
                print("Stopped listening")
                continue
            
            if msg_type == "start_dialogue":
                script = control.get("script", {})
                turns = script.get("turns", [])
                script_id = script.get("id")
                cache_audio_locally = script.get("cacheAudioLocally", True)
                affirmation_enabled = script.get("affirmationUtterancesEnabled", False)
                repeat_enabled = script.get("repeatDetectionEnabled", False)
                settings = control.get("settings", {})
                tts_engine = settings.get("ttsEngine", "kokoro")
                tts_voice = settings.get("ttsVoice", "am_adam")
                language = settings.get("language") or "english"
                lang_lower = language.lower()

                current_engine = tts_engine
                current_voice = tts_voice
                current_language = lang_lower
                current_repeat_enabled = repeat_enabled

                turns_cached = cache_audio_locally and script_id and _all_turns_cached(script_id, turns, tts_engine, tts_voice, lang_lower)
                # Backchannel pools are always persistent (see prepare_backchannel_pools),
                # so their cache readiness doesn't depend on cache_audio_locally.
                backchannel_ready = _backchannel_pools_cached(tts_engine, tts_voice, lang_lower, affirmation_enabled, repeat_enabled)

                if turns_cached and backchannel_ready:
                    print("[Cache] All turns + backchannel pools cached, skipping TTS model load")
                    clips = load_cached_clips(script_id, turns, tts_engine, tts_voice, lang_lower)
                    await sendClips(websocket, clips)
                    continue

                dialogue_service = DialogueService(engine=tts_engine, language=language)

                if not backchannel_ready:
                    await dialogue_service.prepare_backchannel_pools(
                        voice=tts_voice,
                        affirmation_enabled=affirmation_enabled,
                        repeat_enabled=repeat_enabled,
                    )

                clips = await dialogue_service.prepare(
                    turns,
                    voice=tts_voice,
                    script_id=script_id,
                    cache_audio_locally=cache_audio_locally,
                )
                await sendClips(websocket, clips)
                continue

            if msg_type == "check_cache":
                scripts = control.get("scripts", [])
                results = [
                    {"id": script.get("id"), "cached": _script_is_cached(script)}
                    for script in scripts
                    if script.get("id")
                ]
                await websocket.send(json.dumps({
                    "type": "cache_status",
                    "results": results,
                }))
                continue


        elif isinstance(message, bytes):
            if not is_listening:
                continue

            events = vad_service.process_pcm(message)

            for event in events:
                if event["type"] == "partial_ready":
                    # Fire-and-forget: transcription runs in the background
                    # (see handle_partial) so it can never delay thinking_pause
                    # or silence_detected below, which are driven by VAD alone.
                    pending_partial_task = asyncio.create_task(handle_partial(event["audio"]))

                elif event["type"] == "thinking_pause":
                    # Mid-utterance pause, not end-of-turn — only inject a
                    # backchannel if this line's [affirm] tag asked for one and
                    # a clip is already pre-generated (never generate live here,
                    # that would block the listening loop on model inference).
                    # Purely VAD-timed -- does not wait on transcription at all.
                    if affirm_enabled and current_engine and current_voice:
                        audio_bytes = pick_backchannel_audio(current_engine, current_voice, current_language)
                        print(f"[Backchannel] thinking_pause -> {'sending clip' if audio_bytes else 'no clip cached'}", flush=True)
                        if audio_bytes:
                            await websocket.send(json.dumps({
                                "type": "backchannel",
                                "audio": base64.b64encode(audio_bytes).decode("utf-8")
                            }))
                    else:
                        print(f"[Backchannel] thinking_pause (affirm_enabled={affirm_enabled}) -> skipped", flush=True)

                elif event["type"] == "silence_detected":
                    # Each partial_ready's feed() call is a required,
                    # in-order audio increment on the one streaming session
                    # for this utterance now (not a redundant whole-buffer
                    # transcription like before) -- it can't be cancelled or
                    # skipped without permanently losing that chunk from the
                    # transcript. Await it so finish() only has to process
                    # whatever sliver of audio arrived since, which is why
                    # this stays fast despite no longer racing ahead of it.
                    if pending_partial_task is not None:
                        await pending_partial_task
                        pending_partial_task = None

                    final_audio = event.get("audio", b"")
                    if session is not None:
                        t0 = time.monotonic()
                        final_text = await session.finish(final_audio)
                        elapsed = time.monotonic() - t0
                        session = None
                        if final_text.strip():
                            print(f"[Transcribe] final ({elapsed:.2f}s) -> {final_text!r}", flush=True)
                            accumulated_transcript = final_text.strip()
                            try:
                                await websocket.send(json.dumps({
                                    "type": "transcription_partial",
                                    "text": accumulated_transcript
                                }))
                            except websockets.ConnectionClosed:
                                pass
                        else:
                            print(f"[Transcribe] final ({elapsed:.2f}s) -> (empty)", flush=True)

                    print(f"[VAD] silence_detected handling -- accumulated_transcript={accumulated_transcript!r}", flush=True)
                    if current_repeat_enabled and is_repeat_request(accumulated_transcript, current_language):
                        # Not a real answer -- don't advance the dialogue or end the
                        # experiment. Tell Unity to (optionally) play an ack phrase and
                        # replay the current turn's clip, which it already has cached
                        # locally from the original prepare step.
                        print(f"\n[Repeat] Detected repeat request: {accumulated_transcript!r}")
                        ack_audio = None
                        if current_engine and current_voice:
                            ack_audio = pick_repeat_response_audio(current_engine, current_voice, current_language)

                        await websocket.send(json.dumps({
                            "type": "repeat_requested",
                            "audio": base64.b64encode(ack_audio).decode("utf-8") if ack_audio else None
                        }))

                        is_listening = False
                        vad_service.reset()
                        accumulated_transcript = ""
                    else:
                        await websocket.send(json.dumps({
                            "type": "transcription_done"
                        }))

                        if is_last:
                            await websocket.send(json.dumps({
                                "type": "experiment_done"
                            }))

                        is_listening = False
                        is_last = False
                        vad_service.reset()
                        accumulated_transcript = ""


                    


async def main():
    async with websockets.serve(handle, "localhost", 8765):
        print("voxPersona-service running on localhost:8765")
        await asyncio.Future()

asyncio.run(main())