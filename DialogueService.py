import numpy as np
import io
import soundfile as sf
import asyncio
from typing import Optional, List, Dict
import torch
import time
import os
import hashlib
import json
import random
from pathlib import Path

if torch.cuda.is_available():
    deviceGPU = "cuda"
elif torch.backends.mps.is_available():
    deviceGPU = "mps"
else:
    deviceGPU = "cpu"

CHATTERBOX_VOICES_DIR = Path(__file__).parent / "chatterbox_voices"

# Per-machine cache of generated TTS clips, keyed by a hash of everything that
# affects the audio (script id, turn index/text, engine, voice, language) so
# editing a script automatically invalidates just the turns that changed.
AUDIO_CACHE_DIR = Path(__file__).parent / "audio_cache"

# Affirmation/repeat-response phrases are a fixed, shared vocabulary (not
# script-specific dialogue), so they're cached by engine+voice+language only —
# any script using that combination reuses the same clips. See
# _backchannel_pools_cached / DialogueService.prepare_backchannel_pools.
BACKCHANNEL_CACHE_DIR = Path(__file__).parent / "backchannel_cache"

# 2-letter language code per full language name, shared between DialogueService
# (chatterbox multilingual arg) and the backchannel phrase pools below.
LANGUAGE_CODES = {
    "arabic": "ar",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "finnish": "fi",
    "french": "fr",
    "hebrew": "he",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "malay": "ms",
    "dutch": "nl",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "swedish": "sv",
    "swahili": "sw",
    "turkish": "tr",
    "chinese": "zh",
}

AFFIRMATION_PHRASES = {
    "en": ["mhm", "I see", "got it", "right", "okay"],
    "es": ["mhm", "ya veo", "entiendo", "claro", "vale"],
    "de": ["mhm", "ich verstehe", "ah ja", "genau", "okay"],
}

REPEAT_RESPONSE_PHRASES = {
    "en": ["Of course, what I said was...", "Sure, let me repeat that...", "No problem, I said..."],
    "es": ["Claro, lo que dije fue...", "Por supuesto, déjame repetir eso...", "No hay problema, dije..."],
    "de": ["Natürlich, was ich gesagt habe war...", "Klar, lass mich das wiederholen...", "Kein Problem, ich sagte..."],
}

# What the *participant* says to ask for a repeat (distinct from REPEAT_RESPONSE_PHRASES,
# which is what the agent says back). Substring match against the normalized transcript
# is deliberately simple/stem-based rather than exact-phrase, since users phrase this many
# ways ("wiederholen"/"wiederhole"/"wiederholst du" etc all match "wiederhol").
REPEAT_REQUEST_TRIGGERS = {
    "en": ["repeat", "say that again", "again?", "what did you say", "come again", "pardon",
           "one more time", "didn't catch that", "didn't hear", "say it again"],
    "es": ["repite", "puedes repetir", "otra vez", "qué dijiste", "perdón", "no escuché",
           "no entendí", "de nuevo", "cómo"],
    "de": ["wiederhol", "nochmal", "noch einmal", "wie bitte", "was hast du gesagt",
           "nicht verstanden", "kannst du das"],
}


def _audio_cache_key(script_id: str, index: int, text: str, engine: str, voice: str, language: str) -> str:
    raw = f"{script_id}|{index}|{engine}|{voice}|{language}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _audio_cache_paths(script_id: str, cache_key: str):
    script_cache_dir = AUDIO_CACHE_DIR / script_id
    return script_cache_dir / f"{cache_key}.wav"


def _resolve_backchannel_lang_code(language: str) -> str:
    return LANGUAGE_CODES.get((language or "english").lower(), "en")


def _backchannel_cache_path(engine: str, voice: str, lang_code: str, phrase: str) -> Path:
    phrase_hash = hashlib.sha256(phrase.encode("utf-8")).hexdigest()
    return BACKCHANNEL_CACHE_DIR / f"{engine}_{voice}_{lang_code}" / f"{phrase_hash}.wav"


def _backchannel_phrases(lang_code: str, affirmation_enabled: bool, repeat_enabled: bool) -> List[str]:
    phrases = []
    if affirmation_enabled:
        phrases.extend(AFFIRMATION_PHRASES.get(lang_code, AFFIRMATION_PHRASES["en"]))
    if repeat_enabled:
        phrases.extend(REPEAT_RESPONSE_PHRASES.get(lang_code, REPEAT_RESPONSE_PHRASES["en"]))
    return phrases


def _backchannel_pools_cached(engine: str, voice: str, language: str, affirmation_enabled: bool, repeat_enabled: bool) -> bool:
    if not (affirmation_enabled or repeat_enabled):
        return True
    if not engine or not voice:
        return False

    lang_code = _resolve_backchannel_lang_code(language)
    phrases = _backchannel_phrases(lang_code, affirmation_enabled, repeat_enabled)
    return all(_backchannel_cache_path(engine, voice, lang_code, phrase).exists() for phrase in phrases)


def pick_backchannel_audio(engine: str, voice: str, language: str) -> Optional[bytes]:
    """Returns the audio bytes for a random cached affirmation phrase, for live
    injection during a "thinking_pause" while listening to the user. Only picks
    from phrases already on disk — never generates on the fly, since that would
    block the listening loop with model inference."""
    lang_code = _resolve_backchannel_lang_code(language)
    candidates = [
        path for phrase in AFFIRMATION_PHRASES.get(lang_code, AFFIRMATION_PHRASES["en"])
        if (path := _backchannel_cache_path(engine, voice, lang_code, phrase)).exists()
    ]
    if not candidates:
        return None
    return random.choice(candidates).read_bytes()


def is_repeat_request(text: str, language: str) -> bool:
    """True if the participant's utterance looks like a request to repeat the
    last agent line. Deliberately a simple substring/stem match (see
    REPEAT_REQUEST_TRIGGERS) rather than an ML classifier -- these sessions are
    short scripted dialogues, not open-domain chat, so a keyword list covers
    the realistic phrasing without adding a model dependency."""
    if not text or not text.strip():
        return False
    normalized = text.lower().strip()
    lang_code = _resolve_backchannel_lang_code(language)
    triggers = REPEAT_REQUEST_TRIGGERS.get(lang_code, REPEAT_REQUEST_TRIGGERS["en"])
    return any(trigger in normalized for trigger in triggers)


def pick_repeat_response_audio(engine: str, voice: str, language: str) -> Optional[bytes]:
    """Same idea as pick_backchannel_audio, but pulls from the repeat-response
    acknowledgment pool ("Sure, let me repeat that...") instead of the
    affirmation pool. Also never generates on the fly -- only plays what's
    already cached."""
    lang_code = _resolve_backchannel_lang_code(language)
    candidates = [
        path for phrase in REPEAT_RESPONSE_PHRASES.get(lang_code, REPEAT_RESPONSE_PHRASES["en"])
        if (path := _backchannel_cache_path(engine, voice, lang_code, phrase)).exists()
    ]
    if not candidates:
        return None
    return random.choice(candidates).read_bytes()


def _all_turns_cached(script_id: str, turns, engine: str, voice: str, language: str) -> bool:
    agent_turns = [t for t in turns if t.get("type") == "agent"]
    if not script_id or not agent_turns:
        return False

    for index, turn in enumerate(agent_turns):
        cache_key = _audio_cache_key(script_id, index, turn.get("text", ""), engine, voice, language)
        wav_path = _audio_cache_paths(script_id, cache_key)
        if not wav_path.exists():
            return False

    return True


def load_cached_clips(script_id: str, turns, engine: str, voice: str, language: str) -> List[Dict]:
    """Build the same clip dicts prepare() would produce, reading purely from
    audio_cache/ without touching any TTS model. Only call after
    _all_turns_cached confirms every agent turn is present."""
    clips = []
    agent_turns = [t for t in turns if t.get("type") == "agent"]
    for index, turn in enumerate(agent_turns):
        text = turn.get("text", "")
        cache_key = _audio_cache_key(script_id, index, text, engine, voice, language)
        wav_path = _audio_cache_paths(script_id, cache_key)
        clips.append({
            "text": text,
            "audio": wav_path.read_bytes(),
            "index": len(clips)
        })
    return clips

# Per-language-code default voice (base filename, no extension).
# Extend this as more languages/voices get sample files.
CHATTERBOX_DEFAULT_VOICE = {
    "de": "liam",
}

def resolve_chatterbox_voice_sample(lang_code: str, voice: str) -> str:
    lang_dir = CHATTERBOX_VOICES_DIR / lang_code
    if not lang_dir.is_dir():
        raise FileNotFoundError(
            f"No chatterbox voice samples found for language code '{lang_code}' "
            f"(expected directory {lang_dir})"
        )

    if voice in (None, "", "default"):
        voice_name = CHATTERBOX_DEFAULT_VOICE.get(lang_code)
        if voice_name is None:
            raise FileNotFoundError(
                f"No default chatterbox voice configured for language code '{lang_code}'"
            )
    else:
        voice_name = voice

    sample_path = lang_dir / f"{voice_name}.mp3"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Chatterbox voice sample not found: {sample_path}")

    return str(sample_path)

class DialogueService:
    def __init__(
        self,
        engine: str = "kokoro",
        emotion: float = 0.5,
        voice_sample: Optional[str] = None,
        language: str = "english"
    ):

        self.engine = engine
        self.emotion = emotion
        self.voice_sample = voice_sample
        self.model = None
        self.sr = 24000
        self.language = language.lower()

        self.LANGUAGES = LANGUAGE_CODES


        if engine == "kokoro" and self.language == "english":
            from kokoro import KPipeline
            self.model = KPipeline(lang_code='a')



        elif engine == "chatterbox" and self.language == "english":
            import torchaudio as ta
            import torch
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            
            self.model = ChatterboxTurboTTS.from_pretrained(device=deviceGPU)


            # ta.save("test-turbo.wav", wav, model.sr)


        elif engine == "chatterbox" and self.language != "english":
            import torchaudio as ta
            from chatterbox.tts import ChatterboxTTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.model = ChatterboxMultilingualTTS.from_pretrained(device=deviceGPU)
            
        
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'kokoro' or 'chatterbox'")



    
    def _generateAudio(self, text: str, voice: str = 'default') -> dict:
        if self.model is None:
            return {"text": text, "audio": b""}

        if self.engine == "kokoro":
            audio_chunks = []
            if voice == "default": 
                voice = "am_adam"

            for i, (gs, ps, audio) in enumerate(
                self.model(text, voice=voice, speed=1)
            ):
                audio_chunks.append(audio)

            if not audio_chunks:
                return {"text": text, "audio": b""}

            combined = np.concatenate(audio_chunks)

            buffer = io.BytesIO()
            sf.write(buffer, combined, self.sr, format='WAV')
            buffer.seek(0)
            wav_bytes = buffer.read()

            return {"text": text, "audio": wav_bytes}

        elif self.engine == "chatterbox":
            # Get the language code, if none, choose english as fallback
            lang_code = self.LANGUAGES.get(self.language, "en")
            sample_path = self.voice_sample or resolve_chatterbox_voice_sample(lang_code, voice)

            if self.language == "english":
                wav = self.model.generate(
                        text,
                        audio_prompt_path=sample_path
                )
            else:
                wav = self.model.generate(
                    text,
                    audio_prompt_path=sample_path,
                    language_id=lang_code
                )


            buffer = io.BytesIO()
            import torchaudio as ta
            ta.save(buffer, wav, self.model.sr, format="wav")
            buffer.seek(0)
            wav_bytes = buffer.read()

            return {"text": text, "audio": wav_bytes}


    
            
        else:
            raise ValueError(f"Unknown engine: {self.engine}. Use 'kokoro' or 'orpheus'")

    async def runTTS(self, text: str, voice: str = 'default') -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generateAudio, text, voice)
    
    def __del__(self):
        if hasattr(self, 'vllm_process') and self.vllm_process:
            self.vllm_process.terminate()
            print("vLLM server stopped")

    async def prepare(self, turns, voice: str = 'leo', script_id: Optional[str] = None, cache_audio_locally: bool = True):
        clips = []

        print(f"\n[TTS] Processing {len(turns)} turns")
        print(f"[TTS] Engine: {self.engine}")
        print(f"[TTS] Voice: {voice}")

        agent_index = 0
        for turn in turns:
            if turn.get("type") != "agent":
                continue

            text = turn.get("text", "")
            index = agent_index
            agent_index += 1

            print("\n" + "=" * 60)
            print(f"[TTS] Clip #{len(clips)}")
            print(f"[TTS] Text length: {len(text)}")
            print(f"[TTS] Text: {text}")

            cache_key = _audio_cache_key(script_id, index, text, self.engine, voice, self.language) if script_id else None
            wav_path = _audio_cache_paths(script_id, cache_key) if cache_key else None

            if cache_audio_locally and wav_path and wav_path.exists():
                print(f"[TTS] Cache hit, skipping generation")
                audio_bytes = wav_path.read_bytes()
            else:
                start = time.perf_counter()

                audio = await self.runTTS(text, voice)

                elapsed = time.perf_counter() - start

                print(f"[TTS] Generated in {elapsed:.2f}s")
                print(f"[TTS] Audio size: {len(audio['audio']) / 1024:.1f} KB")

                audio_bytes = audio["audio"]

                if cache_audio_locally and wav_path:
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    wav_path.write_bytes(audio_bytes)

            clips.append({
                "text": text,
                "audio": audio_bytes,
                "index": len(clips)
            })

        print(f"\n[TTS] Finished. Generated {len(clips)} clips.\n")

        return clips

    async def prepare_backchannel_pools(self, voice: str, affirmation_enabled: bool, repeat_enabled: bool):
        """Pre-generates and caches the affirmation/repeat-response phrase pools for
        this engine+voice+language combination. Always persists to BACKCHANNEL_CACHE_DIR
        regardless of any single script's cache_audio_locally setting, since the phrases
        are shared vocabulary reusable by any script on the same voice (see the module
        docstring above BACKCHANNEL_CACHE_DIR)."""
        lang_code = _resolve_backchannel_lang_code(self.language)
        phrases = _backchannel_phrases(lang_code, affirmation_enabled, repeat_enabled)

        for phrase in phrases:
            wav_path = _backchannel_cache_path(self.engine, voice, lang_code, phrase)
            if wav_path.exists():
                continue

            print(f"[Backchannel] Generating {phrase!r} ({self.engine}/{voice}/{lang_code})")
            audio = await self.runTTS(phrase, voice)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(audio["audio"])
        