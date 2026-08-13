import numpy as np
import torch
from silero_vad import load_silero_vad
import settings


class VADService:
    # Upper bound on how much audio a single partial re-transcribes. Without
    # this the speech_buffer grows for the whole utterance and every ~0.1s
    # micro-pause re-transcribes more and more audio from scratch, which
    # snowballs into multi-second stalls on a long turn (backend event loop
    # blocks on transcribe(), messages queue up, then all fire at once when
    # it finally catches up). 12s is generous for a scripted spoken answer
    # while keeping worst-case transcribe() cost bounded.
    MAX_BUFFER_SECONDS = 12.0

    def __init__(self, partial_threshold=0.1, thinking_pause_threshold=0.5):
        self.message_buffer = bytearray()
        self.speech_buffer = []
        # Frames not yet handed to the transcriber. Drained (not copied) on
        # every partial_ready/silence_detected so the caller only ever
        # transcribes new audio incrementally instead of re-transcribing the
        # whole growing utterance from scratch each time -- see
        # TranscriptionService.TranscriptionSession.
        self.unfed_frames = []
        self.was_speaking = False
        self.silence_timer = 0.0
        self.fired_partial = False
        self.vad_model = load_silero_vad()
        self.fired_silence = False
        self.fired_thinking = False
        self.partial_threshold = partial_threshold
        self.thinking_pause_threshold = thinking_pause_threshold
        self.pre_buffer = []
        self.max_buffer_frames = int(self.MAX_BUFFER_SECONDS / settings.VAD_FRAME_SECONDS)

    def reset(self):
        self.message_buffer = bytearray()
        self.speech_buffer = []
        self.unfed_frames = []
        self.was_speaking = False
        self.silence_timer = 0.0
        self.fired_partial = False
        self.fired_thinking = False
        self.pre_buffer = []
        # Silero VAD is stateful (RNN hidden state carried across calls on
        # the same model instance) -- without resetting it here, leftover
        # state from the tail of a previous utterance can misclassify the
        # first frames of the next one, which reads as random flakiness.
        self.vad_model.reset_states()

    def pcm16_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return audio_i16.astype(np.float32) / 32768.0

    def is_speech_silero(self, pcm_bytes: bytes, threshold: float = 0.5) -> bool:
        audio_f32 = self.pcm16_bytes_to_float32(pcm_bytes)
        tensor = torch.FloatTensor(audio_f32)
        confidence = self.vad_model(tensor, settings.SAMPLE_RATE).item()
        return confidence > threshold

    def process_pcm(self, pcm_bytes: bytes):
        events = []
        self.message_buffer.extend(pcm_bytes)

        while len(self.message_buffer) >= settings.FRAME_BYTES:
            frame = bytes(self.message_buffer[:settings.FRAME_BYTES])
            self.message_buffer = self.message_buffer[settings.FRAME_BYTES:]

            is_speaking = self.is_speech_silero(frame)

            if is_speaking:
                # was_speaking stays True through any pause shorter than the
                # full silence_detected threshold (only reset() clears it),
                # so it can't be used to tell "first speech of the utterance"
                # apart from "resuming after a mid-utterance pause". Use
                # silence_timer > 0 instead -- it's only nonzero here if we
                # just came out of a pause.
                resumed_from_pause = self.silence_timer > 0
                if not self.was_speaking:
                    print("[VAD] speech started", flush=True)
                    self.speech_buffer.extend(self.pre_buffer)
                    self.unfed_frames.extend(self.pre_buffer)

                # thinking_pause threshold crossed during the pause that
                # just ended, and the user is now still talking -- that
                # confirms it really was a mid-thought pause, not the end of
                # their turn, so it's safe to affirm now. Firing this
                # eagerly during the pause instead would fire on every
                # end-of-turn too, since silence always crosses 0.5s on its
                # way to the 1.5s silence_detected threshold -- there's no
                # way to tell those two cases apart until we see what
                # happens next.
                if resumed_from_pause and self.fired_thinking:
                    print("[VAD] speech resumed after thinking-pause -- confirmed mid-thought, firing affirmation", flush=True)
                    events.append({
                        "type": "thinking_pause"
                    })

                self.was_speaking = True
                self.silence_timer = 0.0
                self.fired_partial = False
                self.fired_thinking = False
                self.fired_silence = False
                self.speech_buffer.append(frame)
                self.unfed_frames.append(frame)

                if len(self.speech_buffer) > self.max_buffer_frames:
                    self.speech_buffer = self.speech_buffer[-self.max_buffer_frames:]

            else:

                self.pre_buffer.append(frame)

                if len(self.pre_buffer) > 8:
                     self.pre_buffer.pop(0)

                if self.was_speaking:
                    self.silence_timer += settings.VAD_FRAME_SECONDS

                    if self.silence_timer >= self.partial_threshold and not self.fired_partial:
                        self.fired_partial = True
                        buffer_seconds = len(self.speech_buffer) * settings.VAD_FRAME_SECONDS
                        print(f"[VAD] partial_ready (silence={self.silence_timer:.2f}s, buffer={buffer_seconds:.2f}s)", flush=True)
                        events.append({
                            "type": "partial_ready",
                            # Only the new audio since the last feed -- the
                            # transcriber is a persistent per-utterance stream
                            # (see TranscriptionSession), not a one-shot call
                            # over the whole buffer each time.
                            "audio": b"".join(self.unfed_frames)
                        })
                        self.unfed_frames = []

                    if self.silence_timer >= self.thinking_pause_threshold and not self.fired_thinking:
                        self.fired_thinking = True
                        print(f"[VAD] thinking-pause threshold crossed (silence={self.silence_timer:.2f}s) -- affirming only if speech resumes", flush=True)

                    if self.silence_timer >= 1.5 and not self.fired_silence:
                        self.fired_silence = True
                        print(f"[VAD] silence_detected -- utterance done (silence={self.silence_timer:.2f}s)", flush=True)
                        events.append({
                            "type": "silence_detected",
                            # Only whatever new audio hasn't been fed to the
                            # transcriber yet -- the session already holds
                            # everything transcribed so far incrementally,
                            # so finishing it just needs this last sliver,
                            # not the whole utterance again (see
                            # BackService.py's silence_detected handling).
                            "audio": b"".join(self.unfed_frames)
                        })
                        self.unfed_frames = []
                        self.was_speaking = False
                        self.reset()

        return events
