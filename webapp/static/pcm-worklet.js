// Runs on the audio render thread (not the main thread), so mic capture
// stays smooth even while the page is busy. Resamples from whatever rate
// the browser's AudioContext actually runs at (commonly 44.1/48kHz, not
// guaranteed to honor a requested rate) down to the 16kHz mono PCM16 that
// core/VADService.py and core/TranscriptionService.py expect -- same
// format core/RecordingService.py captured locally for the old TUI, just
// produced in the browser instead of via sounddevice on the server.
class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const targetRate = options.processorOptions?.targetRate ?? 16000;
    this.ratio = sampleRate / targetRate;
    this.carry = 0;
    this.pending = [];
    // ~100ms per chunk, matching RecordingService.BLOCKSIZE's tuning --
    // frequent enough for responsive VAD without flooding the socket.
    this.chunkSamples = Math.round(targetRate * 0.1);
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    let pos = this.carry;
    while (pos < channel.length) {
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, channel.length - 1);
      const frac = pos - i0;
      this.pending.push(channel[i0] + (channel[i1] - channel[i0]) * frac);
      pos += this.ratio;
    }
    this.carry = pos - channel.length;

    while (this.pending.length >= this.chunkSamples) {
      const chunk = this.pending.splice(0, this.chunkSamples);
      const pcm16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        pcm16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
