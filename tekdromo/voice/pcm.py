"""
The audio contract. ONE definition, imported by everything else.

16 kHz, mono, int16, 20 ms frames. Every stage in the voice pipeline speaks
exactly this, which is what lets a microphone, a WAV file and a synthesised
tone be the same kind of thing - and therefore lets the whole pipeline be
tested with no hardware attached.

Resampling happens ONLY at the hardware edges: Piper emits 22.05 kHz and the
Bluetooth sink runs at 44.1 kHz, but neither of those rates is ever allowed
inside the pipeline. The alternative - letting each stage carry its own rate -
means every pair of stages needs a conversion and every bug is a rate bug.

16 kHz is not arbitrary: it is the native rate of Vosk and of Whisper, so the
recognition path needs no conversion at all.
"""
import numpy as np

RATE = 16000
FRAME_MS = 20
FRAME = RATE * FRAME_MS // 1000        # 320 samples
DTYPE = np.int16
FULL = 32767.0

# 20 ms is not a free choice either: WebRTC's VAD accepts only 10, 20 or 30 ms
# frames, so picking anything else would mean a second framing just for it.


def silence(n=FRAME):
    return np.zeros(n, dtype=DTYPE)


def to_float(x):
    """int16 -> float32 in -1..1."""
    return x.astype(np.float32) / FULL


def from_float(x):
    """float -> int16, clipped. Clipping rather than scaling: a level that
    exceeds full scale is a bug upstream, and silently attenuating the whole
    utterance to hide it makes that bug much harder to find."""
    return (np.clip(x, -1.0, 1.0) * FULL).astype(DTYPE)


def frames(samples, n=FRAME, pad=True):
    """Yield fixed-size frames. The tail is zero-padded unless pad=False.

    Fixed size matters: the VAD rejects short frames outright, and a sink that
    receives ragged frames produces clicks at the joins.
    """
    total = len(samples)
    for i in range(0, total, n):
        f = samples[i:i + n]
        if len(f) < n:
            if not pad:
                return
            f = np.concatenate([f, np.zeros(n - len(f), dtype=samples.dtype)])
        yield f


def resample(x, src, dst):
    """Rate convert. Linear interpolation.

    Deliberately not a windowed-sinc: this runs on four A57s that are already
    rendering a face, and the only signals converted here are either headed for
    a speaker (where the 22.05->44.1 conversion is done by PulseAudio anyway) or
    headed for a recogniser trained on far worse audio than linear
    interpolation produces. Correctness here is not audibility, it is length.
    """
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * float(dst) / float(src)))
    if n <= 0:
        return x[:0]
    # endpoint=False: sample positions, not a closed interval. Using
    # endpoint=True stretches the last sample and drifts by one sample per
    # block, which accumulates into audible pitch error over a long utterance.
    pos = np.linspace(0, len(x), n, endpoint=False)
    out = np.interp(pos, np.arange(len(x)), x.astype(np.float32))
    return out.astype(x.dtype)


def envelope(frame):
    """RMS of a frame as 0..1. This is what drives the mouth.

    Feeds rig.speech.from_envelope(), which already exists and already expects
    exactly this - see its docstring: "Real audio replaces synthetic later".
    """
    if len(frame) == 0:
        return 0.0
    f = to_float(frame)
    return float(np.sqrt(np.mean(f * f)))
