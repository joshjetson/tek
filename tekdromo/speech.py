"""
Mouth drive signals.

Previously this lived in tekfdl alongside a whole precomputed pose table. The
rig's mouth region supersedes that machinery, but the drive signal itself is
still wanted - and when the runner was rewritten it was dropped entirely, so
the face silently stopped talking. Keeping it as its own tiny module makes it
obvious whether anything is driving the mouth.

Real audio replaces `synthetic` later: feed an RMS envelope into
Face.speak(openness, rounding) and the rest is unchanged.
"""
import math


def synthetic(t):
    """Syllable bursts with pauses. Returns (openness, rounding).

    A steady open/close cycle reads as a puppet. What sells it is irregularity:
    3-6 syllables at ~4Hz, a wandering syllable rate, per-syllable amplitude
    variation, and gaps between "words".
    """
    word = 2.35
    ph = (t % word) / word
    if ph < 0.10:
        gate = 0.5 - 0.5 * math.cos(math.pi * ph / 0.10)
    elif ph < 0.62:
        gate = 1.0
    else:
        gate = max(0.0, 1.0 - (ph - 0.62) / 0.10)
    f = 4.1 + 0.8 * math.sin(t * 0.63)
    env = abs(math.sin(math.pi * f * t)) ** 0.65
    amp = 0.52 + 0.30 * math.sin(t * 2.17) + 0.16 * math.sin(t * 5.31)
    rounding = 0.55 * math.sin(t * 1.43) + 0.25 * math.sin(t * 3.7)
    return gate * env * max(0.15, amp), rounding


def from_envelope(rms, floor=0.02, ceiling=0.35):
    """Map an audio RMS level to (openness, rounding).

    Rounding is left at 0: real viseme shape needs phoneme information, which an
    envelope does not carry. Better a neutral mouth than a wrong one.
    """
    if rms <= floor:
        return 0.0, 0.0
    x = min(1.0, (rms - floor) / max(ceiling - floor, 1e-6))
    return x ** 0.7, 0.0
