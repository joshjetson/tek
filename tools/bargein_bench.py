#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barge-in, measured: does echo alone ever stop a reply, and does a voice always?

The number that matters is the FALSE-STOP RATE - how often TEK interrupts itself
when nobody has said anything. A detector that never misses but stops the reply
twice an evening is worse than no detector at all, because the failure it
creates is the one it was built to remove.

    tools/bargein_bench.py                 # synthetic sweep, no hardware
    tools/bargein_bench.py --hours 1       # the real room, through the speaker

The synthetic sweep is not a substitute for the room - it has no reverb, no
codec and no Bluetooth clock drift - but it is where a sign error or an
off-by-one in the alignment shows up instantly, and it runs anywhere.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import numpy as np                                      # noqa: E402

from tekdromo.voice import bargein, pcm                 # noqa: E402

RATE = pcm.RATE


def _frames(x, n=pcm.FRAME):
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


def _speechlike(secs, seed, f0=180.0):
    """Non-stationary, gapped, harmonically simple - enough structure to
    correlate against, which flat noise is not."""
    rs = np.random.RandomState(seed)
    t = np.arange(int(RATE * secs)) / float(RATE)
    x = (0.35 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
         + 0.18 * np.sin(2 * np.pi * (f0 * 2.3) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)))
    # Stop consonants: brief gaps that a naive "reset on quiet" would trip on.
    for _ in range(int(secs * 3)):
        s = rs.randint(0, max(1, len(x) - 400))
        x[s:s + rs.randint(80, 400)] *= 0.05
    return x.astype(np.float32)


def trial(voice_at=None, lag_s=0.18, echo_gain=0.55, noise=0.004,
          secs=5.0, seed=0, voice_gain=0.30):
    reply = _speechlike(secs, seed)
    ref = bargein.Reference(rate=RATE, latency=0.0)
    det = bargein.Detector(ref)
    ref.t0 = time.monotonic()
    det.started = ref.t0 - bargein.WARMUP_S      # the harness starts mid-reply
    for f in _frames(pcm.from_float(reply)):
        ref.write(f)

    lag = int(lag_s * RATE)
    mic = np.zeros(len(reply) + lag, dtype=np.float32)
    mic[lag:] += echo_gain * reply
    mic += np.random.RandomState(seed + 977).randn(len(mic)).astype(np.float32) * noise
    if voice_at is not None:
        s = int(voice_at * RATE)
        mic[s:] += voice_gain * _speechlike(
            (len(mic) - s) / float(RATE), seed + 31, f0=125.0)

    for i, f in enumerate(_frames(pcm.from_float(mic))):
        if det.feed(f, ref.t0 + i * pcm.FRAME_MS / 1000.0):
            return i * pcm.FRAME_MS / 1000.0, det.state()
    return None, det.state()


def sweep():
    print("Echo only - ANY fire here is a false stop")
    print("  %-9s %-9s %-8s %-7s %s" % ("lag", "echo", "noise", "fired", "state"))
    false_stops = trials = 0
    for lag in (0.05, 0.12, 0.18, 0.25, 0.32):
        for gain in (0.35, 0.55, 0.80):
            for noise in (0.002, 0.008, 0.020):
                for seed in range(3):
                    t, st = trial(voice_at=None, lag_s=lag, echo_gain=gain,
                                  noise=noise, seed=seed)
                    trials += 1
                    if t is not None:
                        false_stops += 1
                        print("  %-9.2f %-9.2f %-8.3f %-7.2f %s"
                              % (lag, gain, noise, t, st))
    print("  false stops: %d / %d trials" % (false_stops, trials))

    print()
    print("Echo + a NEAR voice at 2.0s - a MISS here is a failure to notice")
    misses = trials2 = 0
    lats = []
    for lag in (0.05, 0.18, 0.32):
        for gain in (0.35, 0.55, 0.80):
            for vg in (0.30, 0.50):                 # loud: someone addressing it
                for seed in range(3):
                    t, st = trial(voice_at=2.0, lag_s=lag, echo_gain=gain,
                                  seed=seed, voice_gain=vg)
                    trials2 += 1
                    if t is None or t < 2.0:
                        misses += 1
                    else:
                        lats.append(t - 2.0)
    print("  missed: %d / %d trials" % (misses, trials2))
    if lats:
        lats.sort()
        print("  time to notice: median %.0f ms, p90 %.0f ms  (hold is %.0f ms)"
              % (lats[len(lats) // 2] * 1000, lats[int(len(lats) * 0.9)] * 1000,
                 bargein.HOLD_S * 1000))

    # The proximity gate, which is the whole reason a family home is a harder
    # environment than a quiet one. A child downstairs IS a second voice, so
    # voice-presence alone is not the question - "is it aimed at me" is.
    print()
    print("Echo + a DISTANT voice at 2.0s - a STOP here is the wrong behaviour")
    wrong = trials3 = 0
    for lag in (0.05, 0.18, 0.32):
        for gain in (0.35, 0.55, 0.80):
            for vg in (0.06, 0.12):                 # quiet: elsewhere in the house
                for seed in range(3):
                    t, _ = trial(voice_at=2.0, lag_s=lag, echo_gain=gain,
                                 seed=seed, voice_gain=vg)
                    trials3 += 1
                    if t is not None:
                        wrong += 1
    print("  interrupted for a distant voice: %d / %d trials" % (wrong, trials3))
    return false_stops, misses + wrong


def live(hours):
    """Speak into the real room for `hours` and count self-interruptions.

    Nobody should talk during this. Every barge-in it reports is a false stop -
    the detector deciding a person is present when the only thing in the room
    is the reply, the speaker, and whatever the house is doing.
    """
    from . import bus                                   # noqa: F401
    import socket
    from tekdromo.voice import bus as vbus
    lines = ["The quick brown fox jumps over the lazy dog, and keeps going.",
             "Sunlight scatters off air molecules, and the short blue "
             "wavelengths scatter most, which is why the sky looks the way "
             "it does on a clear afternoon.",
             "Sixty degrees for the cylinder, and about seventy for the "
             "radiators, though it depends what the weather is doing."]
    c = vbus.Client(vbus.DEFAULT_PATH, timeout=300)
    t_end = time.time() + hours * 3600
    said = barges = 0
    base = c.request({"cmd": "status"}) or {}
    start_barges = base.get("barges", 0)
    print("speaking for %.2g hours - do NOT talk. ctrl-c to stop early." % hours)
    try:
        while time.time() < t_end:
            c.request({"cmd": "say", "text": lines[said % len(lines)]})
            said += 1
            st = c.request({"cmd": "status"}) or {}
            barges = st.get("barges", start_barges) - start_barges
            print("  %3d spoken, %d false stops (%.0f min left)"
                  % (said, barges, (t_end - time.time()) / 60))
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    c.close()
    print()
    print("FALSE-STOP RATE: %d in %d utterances" % (barges, said))
    return barges


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0.0,
                    help="run the live room test for this long instead")
    a = ap.parse_args()
    if a.hours:
        sys.exit(1 if live(a.hours) else 0)
    fs, ms = sweep()
    print()
    print("BARGEIN %s" % ("OK" if fs == 0 and ms == 0 else "FAIL"))
    sys.exit(0 if (fs == 0 and ms == 0) else 1)
