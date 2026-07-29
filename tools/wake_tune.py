#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What does Vosk actually hear when THIS person says the wake word in THIS room?

`wake_probe.py` answers a different and weaker question - it has Piper say a
candidate phrase and checks the grammar fires. That proves a phrase is
recognisable in principle, which is necessary and nowhere near sufficient:
synthetic speech has no room, no accent, no distance and no noise. A wake word
that passes there can still never fire in the house, which is exactly the state
this box was found in - "[unk] [unk]" at peak 0.205, meaning the ear heard the
speaker loud and clear and matched nothing.

So this records the real person and decodes each attempt TWICE:

  * through the wake grammar, which can only emit its own phrases or [unk] -
    this is what the ear actually uses, and the pass/fail that matters;
  * through the FREE decoder, which has 200k words and will say what it thinks
    it heard - this is the diagnostic, because "hey tek" coming back as
    "hey tack" tells you what to add, and "[unk]" through both tells you the
    problem is level or segmentation rather than vocabulary.

    tools/wake_tune.py [--rounds 5] [--phrase "hey tek"]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lib")
if os.path.isdir(_LIB) and _LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LIB + ":" + os.environ.get(
        "LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import difflib                                          # noqa: E402

import numpy as np                                      # noqa: E402

from tekdromo.voice import bus, io as vio, pcm, stt      # noqa: E402


def say(text, wait=True):
    """Speak through the service if it is up; print otherwise."""
    try:
        c = bus.Client(bus.DEFAULT_PATH, timeout=120)
        c.request({"cmd": "say", "text": text, "wait": wait})
        c.close()
    except Exception:
        print("   (would say: %s)" % text)


def record(device, seconds):
    src = vio.MicSource(device=device)
    frames, t0 = [], time.time()
    while time.time() - t0 < seconds:
        f = src.read()
        if f is None:
            break
        frames.append(np.asarray(f))
    src.close()
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--phrase", default="hey tek")
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()

    device = vio.working_source()
    print("microphone: %s" % device)
    print("grammar:    %s" % ", ".join(stt.WAKE_WORDS))
    print()

    wake = stt.Recogniser(grammar=stt.WAKE_GRAMMAR)
    free = stt.Recogniser()

    say("Wake word test. When I ask, say %s, and nothing else. %d times."
        % (a.phrase, a.rounds))

    fired = 0
    heard = []
    for i in range(1, a.rounds + 1):
        say("Number %d. Say it now." % i)
        # 0.5s of dead time here clipped the leading "hey" on 2 of 5 attempts -
        # the grammar came back '[unk] tek', meaning the distinctive word
        # survived and the carrier did not. That is the tool truncating the
        # speaker, not the ear failing, and the two are indistinguishable in
        # the result. Kept short enough not to lose a fast reply, and the
        # window is long enough to cover a slow one.
        time.sleep(0.15)
        samples = record(device, a.seconds)
        # Report where the speech actually sat in the window, so a clipped
        # start is visible as a number rather than inferred from a bad decode.
        env = np.abs(samples.astype(np.int32)).astype(np.float32) / 32768.0
        loud = np.where(env > 0.05)[0]
        onset = (loud[0] / float(pcm.RATE)) if len(loud) else -1.0
        if not len(samples):
            print("  %d. no audio" % i)
            continue
        peak = float(np.abs(samples.astype(np.int32)).max()) / 32768.0
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        g = (wake.transcribe(samples) or "").strip()
        f = (free.transcribe(samples) or "").strip()
        ok = stt.heard_wake(g)
        fired += 1 if ok else 0
        if f:
            heard.append(f)
        sim = max([difflib.SequenceMatcher(None, f, w).ratio()
                   for w in stt.WAKE_WORDS] or [0.0])
        print("  %d. %-5s peak %.3f onset %+.2fs | grammar %-14r | free %-24r sim %.2f%s"
              % (i, "FIRED" if ok else "miss", peak, onset, g, f, sim,
                 "  <- CLIPPED START" if 0 <= onset < 0.12 else ""))

    print()
    print("  fired %d of %d" % (fired, a.rounds))
    if heard:
        print("  the free decoder heard: %s"
              % ", ".join(repr(h) for h in heard))
        # Anything the free decoder produced repeatedly, and that is close to a
        # wake phrase, is a candidate to ADD to the grammar - that is how
        # "hey deck" and "hey tex" got there.
        counts = {}
        for h in heard:
            counts[h] = counts.get(h, 0) + 1
        cands = [(n, h) for h, n in counts.items()
                 if max(difflib.SequenceMatcher(None, h, w).ratio()
                        for w in stt.WAKE_WORDS) >= 0.5]
        if cands:
            print("  candidates to add to WAKE_WORDS:")
            for n, h in sorted(cands, reverse=True):
                print("      %r  (heard %d/%d)" % (h, n, a.rounds))
    if fired == a.rounds:
        say("That fired every time.")
    elif fired:
        say("That fired %d times out of %d." % (fired, a.rounds))
    else:
        say("That did not fire at all. I have written down what I heard.")
    return 0 if fired else 1


if __name__ == "__main__":
    sys.exit(main())
