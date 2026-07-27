#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which wake phrases can this model actually hear?

A grammar entry containing a word the model has no pronunciation for is
SILENTLY DEAD - Vosk logs a warning to stderr and carries on, so the phrase
looks configured and can never match. That has already happened once on this
box, with "tekdromo".

Two things are checked per candidate:

  * does Vosk complain that a word is out of vocabulary;
  * does the grammar actually FIRE on that phrase spoken by Piper.

The second is the real test. Piper is not the user, so this measures whether a
phrase is recognisable at all, not how well it works across the room - but a
phrase that fails here cannot possibly work there.

    tools/wake_probe.py [phrase ...]
"""
import os
import sys
import tempfile

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lib")
if os.path.isdir(_LIB) and _LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np

from tekdromo.voice import pcm, stt, tts

CANDIDATES = [
    "hey tek", "hey tech", "ok tek", "ok tech",
    "hey take", "hey deck", "hey tec", "hey check", "hey tack",
    "okay tech", "hi tech", "hey tex", "hey tekk",
]


class Capture(object):
    """Grab what the C library writes to stderr - Python cannot see it."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryFile()
        self.saved = os.dup(2)
        os.dup2(self.tmp.fileno(), 2)
        return self

    def __exit__(self, *a):
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.tmp.seek(0)
        self.text = self.tmp.read().decode("utf-8", "replace")
        self.tmp.close()
        return False


def oov(phrase):
    """Words in this phrase the model has no pronunciation for."""
    with Capture() as cap:
        try:
            stt.Recogniser(grammar=json.dumps([phrase, "[unk]"]))
        except Exception:
            pass
    bad = []
    for line in cap.text.splitlines():
        if "not present" in line.lower() or "oov" in line.lower():
            bad.append(line.strip()[-60:])
    return bad


def main():
    phrases = sys.argv[1:] or CANDIDATES
    print("synthesising each phrase and asking a grammar built from it...\n")
    voice = tts.load()
    print("  %-12s %-6s %-8s  %s" % ("phrase", "OOV", "fires", "heard"))
    print("  " + "-" * 62)
    good = []
    for phrase in phrases:
        bad = oov(phrase)
        samples, rate = voice.synth(phrase)
        audio = pcm.resample(np.asarray(samples, np.int16), int(rate), pcm.RATE)
        rec = stt.Recogniser(grammar=json.dumps([phrase, "[unk]"]))
        heard = rec.transcribe(audio)
        fires = stt.heard_wake(heard) or phrase in heard
        if fires and not bad:
            good.append(phrase)
        print("  %-12s %-6s %-8s  %r%s"
              % (phrase, "YES" if bad else "-", "yes" if fires else "NO",
                 heard, ("   " + bad[0]) if bad else ""))

    print("\nusable: %s" % good)

    # And the one that matters: does the CURRENT grammar catch each of them?
    print("\nagainst the CONFIGURED grammar %s:" % (stt.WAKE_WORDS,))
    rec = stt.Recogniser(grammar=stt.WAKE_GRAMMAR)
    for phrase in phrases:
        samples, rate = voice.synth(phrase)
        audio = pcm.resample(np.asarray(samples, np.int16), int(rate), pcm.RATE)
        heard = rec.transcribe(audio)
        print("  saying %-12s -> %-14r %s"
              % (phrase, heard, "WAKES" if stt.heard_wake(heard) else "missed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
