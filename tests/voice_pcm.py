# -*- coding: utf-8 -*-
"""Invariants of the audio contract and the phonemiser.

These are the things every other stage assumes. If framing or resampling is
wrong, the failure shows up much later as clicks, drift or a mouth that runs
ahead of the sound, and looks like a bug in whatever stage noticed first.
"""
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo.voice import pcm, phonemes

FAIL = []


def check(name, cond, extra=""):
    print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


# -- framing ---------------------------------------------------------------
x = np.arange(1000, dtype=np.int16)
fs = list(pcm.frames(x))
check("frames are all exactly FRAME long",
      all(len(f) == pcm.FRAME for f in fs), [len(f) for f in fs])
check("framing covers every sample (pads the tail)",
      len(fs) * pcm.FRAME >= len(x) and len(fs) == 4, len(fs))
check("first frame is the head of the signal", np.array_equal(fs[0], x[:pcm.FRAME]))
check("tail is zero-padded, not wrapped", fs[-1][-1] == 0)
check("pad=False drops the ragged tail instead",
      len(list(pcm.frames(x, pad=False))) == 3)
check("20ms at 16kHz is 320 samples", pcm.FRAME == 320)

# -- float round trip ------------------------------------------------------
r = pcm.from_float(pcm.to_float(x))
check("int16 -> float -> int16 round trips", np.array_equal(r, x))
check("float conversion clips rather than wraps",
      pcm.from_float(np.array([2.0, -2.0], np.float32)).tolist() == [32767, -32767])

# -- resampling ------------------------------------------------------------
# Length is the property everything else depends on: a wrong length desyncs
# the mouth from the audio and, over a long utterance, shifts pitch.
for src, dst in ((22050, 16000), (16000, 22050), (44100, 16000), (16000, 16000)):
    n = 22050
    out = pcm.resample(np.zeros(n, np.int16), src, dst)
    want = int(round(n * float(dst) / src))
    check("resample %d->%d gives the right length" % (src, dst),
          abs(len(out) - want) <= 1, "%d vs %d" % (len(out), want))
check("resample of empty input is empty", len(pcm.resample(np.zeros(0, np.int16), 22050, 16000)) == 0)

# A sine keeps its frequency: catches an off-by-one in the sample positions
# that a length check alone would miss.
t = np.arange(16000, dtype=np.float32) / 16000.0
sine = pcm.from_float(0.5 * np.sin(2 * np.pi * 220.0 * t))
up = pcm.resample(sine, 16000, 22050)
zc_in = np.sum(np.diff(np.signbit(sine.astype(np.int32))) != 0)
zc_out = np.sum(np.diff(np.signbit(up.astype(np.int32))) != 0)
check("resampling preserves frequency (zero crossings)",
      abs(int(zc_in) - int(zc_out)) <= 2, "%d vs %d" % (zc_in, zc_out))

# -- envelope --------------------------------------------------------------
check("envelope of silence is 0", pcm.envelope(pcm.silence()) == 0.0)
loud = pcm.from_float(np.ones(pcm.FRAME, np.float32) * 0.5)
check("envelope of a 0.5 signal is ~0.5", abs(pcm.envelope(loud) - 0.5) < 0.01)
check("envelope is monotonic in level",
      pcm.envelope(loud) > pcm.envelope(pcm.from_float(
          np.ones(pcm.FRAME, np.float32) * 0.1)))
check("envelope of an empty frame is 0", pcm.envelope(pcm.silence(0)) == 0.0)

# -- phonemiser ------------------------------------------------------------
# The whole Piper path rests on espeak-ng's IPA matching the model's map. This
# was the highest-rated risk in the plan and turned out to be a non-issue, so
# it is pinned here: if a future espeak upgrade breaks it, this fails loudly
# rather than the voice quietly degrading.
import json
cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "voices", "en_US-lessac-medium.onnx.json")
if os.path.exists(cfg_path):
    id_map = json.load(open(cfg_path))["phoneme_id_map"]
    text = ("Hello, world. The quick brown fox jumps over the lazy dog; "
            "you should hear this clearly!")
    ph = phonemes.ipa(text)
    ids, missing = phonemes.to_ids(ph, id_map)
    check("espeak IPA is fully covered by the model's phoneme map",
          missing == 0, "%d unmapped" % missing)
    check("punctuation survives phonemisation (pause cues)",
          all(p in ph for p in ",.;!"), repr(ph[:60]))
    check("ids are bracketed with BOS/EOS",
          ids[0] == phonemes.BOS and ids[-1] == phonemes.EOS)
    check("ids are interleaved with PAD", ids[2] == phonemes.PAD)
    check("empty text yields just BOS+EOS",
          phonemes.to_ids("", id_map)[0] == [phonemes.BOS, phonemes.EOS])
    check("rounded vowels score higher than unrounded",
          phonemes.rounding(phonemes.ipa("boot moon")) >
          phonemes.rounding(phonemes.ipa("cat hat")))
else:
    print("  (piper model absent - skipping phonemiser checks)")

print("VOICE PCM " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
