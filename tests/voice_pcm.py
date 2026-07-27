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

# -- chunking for streaming speech ----------------------------------------
# These constants set how long you wait for the first word and whether the
# speech has holes in it. Both were found by measurement and both regressed
# once already, so they are pinned.
from tekdromo.voice import service

check("the first chunk is small (it sets time-to-first-word)",
      service.CHUNK_RAMP[0] <= 60, service.CHUNK_RAMP[0])
check("chunks grow slowly enough for the synthesis rate",
      service.GROWTH <= 1.20, service.GROWTH)
check("the head start is short", service.MIN_LEAD_S <= 2.0, service.MIN_LEAD_S)

for name, txt in (
        ("short", "Sure, done."),
        ("sentences", "One. Two three four. Five six seven eight nine ten."),
        ("one long sentence", "This is a long answer, without any sentence "
         "break at all, of the kind that used to be a single chunk so that "
         "nothing could be spoken until every last word of it had been "
         "synthesised first."),
        ("no punctuation", "words " * 60)):
    ch = service._chunks(txt)
    sizes = [len(c) for c in ch]
    check("%s: produces chunks" % name, len(ch) >= 1, sizes)
    check("%s: first chunk is small" % name,
          sizes[0] <= service.CHUNK_RAMP[0] + 5, sizes)
    # A chunk much bigger than the one before it starves playback: the audio
    # for chunk k runs out before chunk k+1 has finished synthesising.
    #
    # The ratio alone is too strict at small sizes. A chunk can never be
    # smaller than one atom, so a short opening clause followed by a longer
    # atom - 22 characters then 34 - breaks the ratio and cannot be avoided.
    # In absolute terms that is about 0.3 s of extra audio, which no amount of
    # lead is short of. So a jump is only a problem if it is BOTH out of ratio
    # AND large.
    bad = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
           if sizes[i + 1] > sizes[i] * (service.GROWTH + 0.05)
           and sizes[i + 1] - sizes[i] > 25]
    check("%s: no chunk badly outgrows the one before it" % name, not bad,
          "%s jumps %s" % (sizes, bad))
    check("%s: nothing is lost" % name,
          len(" ".join(ch).split()) == len(txt.split()),
          (len(" ".join(ch).split()), len(txt.split())))

check("empty text yields no chunks", service._chunks("") == [])

# -- streaming: chunks must come out BEFORE the reply is finished ----------
# This is what stops a deeper answer costing a longer silence in front of it.
# Answers were capped at one or two sentences partly because the whole reply
# had to exist before a single word could be spoken.
REPLY = ("It's Monday. Sunlight is made of every colour, and the blue part "
         "bounces off air molecules far more than the red. So when you look "
         "up, you are seeing blue light scattered at you from the whole sky. "
         "At sunset the light travels further and the blue is scattered away "
         "before it reaches you, which is why it goes red.")


def dribble(text, n=7):
    """The reply arriving a few characters at a time, as a model writes it."""
    for i in range(0, len(text), n):
        yield text[i:i + n]


said = {"text": ""}
gen = service._stream_chunks(dribble(REPLY), said)
first = next(gen)
check("a chunk is available before the whole reply has arrived",
      len(said["text"]) < len(REPLY), (len(said["text"]), len(REPLY)))
check("the first streamed chunk is a whole sentence",
      first == "It's Monday.", first)
rest = list(gen)
streamed = [first] + rest
check("streaming loses nothing",
      len(" ".join(streamed).split()) == len(REPLY.split()),
      (len(" ".join(streamed).split()), len(REPLY.split())))
check("streaming accumulates the full text for the log",
      said["text"] == REPLY)
sizes = [len(c) for c in streamed]
check("streamed chunks respect the ramp ceiling",
      max(sizes) <= service.CHUNK_RAMP[-1], sizes)


def underruns(sizes, rate=0.98, chars_per_sec=16.0):
    """Seconds of silence a listener would hear, given the chunk sizes.

    The per-pair growth ratio used for the non-streaming packer is the wrong
    invariant here. Streaming legitimately emits a short chunk whenever a short
    sentence completes ("It's Monday."), and the next chunk cannot be smaller
    than one atom, so the ratio is unsatisfiable by construction - while being
    completely harmless, because by then the producer is seconds ahead.

    What actually matters is whether the buffer ever runs dry. So simulate it:
    synthesis at the WORST measured rate (0.98x, under contention), playback
    starting once MIN_LEAD_S is buffered, and check that every chunk is ready
    before the previous one finishes playing.
    """
    audio = [n / chars_per_sec for n in sizes]
    ready, t = [], 0.0
    for a in audio:
        t += a * rate
        ready.append(t)
    lead = 0.0
    start = None
    for i, a in enumerate(audio):
        lead += a
        if lead >= service.MIN_LEAD_S:
            start = ready[i]
            break
    if start is None:
        return 0.0                      # whole reply buffered before playback
    gaps, playhead = 0.0, start
    for i, a in enumerate(audio):
        if ready[i] > playhead:
            gaps += ready[i] - playhead
            playhead = ready[i]
        playhead += a
    return gaps


gap = underruns(sizes)
check("streamed chunks never starve playback (simulated at 0.98x)",
      gap < 0.05, "%.2fs of silence, sizes %s" % (gap, sizes))
# The same simulation must also condemn a genuinely bad pattern, or it proves
# nothing: 30 characters followed by 300 is what the ramp exists to prevent.
check("the starvation check can actually fail",
      underruns([30, 300, 300]) > 0.5, underruns([30, 300, 300]))

# Text with no sentence end at all must still come out, not stall forever.
said2 = {"text": ""}
none_end = list(service._stream_chunks(dribble("words " * 60), said2))
check("a reply with no sentence ending still produces chunks",
      len(none_end) > 1, len(none_end))
check("and loses nothing", len(" ".join(none_end).split()) == 60,
      len(" ".join(none_end).split()))

# A model that says nothing must not produce a chunk.
said3 = {"text": ""}
check("an empty stream yields no chunks",
      list(service._stream_chunks(iter([]), said3)) == [])

check("the shortest speakable prefix is short enough to matter",
      service.MIN_SPEAK <= 15, service.MIN_SPEAK)

print("VOICE PCM " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
