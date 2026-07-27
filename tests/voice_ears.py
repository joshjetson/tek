# -*- coding: utf-8 -*-
"""Continuous listening: the gate, the wake word, and the command path.

The expensive part (Vosk over real audio) is already covered by voice_stt.py.
What is under test here is the logic that decides whether something was said TO
us, because that is what stands between a useful device and one that:

  * answers its own replies forever - the mic hears the Bluetooth speaker at
    about 11x ambient and Vosk transcribes Piper perfectly, so this is not a
    theoretical risk, it is the default behaviour without a gate;
  * transcribes the household - full decoding must happen only after the wake
    word, which is the privacy posture the project committed to;
  * goes permanently deaf when someone bumps the USB plug - the microphone is
    physically part of the camera, and camera replugs are now routine.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo.voice import ears, io as vio, pcm

FAIL = []


def check(name, cond, extra=""):
    print("  %-56s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


class FakeService(object):
    def __init__(self):
        self.speaking = False
        self.events = []

    def on_event(self, ev):
        self.events.append(ev)
        return {"ok": True}


# -- the gate: it must not hear itself -------------------------------------
loud = pcm.from_float(np.ones(pcm.FRAME, np.float32) * 0.5)
svc = FakeService()
src = vio.ArraySource(np.tile(loud, 40), pcm.RATE)
g = ears.Gate(src, svc, tail=0.5)

f = g.read()
check("passes audio through when not speaking", pcm.envelope(f) > 0.4,
      pcm.envelope(f))

svc.speaking = True
f = g.read()
check("feeds silence while speaking", pcm.envelope(f) == 0.0, pcm.envelope(f))

svc.speaking = False
f = g.read()
check("stays muted immediately after speaking stops (reverb + A2DP tail)",
      pcm.envelope(f) == 0.0, pcm.envelope(f))

time.sleep(0.55)
f = g.read()
check("unmutes once the tail has elapsed", pcm.envelope(f) > 0.4,
      pcm.envelope(f))
check("it counted what it muted", g.muted == 2, g.muted)

# Frames must still be CONSUMED while muted, not left to pile up: parec keeps
# producing regardless, and a reader that stalls delivers a burst of stale
# audio later.
svc2 = FakeService()
src2 = vio.ArraySource(np.tile(loud, 10), pcm.RATE)
g2 = ears.Gate(src2, svc2, tail=10.0)
svc2.speaking = True
for _ in range(5):
    g2.read()
svc2.speaking = False
remaining = sum(1 for _ in src2)
check("muted frames are consumed, not queued up", remaining == 5, remaining)

check("the gate ends when the source ends",
      ears.Gate(vio.ArraySource(pcm.silence(0), pcm.RATE), svc).read() is None)


# -- the wake/command decision ---------------------------------------------
class FakeRec(object):
    """Stands in for a Recogniser. Returns whatever it was told to."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        return self.script.pop(0) if self.script else ""


def make_ears(wake_script, free_script):
    e = ears.Ears.__new__(ears.Ears)
    e.service = FakeService()
    e.window = 8.0
    e.armed_until = 0.0
    e.utterances = e.wakes = e.commands = e.opens = 0
    e.last_heard = None
    e._run = True
    e._t = None
    e.wake = FakeRec(wake_script)
    e.free = FakeRec(free_script)
    return e


DUMMY = pcm.silence(pcm.RATE)          # one second, contents irrelevant

# Ordinary conversation: the wake grammar returns junk, and the free decoder
# must NEVER be reached. This is the privacy guarantee, as a test.
e = make_ears(["[unk]"], ["this should never be transcribed"])
e._utterance(DUMMY)
check("no wake word -> nothing is reported", e.service.events == [])
check("no wake word -> the free recogniser is never even called",
      e.free.calls == 0, e.free.calls)

# Wake word and command in one breath - how people actually talk.
e = make_ears(["hey tek"], ["hey tek what time is it"])
e._utterance(DUMMY)
check("wake + command in one breath is dispatched immediately",
      len(e.service.events) == 1, e.service.events)
check("the wake word is stripped from the command",
      e.service.events[0].get("heard") == "what time is it",
      e.service.events[0].get("heard"))
check("the event is a speech event",
      e.service.events[0].get("kind") == "speech")
check("the transcript reaches the brain's prompt text",
      "what time is it" in e.service.events[0].get("what", ""))

# Wake word alone -> arm, then take the next utterance as the command.
e = make_ears(["hey tek"], ["hey tek", "turn the lights off"])
e._utterance(DUMMY)
check("wake word alone dispatches nothing yet", e.service.events == [])
check("wake word alone arms it", time.monotonic() < e.armed_until)
e._utterance(DUMMY)
check("the next utterance becomes the command",
      len(e.service.events) == 1
      and e.service.events[0].get("heard") == "turn the lights off",
      e.service.events)
check("it disarms after taking one command", e.armed_until == 0.0)
check("while armed it does not re-check the wake word",
      e.wake.calls == 1, e.wake.calls)

# An armed window that catches nothing usable must not fire a blank command.
e = make_ears(["hey tek"], ["hey tek", ""])
e._utterance(DUMMY)
e._utterance(DUMMY)
check("an empty follow-up does not dispatch an empty command",
      e.service.events == [], e.service.events)
check("and it disarms rather than staying armed forever",
      e.armed_until == 0.0)

# Counters, so `tek ears` reports something truthful.
e = make_ears(["[unk]", "hey tek"], ["hey tek hello there"])
e._utterance(DUMMY)
e._utterance(DUMMY)
check("counts every utterance, not just the ones that woke it",
      e.utterances == 2, e.utterances)
check("counts wakes and commands separately",
      (e.wakes, e.commands) == (1, 1), (e.wakes, e.commands))

# -- the free decoder mishearing the wake word -----------------------------
# Observed in the room: the grammar matched "hey tech" perfectly (it can only
# choose between four phrases) while the free decoder wrote down "hate tech",
# so the command dispatched was "hate tech what's up".
from tekdromo.voice import stt

check("an exact wake word is stripped",
      stt.strip_wake("hey tek what time is it") == "what time is it")
check("a misheard wake word is stripped too",
      stt.strip_wake("hate tech what's up") == "what's up",
      stt.strip_wake("hate tech what's up"))
check("a badly misheard wake word is stripped ('we tank' for 'hey tek')",
      stt.strip_wake("we tank is the microphone working")
      == "is the microphone working",
      stt.strip_wake("we tank is the microphone working"))
check("punctuation around the wake word does not defeat it",
      stt.strip_wake("Hey, tek, what time is it") == "what time is it",
      stt.strip_wake("Hey, tek, what time is it"))
# And it must not eat real words. "hey there" has no tek-like second token;
# "take the bins out" starts with a tek-like token but nothing preceded it, so
# only the leading-token rule could fire - it must not.
check("an ordinary command is left alone",
      stt.strip_wake("what time is it") == "what time is it")
# Openings that must survive. These are the ones measured closest to the
# threshold, so if it is ever loosened this is what breaks first.
for phrase in ("what time is it", "is the door locked", "how are you doing",
               "turn the lights off", "we should go outside",
               "tell me a joke", "play some music", "what day is it"):
    check("survives an ordinary opening: %r" % phrase[:22],
          stt.strip_wake(phrase) == phrase, stt.strip_wake(phrase))
check("a short phrase is never stripped to nothing",
      stt.strip_wake("hey tak") == "hey tak", stt.strip_wake("hey tak"))

# "Just the wake word", when the decoder mangled it. Observed live: someone
# said "hey tek" alone, the decoder wrote "hate tech", nothing stripped it
# because there was no command to strip from, and it was dispatched as the
# question - so it answered as though they had announced they dislike
# technology.
for phrase in ("hey tek", "hate tech", "ok tek", "hey tak", "HEY TECH.",
               "we tank", ""):
    check("recognises %r as the wake word alone" % phrase,
          stt.wake_only(phrase), phrase)
for phrase in ("what time is it", "hate technology in general",
               "turn the lights off", "tell me a joke"):
    check("does NOT mistake %r for a bare wake word" % phrase[:24],
          not stt.wake_only(phrase), phrase)

# The whole-string fuzzy test CANNOT separate a mangled wake word from a short
# greeting - "we tank" scores 0.57 and "hey there" 0.75 - so it is only ever
# consulted when strip_wake removed nothing. Pin the routing, since that is
# what makes the ambiguity harmless.
e = make_ears(["hey tek"], ["hey tek hello there"])
e._utterance(DUMMY)
check("a stripped remainder is dispatched even if it looks wake-ish",
      [ev.get("heard") for ev in e.service.events] == ["hello there"],
      e.service.events)

e = make_ears(["hey tech"], ["hate tech"])
e._utterance(DUMMY)
check("a mangled BARE wake word arms instead of being asked as a question",
      e.service.events == [] and time.monotonic() < e.armed_until,
      e.service.events)

# -- constants -------------------------------------------------------------
check("the speak tail covers A2DP latency and some reverb",
      ears.SPEAK_TAIL >= 0.5, ears.SPEAK_TAIL)
check("the wake window is long enough to draw breath but not to forget",
      3.0 <= ears.WAKE_WINDOW <= 15.0, ears.WAKE_WINDOW)

print("VOICE EARS " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
