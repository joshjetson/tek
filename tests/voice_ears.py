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
    e.misses = []
    e._gate = None
    e._drain = None
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
# A miss is recorded so "sometimes it says nothing" can be diagnosed at all.
# What is kept can only ever be the grammar's own output, which is physically
# incapable of containing anything but the wake phrases and "[unk]".
check("a near miss is recorded for diagnosis", len(e.misses) == 1, e.misses)
check("the miss records what the grammar returned",
      e.misses[0]["got"] == "[unk]", e.misses)
check("the miss records the level, to tell 'too quiet' from 'misheard'",
      "peak" in e.misses[0] and "secs" in e.misses[0], e.misses)
for i in range(30):
    e._utterance(DUMMY)
check("misses are bounded, not a leak", len(e.misses) <= 12, len(e.misses))

# Wake word and command in one breath - how people actually talk.
# "hey tek [unk]" is what the grammar really returns when a command follows -
# the bare form means nothing came after it.
e = make_ears(["hey tek [unk]"], ["hey tek what time is it"])
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
e = make_ears(["hey tek"], ["turn the lights off"])
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

# Saying "hey tek" twice - normal when the first seemed unheard - must not
# become a question about the words "hey tek".
e = make_ears(["hey tek"], ["hey tek"])
e._utterance(DUMMY)
e._utterance(DUMMY)
check("a repeated wake word is not dispatched as the question",
      e.service.events == [], e.service.events)

# An armed window that catches nothing usable must not fire a blank command.
e = make_ears(["hey tek"], [""])
e._utterance(DUMMY)
e._utterance(DUMMY)
check("an empty follow-up does not dispatch an empty command",
      e.service.events == [], e.service.events)
# ...and the window must SURVIVE it. After a reply the first thing segmented is
# often the room settling, which decodes to nothing; spending the arm on that
# left the real follow-up arriving to a closed door.
check("a blank scrap does not consume the listening window",
      time.monotonic() < e.armed_until, e.armed_until - time.monotonic())
e.free.script = ["and what about winter"]
e._utterance(DUMMY)
check("the real follow-up still lands after a blank one",
      [ev.get("heard") for ev in e.service.events] == ["and what about winter"],
      e.service.events)
check("and THAT disarms it", e.armed_until == 0.0)

# Counters, so `tek ears` reports something truthful.
e = make_ears(["[unk]", "hey tek [unk]"], ["hey tek hello there"])
e._utterance(DUMMY)
e._utterance(DUMMY)
check("counts every utterance, not just the ones that woke it",
      e.utterances == 2, e.utterances)
check("counts wakes and commands separately",
      (e.wakes, e.commands) == (1, 1), (e.wakes, e.commands))

# -- the wake word ALONE must not pay for a free decode --------------------
# Recognition is slower than real time on this board (a 0.7s "hey tek" takes
# 1.47s to free-decode) and parec's pipe holds 2.05s, so paying for a decode
# the instant somebody says the wake word is exactly when it can least afford
# to be deaf. The grammar already distinguishes the two cases: it returns
# "hey tek" alone, and "hey tek [unk]" when more followed.
e = make_ears(["hey tek"], ["SHOULD NOT BE CALLED"])
e._utterance(DUMMY)
check("a bare wake word arms without a free decode at all",
      e.free.calls == 0 and time.monotonic() < e.armed_until, e.free.calls)
e = make_ears(["hey tek [unk]"], ["hey tek what time is it"])
e._utterance(DUMMY)
check("a wake word WITH speech after it does decode",
      e.free.calls == 1 and e.service.events, (e.free.calls, e.service.events))


# -- the reader must never be blocked by the recogniser --------------------
# The whole failure was that transcription ran inline, so nothing read the
# microphone while it worked and the 2.05s pipe overflowed.
slow = vio.ArraySource(np.tile(loud, 300), pcm.RATE)
d = ears.Draining(slow, seconds=1.0)
time.sleep(0.3)
before = len(d.q)
time.sleep(0.5)                       # a "transcription" during which we read nothing
check("audio keeps being read while nothing consumes it",
      len(d.q) >= before and d.deepest > 0, (before, len(d.q), d.deepest))
check("the backlog is bounded, not unbounded",
      len(d.q) <= d.max, (len(d.q), d.max))
check("and it says how much it threw away", d.dropped > 0, d.dropped)
got = d.read()
check("what comes out is real audio", got is not None and pcm.envelope(got) > 0.4,
      None if got is None else pcm.envelope(got))
d.close()

empty = ears.Draining(vio.ArraySource(pcm.silence(0), pcm.RATE))
time.sleep(0.2)
check("a source that ends ends the drain too", empty.read() is None)
empty.close()


# -- a follow-up must not need the wake word again -------------------------
class _Gate(object):
    def __init__(self, spoke_at):
        self.spoke_at = spoke_at


e = make_ears(["[unk]"], ["yes please tell me more"])
e._gate = _Gate(time.monotonic())          # it just finished speaking
e._utterance(DUMMY)
check("straight after a reply, a follow-up needs no wake word",
      [ev.get("heard") for ev in e.service.events] == ["yes please tell me more"],
      e.service.events)

e = make_ears(["[unk]"], ["someone talking about something else"])
e._gate = _Gate(time.monotonic() - (ears.FOLLOWUP_S + 5))
e._utterance(DUMMY)
check("but ordinary chatter long afterwards is still ignored",
      e.service.events == [], e.service.events)
check("the follow-up window is short", ears.FOLLOWUP_S <= 20.0, ears.FOLLOWUP_S)


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
e = make_ears(["hey tek [unk]"], ["hey tek hello there"])
e._utterance(DUMMY)
check("a stripped remainder is dispatched even if it looks wake-ish",
      [ev.get("heard") for ev in e.service.events] == ["hello there"],
      e.service.events)

e = make_ears(["hey tech [unk]"], ["hate tech"])
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
