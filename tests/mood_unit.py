#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The expression a reply asks for: parsing it, guessing it, and never speaking it.

The failure this guards against is not subtle - a tag that survives parsing is
the face saying the word "amused" out loud before its sentence - but it is easy
to reintroduce, because the tag has to be stripped on three separate paths:
the blocking reply, the streamed reply, and the streamed reply that turned out
to be short enough to arrive whole.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

from tekdromo import rig                                # noqa: E402
from tekdromo.voice import agent                        # noqa: E402

FAIL = []


def check(label, got, want):
    if got != want:
        FAIL.append("%s: got %r want %r" % (label, got, want))


def ok(label, cond, detail=""):
    if not cond:
        FAIL.append("%s%s" % (label, (" - " + detail) if detail else ""))


# -- every mood the brain may ask for must actually exist in the rig --------
# Otherwise express() raises KeyError inside the frame loop, which is the one
# place in this project nothing is allowed to throw.
for m in agent.MOODS:
    ok("rig knows %r" % m, m in rig.EXPRESSIONS)

# ...and the states the display owns must NOT be settable by the brain.
for reserved in ("asleep", "speaking", "listening"):
    ok("%r is not offerable" % reserved, reserved not in agent.MOODS)

# -- parsing ----------------------------------------------------------------
check("square brackets", agent.split_mood("[amused] Of course it was."),
      ("amused", "Of course it was."))
check("round brackets, mixed case",
      agent.split_mood("(Concerned)  The garage is open."),
      ("concerned", "The garage is open."))
check("no tag", agent.split_mood("Just words."), (None, "Just words."))
check("unknown tag is left alone", agent.split_mood("[wry] Hah."),
      (None, "[wry] Hah."))
# Anchored: a bracket mid-sentence is text, not a tag.
check("mid-sentence bracket is not a tag",
      agent.split_mood("It was (happy) enough."),
      (None, "It was (happy) enough."))
check("empty", agent.split_mood(""), (None, ""))

# The tag must never reach the speaker.
for raw in ("[happy] Hello there.", "(surprised) You are early."):
    _, rest = agent.split_mood(raw)
    ok("tag never spoken: %r" % raw, "[" not in rest and "(" not in rest)

# -- the keyword fallback ---------------------------------------------------
check("apology", agent.guess_mood("Sorry, that failed."), "concerned")
check("welcome", agent.guess_mood("Great to see you back."), "happy")
check("uncertainty", agent.guess_mood("I am not sure what you meant."),
      "confused")
check("plain statement", agent.guess_mood("It is half past four."), "neutral")
check("nothing", agent.guess_mood(""), "neutral")
ok("fallback only returns offerable moods",
   all(agent.guess_mood(s) in agent.MOODS
       for s in ("sorry", "great", "not sure", "wow", "", "plain")))

# -- silence must not set a face --------------------------------------------
# A mood taken from a reply that turned out to be SILENCE would leave the face
# wearing an expression for a sentence nobody ever heard.
class _Fake(agent.ClaudeBrain):
    def __init__(self, out):
        agent.ClaudeBrain.__init__(self)
        self._out = out

    def respond(self, event):
        mood, text = agent.split_mood(self._out)
        words = agent.parse(text, agent.limit_for(event.get("kind")))
        self.last_mood = (mood or agent.guess_mood(words)) if words else None
        return words


b = _Fake("[amused] SILENCE")
check("a declined reply says nothing", b.respond({"kind": "arrival"}), None)
check("...and sets no mood", b.last_mood, None)

b = _Fake("[amused] Of course it was the cat.")
check("a real reply is spoken without its tag",
      b.respond({"kind": "arrival"}), "Of course it was the cat.")
check("...and sets the mood", b.last_mood, "amused")

b = _Fake("Sorry, the boiler is off.")
check("an untagged reply still gets a face",
      b.respond({"kind": "speech"}), "Sorry, the boiler is off.")
check("...from the keywords", b.last_mood, "concerned")

# -- the rig can actually wear them, and they move the face -----------------
f = rig.Face()
f.express("neutral", blend=0.01)
f.update(0.0, 0.033)
base = dict(f.controls)
moved = []
for m in agent.MOODS:
    f.express(m, blend=0.01)
    for i in range(6):                      # let the blend finish
        f.update(i * 0.033, 0.033)
    if any(abs(f.controls[k] - base[k]) > 1e-3 for k in base):
        moved.append(m)
ok("every mood but neutral changes the face",
   set(moved) == set(m for m in agent.MOODS if m != "neutral"),
   "moved: %s" % sorted(moved))

if FAIL:
    print("MOOD FAIL")
    for f_ in FAIL:
        print("  -", f_)
    sys.exit(1)
print("MOOD OK")
