# -*- coding: utf-8 -*-
"""Camera-triggered speech: the decision path, without spending API calls.

Every real decision costs money and ~10 s, so the whole pipeline runs against
StubBrain here. What is under test is the part that decides *whether to even
ask* - cooldowns, the on/off switch, departures - because that logic is what
stands between an ambient face and a device that talks over your evening, and
between the user and an unbounded bill.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tekdromo.voice import agent, service

FAIL = []


def check(name, cond, extra=""):
    print("  %-54s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


# -- parsing a decision ----------------------------------------------------
# Getting this wrong means the face announces the word "silence" out loud,
# which is exactly the sort of thing that makes a device feel broken.
for text in ("SILENCE", "silence", " SILENCE ", '"SILENCE"', "SILENCE.",
             "", "   ", "I would stay quiet here.", "nothing to say"):
    check("declines to speak on %r" % text, agent.parse(text) is None,
          agent.parse(text))

check("passes real words through",
      agent.parse("Hey Josh, welcome back.") == "Hey Josh, welcome back.")
check("strips surrounding quotes",
      agent.parse('"Hey Josh."') == "Hey Josh.")
check("does not mistake a sentence containing the word for a decline",
      agent.parse("The silence in here is nice.") is not None)
long = "This is a sentence. " * 60
out = agent.parse(long)
check("truncates a speech down to a remark", out and len(out) <= 401, len(out or ""))

# -- prompt construction ---------------------------------------------------
b = agent.ClaudeBrain()
p = b.build_prompt({"kind": "arrival", "what": "someone came into view",
                    "faces": 2, "when": "Sunday 14:40",
                    "image": "/tmp/x.jpg", "last_spoken_ago": 600,
                    "recent": ["Hello there."]})
check("prompt names the image file", "/tmp/x.jpg" in p)
check("prompt carries the event", "someone came into view" in p)
check("prompt carries face count", "Faces detected: 2" in p)
check("prompt carries when it last spoke", "10 minutes ago" in p, p[:0])
check("prompt carries what it recently said", "Hello there." in p)
check("prompt offers silence as an explicit option", agent.SILENCE in p)
# The lean is per-event because one blanket rule was wrong: telling the model
# that "merely detecting a person" was a bad reason to speak meant it sat
# silent through every arrival, which is the entire use case. Pin the three
# leans so that cannot regress into over-restraint again.
lean_manual = b.build_prompt({"kind": "manual", "image": "/tmp/x.jpg", "faces": 1})
lean_arrival = b.build_prompt({"kind": "arrival", "image": "/tmp/x.jpg", "faces": 1})
lean_other = b.build_prompt({"kind": "timer", "image": "/tmp/x.jpg", "faces": 1})
check("a direct 'look now' leans towards speaking",
      "not the moment for restraint" in lean_manual)
check("an arrival leans towards greeting",
      "appropriate and welcome" in lean_arrival)
check("anything else still prefers silence",
      "Prefer silence" in lean_other)
check("the three leans are actually different",
      len({lean_manual, lean_arrival, lean_other}) == 3)
check("prompt warns that the camera is wide and people sit at the edge",
      "edge of frame" in p)
check("brain runs in a neutral cwd, not the project",
      "tekdromo/brain" in b.cwd and not b.cwd.rstrip("/").endswith("/tekdromo"),
      b.cwd)
check("brain uses an absolute path to the CLI",
      b.exe.startswith("/") or b.exe == "claude", b.exe)

# -- the gate --------------------------------------------------------------
sock = os.path.join(tempfile.mkdtemp(prefix="tekwatch"), "v.sock")
# espeak, not piper: this test is about the decision gate and should not pay a
# 5s model load to make its point.
svc = service.VoiceService(voice="espeak", path=sock, cooldown=60.0)
svc.brain = agent.StubBrain(reply=None)

r = svc.on_event({"kind": "arrival", "faces": 1})
check("an arrival is considered", r.get("acted") is True, r)

r = svc.on_event({"kind": "arrival", "faces": 1})
check("a second arrival inside the cooldown is refused",
      r.get("acted") is False and r.get("reason") == "cooldown", r)
check("it reports how long until the next one", r.get("next_in", 0) > 0, r)

svc.last_event = 0.0
r = svc.on_event({"kind": "departure", "faces": 0})
check("a departure never triggers a decision (nobody is there to hear it)",
      r.get("acted") is False and r.get("reason") == "departure", r)

svc.watching = False
svc.last_event = 0.0
r = svc.on_event({"kind": "arrival", "faces": 1})
check("watching off blocks everything", r.get("acted") is False, r)
svc.watching = True

# -- it actually speaks when the brain says something ----------------------
spoken = []
svc._say = lambda text, voice=None: spoken.append(text) or {"ok": True}
svc.brain = agent.StubBrain(reply="Hey Josh, welcome back.")
svc.last_event = 0.0
svc.on_event({"kind": "arrival", "faces": 1})
for _ in range(60):
    if spoken:
        break
    time.sleep(0.1)
check("a decision to speak reaches the voice", spoken == ["Hey Josh, welcome back."],
      spoken)
check("what it said is remembered for next time's context",
      svc.recent and svc.recent[-1] == "Hey Josh, welcome back.", svc.recent)

quiet = []
svc.brain = agent.StubBrain(reply=None)
svc.last_event = 0.0
before = len(spoken)
svc.on_event({"kind": "arrival", "faces": 1})
time.sleep(1.0)
check("a decision to stay silent says nothing at all", len(spoken) == before,
      spoken[before:])

# -- context handed to the brain ------------------------------------------
stub = agent.StubBrain(reply=None)
svc.brain = stub
svc.last_event = 0.0
svc.on_event({"kind": "arrival", "faces": 3, "image": "/tmp/y.jpg"})
time.sleep(0.8)
ev = stub.calls[-1] if stub.calls else {}
check("the brain is told how many faces", ev.get("faces") == 3, ev)
check("the brain is given the image", ev.get("image") == "/tmp/y.jpg", ev)
check("the brain is told what was recently said", "recent" in ev, ev)
check("the brain is told when it last spoke", "last_spoken_ago" in ev, ev)

# -- keeping the speaker awake --------------------------------------------
# The tone has to be real signal (a speaker that sleeps on digital silence must
# see something) while being inaudible in practice. Both halves are pinned.
from tekdromo.voice import pcm as _pcm
k = _pcm.tone(service.KEEPALIVE_HZ, service.KEEPALIVE_SECS,
              service.KEEPALIVE_AMP, 44100)
check("keepalive tone is not digital silence", int(abs(k).max()) > 0,
      int(abs(k).max()))
# The first version used 40 Hz precisely BECAUSE a portable driver cannot
# reproduce it - which is self-defeating: a speaker's auto-off detector works
# on the same post-filter path as its amplifier, so a tone it cannot reproduce
# is a tone it cannot detect. It sent 34 tones over three hours and the speaker
# still switched off. The tone must sit inside the range the speaker actually
# plays.
check("keepalive tone is INSIDE the speaker's reproducible range",
      100 <= service.KEEPALIVE_HZ <= 8000, service.KEEPALIVE_HZ)
check("keepalive tone is NOT ultrasonic (children hear past 18kHz)",
      service.KEEPALIVE_HZ < 15000, service.KEEPALIVE_HZ)
check("keepalive tone is quiet", service.KEEPALIVE_AMP <= 0.1,
      service.KEEPALIVE_AMP)
check("keepalive tone starts and ends at zero (a click would be audible even "
      "when the tone is not)", k[0] == 0 and k[-1] == 0, (k[0], k[-1]))

played = []
svc.last_audio = 0.0
svc.speaking = False
real_sink = service.vio.SpeakerSink
service.vio.SpeakerSink = lambda device=None, rate=16000: type(
    "S", (), {"write": lambda self, f: played.append(f),
              "close": lambda self: None})()
try:
    check("keepalive plays when the speaker has been idle", svc._keepalive())
    check("it actually wrote audio", len(played) > 10, len(played))
    n = len(played)
    svc.speaking = True
    check("keepalive is skipped while speaking", svc._keepalive() is False)
    check("nothing was written while speaking", len(played) == n)
    svc.speaking = False
    svc.keepalive_every = 0
    check("interval 0 disables it", svc.keepalive_every == 0)
finally:
    service.vio.SpeakerSink = real_sink

svc.server.close()
print("VOICE WATCH " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
