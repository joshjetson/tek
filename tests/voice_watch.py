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
check("prompt carries face count", "Faces visible: 2" in p)
check("prompt carries when it last spoke", "10 minutes ago" in p, p[:0])
check("prompt carries what it recently said", "Hello there." in p)
check("prompt makes silence the explicit default", "STAY SILENT" in p)
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

svc.server.close()
print("VOICE WATCH " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
