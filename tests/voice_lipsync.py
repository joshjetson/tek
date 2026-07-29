# -*- coding: utf-8 -*-
"""End-to-end: does saying something actually move the face?

Everything else in the voice tests runs on stubs. This one is the integration
check, and it reads the real framebuffer while the real voice service speaks
through the real speaker - because "mouth frames were published" is not the
same claim as "the mouth moved", and only the second one matters.

Needs tek-display and tek-voice running.

IT TALKS OUT LOUD, and is therefore SKIPPED by default. Set TEK_AUDIBLE=1 to
run it.

The project already draws this line - README section 6: "Three are disruptive
and therefore live in tools/, not tests/". A test that says a sentence through
the speaker in somebody's house is disruptive by the same standard, and it did
not get the same treatment. Running the suite is a routine thing to do,
including from a cron job or while a family is in the room, and it should not
be a thing that makes the house talk. Reported from the sofa as "a random test
that happens randomly".

It stays in tests/ rather than moving to tools/ because it IS an assertion -
it fails the build when lip-sync breaks, which nothing in tools/ does.
"""
import mmap
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo.voice import bus

if os.environ.get("TEK_AUDIBLE", "").strip() not in ("1", "true", "yes"):
    print("  SKIPPED - this test speaks out loud. TEK_AUDIBLE=1 to run it.")
    print("VOICE LIPSYNC SKIPPED")
    raise SystemExit(0)

W, H = 1024, 600
FAIL = []


def check(name, cond, extra=""):
    print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


def grab():
    """Current framebuffer as a grayscale array."""
    fd = os.open("/dev/fb0", os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, W * H * 4, mmap.MAP_SHARED, mmap.PROT_READ)
        a = np.frombuffer(mm, np.uint8, W * H * 4).reshape(H, W, 4)
        out = a[:, :, :3].mean(axis=2).copy()
        mm.close()
        return out
    finally:
        os.close(fd)


# The mouth sits in the lower middle of the head. Deliberately narrow: using
# the whole frame would let the idle head-sway and the twinkling starfield
# swamp the signal we are actually looking for.
MOUTH = (slice(int(H * 0.56), int(H * 0.74)), slice(int(W * 0.40), int(W * 0.60)))

# -- is anything there to test? -------------------------------------------
try:
    c = bus.Client(bus.DEFAULT_PATH, timeout=10)
    st = c.request({"cmd": "status"})
    c.close()
except Exception as e:
    print("  voice service unreachable (%s) - start tek-voice" % e)
    sys.exit(2)
if not os.path.exists("/dev/fb0"):
    print("  no framebuffer - start tek-display")
    sys.exit(2)
print("  voice=%s rate=%s" % (st.get("voice"), st.get("rate")))

# -- collect mouth frames while speaking ----------------------------------
frames = []
stamps = []
speaking = []
reported = {}


def subscriber():
    s = bus.Client(bus.DEFAULT_PATH, timeout=40)
    s.subscribe()
    for msg in s:
        if "mouth" in msg:
            frames.append(msg["mouth"])
            stamps.append(time.time())
        if "speaking" in msg:
            speaking.append(msg["speaking"])
            if msg["speaking"] and "duration" in msg:
                reported["duration"] = msg["duration"]
            if msg["speaking"] is False and frames:
                break
    s.close()


t = threading.Thread(target=subscriber)
t.daemon = True
t.start()
time.sleep(1.0)

quiet_before = grab()[MOUTH]

shots = []


def watch():
    t0 = time.time()
    while time.time() - t0 < 8.0:
        shots.append(grab()[MOUTH])
        time.sleep(0.08)


w = threading.Thread(target=watch)
w.daemon = True
w.start()

TEXT = ("Testing the mouth. Watch how it opens and closes while I am talking, "
        "and then stops when I stop.")
r = subprocess.run([sys.executable, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tek"),
    "say", TEXT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
w.join(timeout=10)
t.join(timeout=6)
print("  " + r.stdout.decode().strip())

# -- the mouth stream ------------------------------------------------------
check("the voice service published mouth frames", len(frames) > 20, len(frames))
if frames:
    opens = [f[0] for f in frames]
    check("openness varies (not a stuck value)",
          max(opens) - min(opens) > 0.05,
          "range %.3f-%.3f" % (min(opens), max(opens)))
    check("the mouth actually opens", max(opens) > 0.08, "max %.3f" % max(opens))
    check("openness returns to zero at the end", opens[-1] == 0.0, opens[-1])
    rounds = [f[1] for f in frames]
    check("rounding comes from the phonemes, not hardcoded 0",
          max(rounds) > 0.0, "max %.3f" % max(rounds))
check("speaking went true then false", speaking[:1] == [True] and
      speaking[-1:] == [False], speaking)

# THE regression that mattered: the mouth must last as long as the sound.
# pacat's stdin is a pipe with no backpressure - it accepted 3.0s of audio in
# 0.01s - so driving the mouth from the write loop animated a whole sentence
# in ten milliseconds and then left the face still. Visibly: "it stopped
# before you stopped".
if len(stamps) > 2 and reported.get("duration"):
    spread = stamps[-1] - stamps[0]
    want = reported["duration"]
    print("    mouth stream lasted %.2fs for %.2fs of audio" % (spread, want))
    check("the mouth lasts as long as the audio (not a 10ms burst)",
          abs(spread - want) < 0.6 * want, "%.2fs vs %.2fs" % (spread, want))
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    med = sorted(gaps)[len(gaps) // 2]
    check("frames are paced at ~20ms, not dumped at once",
          0.010 < med < 0.060, "median gap %.1f ms" % (med * 1000))

# -- the actual pixels -----------------------------------------------------
check("captured framebuffer during speech", len(shots) > 20, len(shots))
if shots:
    stack = np.stack(shots)
    per_frame = stack.reshape(len(shots), -1).mean(axis=1)
    motion = float(per_frame.max() - per_frame.min())
    # Compare against the quiet baseline: how far the mouth region departed
    # from its resting appearance at the most extreme moment.
    departure = float(np.abs(stack - quiet_before).reshape(len(shots), -1)
                      .mean(axis=1).max())
    print("    mouth-region brightness swing during speech : %.2f" % motion)
    print("    max departure from the quiet baseline       : %.2f" % departure)
    check("the mouth region visibly changes while speaking", motion > 0.5, motion)
    check("it departs from the resting face", departure > 0.5, departure)

print("VOICE LIPSYNC " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
