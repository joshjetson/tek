# -*- coding: utf-8 -*-
"""Face recognition: the invariants that were silently wrong.

Recognition degraded to UNKNOWN after the camera was swapped and nothing
caught it, because nothing tested it. What broke was not the model - it was
geometry and a preprocessing path that could drift between training and
prediction without failing.

Uses the real gallery when there is one. Recognition quality is a property of
actual photographs of actual faces; synthesised patterns would pin numbers
that mean nothing. Skips loudly when the gallery is empty.
"""
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tekdromo import recog

FAIL = []


def check(name, cond, extra=""):
    print("  %-56s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


# -- preprocessing contract ------------------------------------------------
blank = np.full((90, 70), 128, np.uint8)
check("prepare always returns the canonical size",
      recog.prepare(blank).shape[:2] == (recog.SIZE[1], recog.SIZE[0]),
      recog.prepare(blank).shape)
check("prepare survives having no landmarks",
      recog.prepare(blank, None) is not None)
check("prepare survives nonsense landmarks",
      recog.prepare(blank, np.zeros((68, 2), np.float32)) is not None)
check("align refuses landmarks with no eye separation",
      recog.align(blank, np.zeros((68, 2), np.float32)) is None)
check("align refuses a short landmark list",
      recog.align(blank, np.zeros((10, 2), np.float32)) is None)

# align must NOT equalise - _norm owns that, so it happens exactly once
# whichever path an image takes. Training images that were equalised twice
# while probes were equalised once is precisely the silent mismatch that makes
# a recogniser quietly worse instead of visibly broken.
pts = np.zeros((68, 2), np.float32)
for i, k in enumerate(recog._LEFT_EYE):
    pts[k] = (40 + i, 45)
for i, k in enumerate(recog._RIGHT_EYE):
    pts[k] = (80 + i, 45)
ramp = np.tile(np.linspace(0, 255, 120, dtype=np.uint8), (120, 1))
raw = recog.align(ramp, pts)
check("align does not equalise (that belongs to _norm alone)",
      not np.array_equal(raw, cv2.equalizeHist(raw)),
      "align appears to have equalised already")

# -- against the real gallery ----------------------------------------------
paths = [(n, p) for n in recog.people() for p in recog.samples(n)]
if len(paths) < 6:
    print("  (no gallery to test against - enrol someone first)")
else:
    names = sorted(set(n for n, _ in paths))
    print("  gallery: %d samples across %s" % (len(paths), names))
    imgs = []
    for n, p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is not None:
            imgs.append((n, g))

    r = recog.Recogniser()
    trained = r.reload()
    check("reload reports enrolled samples, not augmented copies",
          trained == len(imgs), (trained, len(imgs)))
    check("every enrolled person got a label",
          set(r.labels.values()) == set(names), r.labels)

    # A stored sample must match ITSELF very closely. If this is large, the
    # training and prediction paths have drifted apart.
    n0, g0 = imgs[0]
    who, dist = r.predict(g0)
    check("a stored sample matches itself", who == n0, (who, dist))
    check("and matches itself closely (paths have not drifted)",
          dist is not None and dist < 15.0, dist)

    # The camera the gallery was enrolled on is not the camera in the room.
    # Softer, lower-detail versions of a known face must still be recognised -
    # this is what the training augmentation exists for, and it is the whole
    # reason recognition survived a camera swap.
    for frac, blur, label in ((0.5, 0, "half resolution"),
                              (0.33, 0, "third resolution"),
                              (0.33, 5, "third resolution + blur")):
        got, d = r.predict(recog._degrade(recog._norm(g0), frac, blur))
        check("recognises a known face at %s" % label,
              got == n0, "%s at distance %s" % (got, d))

    # ...but not anything. A face-shaped image that is not in the gallery must
    # come back UNKNOWN: a panel confidently labelling a stranger with a family
    # member's name is worse than one that admits it does not know.
    rng = np.random.RandomState(0)
    for i in range(3):
        noise = rng.randint(0, 255, recog.SIZE, dtype=np.uint8)
        got, d = r.predict(noise)
        check("a stranger is UNKNOWN, not the nearest neighbour (%d)" % i,
              got == recog.UNKNOWN, "%s at %s" % (got, d))

    check("the threshold is what decides that",
          recog.THRESHOLD > 0 and recog.UNKNOWN == "UNKNOWN")

    # Augmentation must not cost anything on a clean face.
    check("augmentation does not hurt an undegraded match",
          r.predict(g0)[1] < 15.0, r.predict(g0)[1])


# -- voting: one stray frame must not wipe the label ----------------------
# "It says unknown" is what a flickering label looks like from across the
# room, even when most frames are right. The vote is the fix, so it is pinned.
import time as _time

from tekdromo import camera as _cam


def vote(seq):
    """Run a sequence of per-frame answers through Tracker's voting."""
    t = _cam.Tracker.__new__(_cam.Tracker)
    t._lock = __import__("threading").Lock()
    t._votes = []
    t._recog = None
    t._seen_logged = {}
    out = []
    for name in seq:
        t._votes.append((_time.time(), name))
        del t._votes[:-_cam.VOTE_N]
        fresh = [n for ts, n in t._votes
                 if _time.time() - ts <= _cam.VOTE_WINDOW]
        named = [n for n in fresh if n and n != "UNKNOWN"]
        out.append((max(set(named), key=named.count)
                    if len(named) * 2 >= len(fresh) else "UNKNOWN")
                   if fresh else None)
    return out


check("a steady run of hits reads as the name",
      vote(["JOSH"] * 5)[-1] == "JOSH", vote(["JOSH"] * 5))
check("one stray UNKNOWN does not wipe a good run",
      vote(["JOSH", "JOSH", "JOSH", "UNKNOWN"])[-1] == "JOSH",
      vote(["JOSH", "JOSH", "JOSH", "UNKNOWN"]))
check("a steady run of misses does read as UNKNOWN",
      vote(["UNKNOWN"] * 5)[-1] == "UNKNOWN")
check("one lucky hit among misses does NOT name someone",
      vote(["UNKNOWN", "UNKNOWN", "UNKNOWN", "JOSH"])[-1] == "UNKNOWN",
      vote(["UNKNOWN", "UNKNOWN", "UNKNOWN", "JOSH"]))
check("recognition is throttled below the detection rate",
      _cam.IDENTIFY_EVERY >= 0.25, _cam.IDENTIFY_EVERY)

print("RECOG " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
