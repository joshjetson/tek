# -*- coding: utf-8 -*-
"""The clock/date panel.

A clock is a thing people glance at, so the failures that matter are not
crashes - they are being wrong, jittering, or creeping over the face. All three
are checked here across every hour of the day rather than at whatever time the
test happens to run.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo import hud

W, H = 1024, 600
FAIL = []


def check(name, cond, extra=""):
    print("  %-54s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


def at(y, mo, d, hh, mm):
    return time.mktime((y, mo, d, hh, mm, 0, 0, 0, -1))


# -- correctness -----------------------------------------------------------
c = hud.Clock(W, H)
t, mer, date = c.strings()
check("time agrees with the system clock",
      "%s %s" % (t.strip(), mer) == time.strftime("%-I:%M %p"),
      "%s %s vs %s" % (t.strip(), mer, time.strftime("%-I:%M %p")))
check("date is mm/dd/yyyy and agrees",
      date == time.strftime("%m/%d/%Y"), date)

# 12-hour conventions are where clocks quietly get it wrong.
check("midnight is 12 AM, not 0 AM", c.strings(at(2026, 7, 26, 0, 5))[:2]
      == ("12:05", "AM"), c.strings(at(2026, 7, 26, 0, 5))[:2])
check("noon is 12 PM, not 0 PM", c.strings(at(2026, 7, 26, 12, 5))[:2]
      == ("12:05", "PM"), c.strings(at(2026, 7, 26, 12, 5))[:2])
check("11pm is 11 PM", c.strings(at(2026, 7, 26, 23, 5))[:2] == ("11:05", "PM"))
check("1am is 1 AM", c.strings(at(2026, 7, 26, 1, 5))[1] == "AM")
check("single-digit hours are padded to a fixed width",
      c.strings(at(2026, 7, 26, 9, 5))[0] == " 9:05",
      repr(c.strings(at(2026, 7, 26, 9, 5))[0]))

# -- the panel must not move or resize -------------------------------------
rects = set()
for hh in range(24):
    for mm in (0, 8, 11, 59):
        cc = hud.Clock(W, H)
        cc.points(at(2026, 7, 26, hh, mm))
        rects.add(tuple(int(v) for v in cc.rect))
check("the panel is identical at every time of day (never resizes or jumps)",
      len(rects) == 1, sorted(rects))

# A date with wide digits must not change it either.
for mo, d, y in ((11, 28, 2026), (1, 1, 2027), (12, 30, 2025)):
    cc = hud.Clock(W, H)
    cc.points(at(y, mo, d, 10, 8))
    rects.add(tuple(int(v) for v in cc.rect))
check("nor does a different date", len(rects) == 1, sorted(rects))

x, y, bw, bh = list(rects)[0]

# -- placement -------------------------------------------------------------
check("panel is in the upper right", x > W * 0.6 and y < H * 0.2, (x, y))
check("panel is fully on screen", x >= 0 and y >= 0 and x + bw <= W and y + bh <= H,
      (x, y, bw, bh))
# The head is drawn centred; the panel must not sit over it.
check("panel clears the centre where the head is drawn", x > W * 0.62, x)

# -- the vectors themselves ------------------------------------------------
pts = hud.Clock(W, H).points(at(2026, 7, 26, 10, 8))
check("emits the same (N,2,2) int32 the head does",
      pts.ndim == 3 and pts.shape[1:] == (2, 2) and pts.dtype == np.int32,
      (pts.shape, pts.dtype))
check("emits a sensible number of segments", 40 < len(pts) < 400, len(pts))
check("every vertex is on screen",
      pts[..., 0].min() >= 0 and pts[..., 0].max() < W
      and pts[..., 1].min() >= 0 and pts[..., 1].max() < H,
      (pts[..., 0].max(), pts[..., 1].max()))
check("no zero-length segments (they draw nothing and cost a stroke)",
      bool(np.any(pts[:, 0] != pts[:, 1], axis=1).all()))

# -- glyph coverage --------------------------------------------------------
# A missing glyph is silent: the character simply does not draw, and a clock
# with an invisible digit still looks like a clock.
for ch in "0123456789:/APM":
    check("glyph exists for %r" % ch, len(hud._seg_lines(ch)) > 0)
check("unknown characters draw nothing rather than raising",
      hud._seg_lines("Z") == [])

# -- caching ---------------------------------------------------------------
# The render loop runs 30 times a second and must not rebuild 30 times.
cc = hud.Clock(W, H)
base = at(2026, 7, 26, 10, 8)
first = cc.points(base)
again = cc.points(base + 0.3)
check("the same second reuses the cached geometry", first is again)
later = cc.points(base + 61)
check("a changed minute rebuilds", later is not first)

print("HUD " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
