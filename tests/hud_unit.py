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


def at(y, mo, d, hh, mm, ss=0):
    return time.mktime((y, mo, d, hh, mm, ss, 0, 0, -1))


# -- correctness -----------------------------------------------------------
c = hud.Clock(W, H)


def shown(s):
    """What a person reads. The pad is "_", a full-width blank glyph that
    draws nothing, so it has to come out before comparing with a real clock."""
    return s.replace("_", "").strip()


t, mer, date = c.strings()
check("time agrees with the system clock",
      "%s %s" % (shown(t), mer) == time.strftime("%-I:%M %p"),
      "%s %s vs %s" % (shown(t), mer, time.strftime("%-I:%M %p")))
check("date is mm/dd/yyyy and agrees",
      date == time.strftime("%m/%d/%Y"), date)

# 12-hour conventions are where clocks quietly get it wrong.
check("midnight is 12 AM, not 0 AM",
      (shown(c.strings(at(2026, 7, 26, 0, 5))[0]),
       c.strings(at(2026, 7, 26, 0, 5))[1]) == ("12:05", "AM"))
check("noon is 12 PM, not 0 PM",
      (shown(c.strings(at(2026, 7, 26, 12, 5))[0]),
       c.strings(at(2026, 7, 26, 12, 5))[1]) == ("12:05", "PM"))
check("11pm is 11 PM",
      (shown(c.strings(at(2026, 7, 26, 23, 5))[0]),
       c.strings(at(2026, 7, 26, 23, 5))[1]) == ("11:05", "PM"))
check("1am is 1 AM", c.strings(at(2026, 7, 26, 1, 5))[1] == "AM")
check("single-digit hours are padded to a fixed width",
      c.strings(at(2026, 7, 26, 9, 5))[0] == "_9:05",
      repr(c.strings(at(2026, 7, 26, 9, 5))[0]))
check("the pad is a FULL-width blank, the blink blank is NARROW",
      hud.text("_", 0, 0, 20, 30)[1] == hud.text("8", 0, 0, 20, 30)[1]
      and hud.text(" ", 0, 0, 20, 30)[1] == hud.text(":", 0, 0, 20, 30)[1],
      (hud.text("_", 0, 0, 20, 30)[1], hud.text(" ", 0, 0, 20, 30)[1]))

# -- the panel must not move or resize -------------------------------------
rects = set()
for hh in range(24):
    for mm in (0, 8, 11, 59):
        # BOTH blink states. The original version of this test only ever used
        # seconds=0, so the colon was always in the same phase and it missed
        # the panel growing and shrinking once a second: the blink swaps ":"
        # for a blank, and the two were different widths. It was obvious on
        # screen and invisible here.
        for ss in (0, 1):
            cc = hud.Clock(W, H)
            cc.points(at(2026, 7, 26, hh, mm, ss))
            rects.add(tuple(int(v) for v in cc.rect))
check("the panel is identical at every time of day (never resizes or jumps)",
      len(rects) == 1, sorted(rects))
check("the blinking colon does not change the panel width",
      len({tuple(int(v) for v in (lambda c: (c.points(at(2026, 7, 26, 3, 4, s)),
                                             c.rect)[1])(hud.Clock(W, H)))
           for s in (0, 1)}) == 1)

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

# -- the oscilloscope ------------------------------------------------------
sc = hud.Scope(W, H)
flat = sc.points()
check("scope emits the renderer's segment format",
      flat.ndim == 3 and flat.shape[1:] == (2, 2) and flat.dtype == np.int32,
      (flat.shape, flat.dtype))
check("scope sits in the lower right", sc.bx > W * 0.6 and sc.by > H * 0.6,
      (sc.bx, sc.by))
check("scope is fully on screen",
      sc.bx >= 0 and sc.by >= 0 and sc.bx + sc.bw <= W and sc.by + sc.bh <= H)
# It must not collide with the clock, which is in the other right-hand corner.
ck = hud.Clock(W, H); ck.points()
cx, cy, cw2, chh = ck.rect
check("scope does not overlap the clock panel",
      sc.by > cy + chh, (sc.by, cy + chh))


def deviation(pts, sc):
    """How far the TRACE departs from the zero axis.

    Only the trace - the last cols-1 segments. Measuring all the points
    includes the bezel, which sits at a constant +/-52 px and swamps the
    signal, so every case reported 52.0 and the test could not tell silence
    from a sine wave.
    """
    trace = pts[-(sc.cols - 1):]
    mid = sc.by + sc.bh * 0.5
    return float(np.abs(trace[..., 1].astype(np.float32) - mid).max())


sil = hud.Scope(W, H)
sil.push(np.zeros(2048, np.int16))
quiet_dev = deviation(sil.points(), sil)

loud = hud.Scope(W, H)
t = np.arange(4096, dtype=np.float32) / 16000.0
loud.push((np.sin(2 * np.pi * 200 * t) * 12000).astype(np.int16))
loud_dev = deviation(loud.points(), loud)
check("silence draws a flat trace", quiet_dev < sc.bh * 0.35, quiet_dev)
check("a signal deflects the trace", loud_dev > quiet_dev + 8,
      (loud_dev, quiet_dev))

# Auto-gain: quiet music must still be visible, not a flat line.
faint = hud.Scope(W, H)
# Push repeatedly. The peak decays 0.9 per push and audio arrives ~50 times a
# second, so a single push leaves the tracker still sitting on its initial
# value - which is not how it is ever fed in practice.
for _ in range(60):
    faint.push((np.sin(2 * np.pi * 200 * t[:320]) * 300).astype(np.int16))
faint_dev = deviation(faint.points(), faint)
check("a quiet signal becomes visible once the gain settles",
      faint_dev > quiet_dev + 15, (faint_dev, quiet_dev))
check("and it does not blow up past the panel",
      faint_dev <= sc.bh * 0.5, faint_dev)

# Robustness: the feeder hands it whatever parec produced.
for bad in (np.zeros(0, np.int16), np.zeros(3, np.int16),
            np.zeros(99999, np.int16)):
    try:
        sc.push(bad); sc.points()
        check("survives a %d-sample push" % len(bad), True)
    except Exception as e:
        check("survives a %d-sample push" % len(bad), False, e)

# The trigger is what stops the trace sliding sideways every frame.
trig = hud.Scope(W, H)
trig.push((np.sin(2 * np.pi * 200 * t) * 12000).astype(np.int16))
a1 = trig.points()
a2 = trig.points()
check("the same buffer draws the same trace (stable trigger)",
      np.array_equal(a1, a2))

# Budget. A decoration must not eat the frame: the first version cost 30ms.
import time as _t
trig.points()
t0 = _t.time()
for _ in range(300):
    trig.points()
ms = (_t.time() - t0) / 300 * 1000
print("    scope costs %.3f ms/frame" % ms)
check("scope stays well inside the frame budget", ms < 3.0, "%.2f ms" % ms)

print("HUD " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
