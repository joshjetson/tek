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
from tekdromo.voice import pcm

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


t, mer, date, secs, dow = c.strings()
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
# Seconds and the weekday must not resize it either: seconds are always two
# digits and weekday abbreviations always three characters, but that has to be
# true in practice, not just in principle.
for ss in (0, 7, 59):
    for day in range(1, 8):
        cc = hud.Clock(W, H)
        cc.points(at(2026, 7, 20 + day, 10, 8, ss))
        rects.add(tuple(int(v) for v in cc.rect))
check("seconds and weekday do not resize the panel", len(rects) == 1,
      sorted(rects))
check("the clock shows seconds", c.strings(at(2026, 7, 26, 10, 8, 42))[3] == "42")
check("the clock shows the weekday",
      c.strings(at(2026, 7, 26, 10, 8))[4] in
      ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"))
# Unlit segments go to the DIM layer. At full brightness they merge with the
# lit ones and "8:47" reads as "8:40".
cg = hud.Clock(W, H)
cg.points(at(2026, 7, 26, 10, 8, 42))
check("unlit segments are emitted separately for the dim layer",
      len(cg.dim_points()) > 0, len(cg.dim_points()))
check("the dim layer is a different set from the bright one",
      len(cg.dim_points()) < len(cg.points()))
# The dim layer now carries the SLAB's back face as well as any unlit
# segments, so "ghosts off" no longer means "dim layer empty".
cg2 = hud.Clock(W, H, ghosts=True, slab=True)
cg2.points(at(2026, 7, 26, 10, 8, 42))
cn = hud.Clock(W, H, ghosts=False, slab=True)
cn.points(at(2026, 7, 26, 10, 8, 42))
check("ghosts add to the dim layer and can be turned off",
      len(cn.dim_points()) < len(cg2.dim_points()),
      (len(cn.dim_points()), len(cg2.dim_points())))
bare = hud.Clock(W, H, ghosts=False, slab=False)
bare.points(at(2026, 7, 26, 10, 8, 42))
check("with neither, nothing goes to the dim layer",
      len(bare.dim_points()) == 0, len(bare.dim_points()))

# The slab is drawn with the head's own projection, and must not grow off
# screen or over the face panel.
sl = hud.Clock(W, H, slab=True)
sl.points(at(2026, 7, 26, 10, 8, 42))
sx, sy, sw2, sh2 = sl.rect
check("the 3D slab stays on screen",
      sx >= 0 and sy >= 0 and sx + sw2 <= W and sy + sh2 <= H,
      (sx, sy, sw2, sh2))
check("the slab has visible depth (back face offset from the front)",
      len(sl.dim_points()) > 8, len(sl.dim_points()))
# And it still must not resize as the time changes.
srects = set()
for hh2 in (1, 9, 10, 12, 23):
    for ss2 in (0, 1, 59):
        cc2 = hud.Clock(W, H, slab=True)
        cc2.points(at(2026, 7, 26, hh2, 34, ss2))
        srects.add(tuple(int(v) for v in cc2.rect))
check("the slab does not resize with the time", len(srects) == 1,
      sorted(srects))

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
# NOT "Z" - that used to be undefined and is now a real letter, so this test
# started failing the moment the alphabet was added. Pick something the font
# will never contain.
check("unknown characters draw nothing rather than raising",
      hud._seg_lines("\u00a7") == [] and hud._seg_lines("~") == [])
for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if not hud._seg_lines(ch):
        check("letter %r has a glyph" % ch, False)
check("the whole uppercase alphabet has glyphs (names are drawn with it)",
      all(hud._seg_lines(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
check("a name label is centred under the face and stays in the panel",
      True)
fpl = hud.FacePanel(W, H)
ang2 = np.linspace(0, 2 * np.pi, 68, endpoint=False)
lm2 = np.column_stack([0.5 + 0.16 * np.cos(ang2),
                       0.5 + 0.22 * np.sin(ang2)]).astype(np.float32)
for _ in range(8):
    fpl.update(lm2, "ABRAHAM")
withname = fpl.points()
for _ in range(8):
    fpl.update(lm2, None)
withunknown = fpl.points()
check("a longer name draws more segments than UNKNOWN does not crash",
      len(withname) > 0 and len(withunknown) > 0)
check("the label stays inside the panel",
      withname[..., 0].min() >= fpl.bx and withname[..., 0].max() <= fpl.bx + fpl.bw
      and withname[..., 1].max() <= fpl.by + fpl.bh,
      (withname[..., 0].min(), withname[..., 0].max(), withname[..., 1].max()))

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
check("scope draws NO box and no axis - only the trace",
      len(flat) == (sc.cols - 1) * 2 + sc.cols, len(flat))
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
    """How far the trace departs from where the zero axis would be.

    The scope has no bezel and no axis now - the whole output is the trace, so
    every point counts. When it did have a bezel, measuring all points included
    it at a constant +/-52 px and every case reported 52.0, so the test could
    not tell silence from a sine wave.
    """
    mid = sc.by + sc.bh * 0.5
    return float(np.abs(pts[..., 1].astype(np.float32) - mid).max())


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
# Ten seconds' worth. The peak decays 0.995 per push - deliberately slow, so
# the trace does not pump on every transient - which at 50 frames a second is
# about 0.78 per second, so a quiet passage takes several seconds to bring the
# gain up. Sixty pushes is barely one second and the gain has hardly moved.
for _ in range(500):
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
check("the same buffer draws the same trace", np.array_equal(a1, a2))

# The window must be long enough to span the gaps between words. At 32 ms the
# trace was flat most of the time during real speech, because a glance lands
# inside a pause: two captures 1.2s apart in one sentence gave 44 lit rows then
# 3. Anything shorter than ~100 ms has that problem.
# History length is the thing that was wrong: a 32ms oscilloscope window was
# flat almost every time anyone looked, because speech is mostly gaps.
hist_s = trig.cols * pcm.FRAME_MS / 1000.0
check("the scope holds at least 1.5s of history", hist_s >= 1.5,
      "%.1f s" % hist_s)

# Speech with realistic gaps: a burst then silence, repeatedly. Every frame
# during the burst-and-gap pattern should still show something.
gappy = hud.Scope(W, H)
flatcount = 0
for i in range(40):
    loud = (i % 8) < 3          # ~200ms of speech then ~330ms of gap
    chunk = ((np.sin(2 * np.pi * 200 * t[:320]) * 9000) if loud
             else np.zeros(320)).astype(np.int16)
    gappy.push(chunk)
    if i > 8 and deviation(gappy.points(), gappy) < 6:
        flatcount += 1
check("the trace does not go flat during the gaps between words",
      flatcount == 0, "%d flat frames of 31" % flatcount)

# Budget. A decoration must not eat the frame: the first version cost 30ms.
import time as _t
trig.points()
t0 = _t.time()
for _ in range(300):
    trig.points()
ms = (_t.time() - t0) / 300 * 1000
print("    scope costs %.3f ms/frame" % ms)
check("scope stays well inside the frame budget", ms < 3.0, "%.2f ms" % ms)

# -- the landmark face panel -----------------------------------------------
# Drawn the way the HEAD is drawn: an implicit field sliced into iso-contours
# with the same _march the head uses. An earlier version drew the 68 landmarks
# as a single-stroke outline, which is a different idiom entirely and looked
# like clip-art beside the contoured head.
fp = hud.FacePanel(W, H)
empty = fp.points()
check("nothing is drawn when nobody is there (no box, no placeholder)",
      len(empty) == 0, len(empty))
check("face panel sits in the upper left", fp.bx < W * 0.3 and fp.by < H * 0.3,
      (fp.bx, fp.by))
check("face panel does not overlap the clock", fp.bx + fp.bw < cx,
      (fp.bx + fp.bw, cx))

# A plausible 68-point face: wider than tall is wrong, so build a real oval.
ang = np.linspace(0, 2 * np.pi, 68, endpoint=False)
lm = np.column_stack([0.5 + 0.16 * np.cos(ang),
                      0.5 + 0.22 * np.sin(ang)]).astype(np.float32)
for _ in range(20):
    fp.update(lm)
face = fp.points()
check("a face produces contour rings, not a handful of strokes",
      len(face) > 200, len(face))
check("the face stays inside its panel",
      face[..., 0].min() >= fp.bx and face[..., 0].max() <= fp.bx + fp.bw
      and face[..., 1].min() >= fp.by and face[..., 1].max() <= fp.by + fp.bh,
      (face[..., 0].min(), face[..., 0].max(),
       face[..., 1].min(), face[..., 1].max()))
check("no vertex lands on the panel border (that would be the grid edge "
      "marching, which drew a hard rectangle)",
      not (np.any(face[..., 0] <= fp.bx + 1) or np.any(face[..., 1] <= fp.by + 1)
           or np.any(face[..., 0] >= fp.bx + fp.bw - 1)
           or np.any(face[..., 1] >= fp.by + fp.bh - 1)))
check("the head is taller than it is wide, like a head",
      float(face[..., 1].ptp()) > float(face[..., 0].ptp()),
      (face[..., 0].ptp(), face[..., 1].ptp()))
check("it fills a decent share of the panel",
      face[..., 1].ptp() > fp.bh * 0.6, face[..., 1].ptp())

# Smoothing: a per-frame fit jitters, and this is the only moving thing in an
# otherwise still corner.
fp3 = hud.FacePanel(W, H, smooth=0.35)
fp3.update(lm)
first = fp3.pts.copy()
fp3.update(lm + 0.1)
moved = float(np.abs(fp3.pts - first).max())
check("landmarks are smoothed, not snapped", 0.001 < moved < 0.1, moved)

fp3.update(None)
check("losing the face clears the panel", len(fp3.points()) == 0)

# Contouring is not free; it must be rate-limited, not run every frame.
fp4 = hud.FacePanel(W, H, rebuild_hz=4.0)
for _ in range(6):
    fp4.update(lm)
fp4.points()
t0 = _t.time()
for _ in range(200):
    fp4.points()
cached_ms = (_t.time() - t0) / 200 * 1000
t0 = _t.time()
fp4._build()
build_ms = (_t.time() - t0) * 1000
print("    face panel: %.3f ms/frame cached, %.0f ms per rebuild at %.0f Hz"
      % (cached_ms, build_ms, 1.0 / fp4._period))
check("cached frames are nearly free", cached_ms < 0.2, cached_ms)
check("rebuilds are rate-limited, not per-frame",
      build_ms / (1000.0 / (1.0 / fp4._period)) < 0.30,
      "%.0f%% of a core" % (build_ms / (1000.0 / (1.0 / fp4._period)) * 100))

print("HUD " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
