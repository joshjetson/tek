"""Break down the render stage - it is the steady-state floor."""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tekdromo import app, geometry, phosphor, rig

W, H = 1024, 600
v, e, n, _ = app.load_geometry()
pts = geometry.build_pts_culled(v, e, n, W, H, (-0.045, 0.1, 0.0),
                                16.0, -0.05, "and", 11.4)
vig, grain, lut = phosphor.build_statics(W, H)[:3]
print("%d segments, %dx%d" % (len(pts), W, H))


def T(label, fn, n_=25):
    fn()
    t = time.perf_counter()
    for _ in range(n_):
        r = fn()
    print("  %-34s %6.2f ms" % (label, (time.perf_counter() - t) / n_ * 1000))
    return r


beam = np.zeros((H, W), np.float32)
T("polylines (draw the vectors)",
  lambda: cv2.polylines(np.zeros((H, W), np.float32), pts, False, 1.0, 1, cv2.LINE_AA))
cv2.polylines(beam, pts, False, 1.0, 1, cv2.LINE_AA)

hw, hh = W // 2, H // 2


def bloom():
    small = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
    quart = cv2.resize(small, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
    wide = cv2.GaussianBlur(quart, (9, 9), 0) * 0.40 + \
        cv2.GaussianBlur(quart, (21, 21), 0) * 0.30
    glow_s = cv2.GaussianBlur(small, (5, 5), 0) * 0.55 + \
        cv2.resize(wide, (hw, hh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(glow_s, (W, H), interpolation=cv2.INTER_LINEAR)


glow = T("bloom pyramid", bloom)
T("  - downsample only",
  lambda: cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA))
small = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
T("  - gaussian 5x5 @half", lambda: cv2.GaussianBlur(small, (5, 5), 0))
quart = cv2.resize(small, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
T("  - gaussian 21x21 @quarter", lambda: cv2.GaussianBlur(quart, (21, 21), 0))
T("  - upsample half->full",
  lambda: cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR))


def composite():
    off = np.random.randint(0, H)
    inten = cv2.add(cv2.multiply(beam, 1.15), glow)
    inten = cv2.add(cv2.multiply(inten, vig), grain[off:off + H])
    idx = cv2.convertScaleAbs(inten)
    return cv2.LUT(cv2.cvtColor(idx, cv2.COLOR_GRAY2BGRA), lut)


T("composite (mul/add/LUT)", composite)
T("  - cv2.multiply full", lambda: cv2.multiply(beam, vig))
T("  - convertScaleAbs", lambda: cv2.convertScaleAbs(beam))
idx = cv2.convertScaleAbs(beam)
T("  - cvtColor GRAY2BGRA", lambda: cv2.cvtColor(idx, cv2.COLOR_GRAY2BGRA))
bgra = cv2.cvtColor(idx, cv2.COLOR_GRAY2BGRA)
T("  - cv2.LUT 4ch", lambda: cv2.LUT(bgra, lut))

print("\nTOTAL render_bgra:")
T("  render_bgra", lambda: phosphor.render_bgra(pts, W, H, (vig, grain, lut)))
print("\ncv2 threads: %d   cores: %d" % (cv2.getNumThreads(), os.cpu_count()))
print("cv2 optimised: %s" % cv2.useOptimized())
