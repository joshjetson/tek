"""Candidate optimisations for the render stage, measured against the current one."""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tekdromo import app, geometry, phosphor

W, H = 1024, 600
v, e, n, _ = app.load_geometry()
pts = geometry.build_pts_culled(v, e, n, W, H, (-0.045, 0.1, 0.0),
                                16.0, -0.05, "and", 11.4)
vig, grain, lut = phosphor.build_statics(W, H)[:3]
beam = np.zeros((H, W), np.float32)
cv2.polylines(beam, pts, False, 1.0, 1, cv2.LINE_AA)
hw, hh = W // 2, H // 2
small = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
quart = cv2.resize(small, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
wide = cv2.GaussianBlur(quart, (9, 9), 0) * .40 + cv2.GaussianBlur(quart, (21, 21), 0) * .30
glow = cv2.resize(cv2.GaussianBlur(small, (5, 5), 0) * .55
                  + cv2.resize(wide, (hw, hh), interpolation=cv2.INTER_LINEAR),
                  (W, H), interpolation=cv2.INTER_LINEAR)
idx8 = cv2.convertScaleAbs(cv2.add(cv2.multiply(beam, 1.15), glow))


def T(label, fn, n_=30):
    fn()
    t = time.perf_counter()
    for _ in range(n_):
        r = fn()
    ms = (time.perf_counter() - t) / n_ * 1000
    print("  %-40s %6.2f ms" % (label, ms))
    return ms, r


print("A. INTENSITY -> BGRA")
base, ref = T("cv2.LUT on 4ch (current)",
              lambda: cv2.LUT(cv2.cvtColor(idx8, cv2.COLOR_GRAY2BGRA), lut))
lut2d = lut.reshape(256, 4)
ms, out = T("numpy fancy-index lut2d[idx]", lambda: lut2d[idx8])
print("     identical: %s" % np.array_equal(out, ref))
ms2, out2 = T("np.take(lut2d, idx, axis=0)",
              lambda: np.take(lut2d, idx8, axis=0))
print("     identical: %s" % np.array_equal(out2, ref))

print("\nB. COMPOSITE ARITHMETIC")
vig115 = (vig * 1.15).astype(np.float32)


def current():
    off = np.random.randint(0, H)
    i = cv2.add(cv2.multiply(beam, 1.15), glow)
    i = cv2.add(cv2.multiply(i, vig), grain[off:off + H])
    return cv2.convertScaleAbs(i)


def fused():
    # (beam*1.15 + glow)*vig + grain  ==  beam*(1.15*vig) + glow*vig + grain
    # one fewer full-frame pass by pre-multiplying the constant into vig
    off = np.random.randint(0, H)
    i = cv2.multiply(beam, vig115)
    i = cv2.scaleAdd(glow, 1.0, i)          # fused multiply-add
    i = cv2.multiply(i, vig)
    return cv2.convertScaleAbs(cv2.add(i, grain[off:off + H]))


c0, r0 = T("current chain", current)
c1, r1 = T("pre-multiplied vignette", fused)

print("\nC. BLOOM")
def bloom_now():
    s = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
    q = cv2.resize(s, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
    w_ = cv2.GaussianBlur(q, (9, 9), 0) * .40 + cv2.GaussianBlur(q, (21, 21), 0) * .30
    g = cv2.GaussianBlur(s, (5, 5), 0) * .55 + cv2.resize(w_, (hw, hh),
                                                          interpolation=cv2.INTER_LINEAR)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)


def bloom_boxed():
    # box blur repeated ~= gaussian, and cv2.blur is far cheaper
    s = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
    q = cv2.resize(s, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
    w_ = cv2.blur(cv2.blur(q, (7, 7)), (7, 7)) * .40 + \
        cv2.blur(cv2.blur(q, (15, 15)), (15, 15)) * .30
    g = cv2.blur(s, (5, 5)) * .55 + cv2.resize(w_, (hw, hh),
                                               interpolation=cv2.INTER_LINEAR)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)


b0, _ = T("gaussian pyramid (current)", bloom_now)
b1, gb = T("box-blur pyramid", bloom_boxed)
print("     mean abs difference vs gaussian: %.4f" % np.abs(gb - glow).mean())

print("\nD. DRAWING")
d0, _ = T("polylines LINE_AA (current)",
          lambda: cv2.polylines(np.zeros((H, W), np.float32), pts, False, 1.0, 1, cv2.LINE_AA))
d1, _ = T("polylines LINE_8 (no antialias)",
          lambda: cv2.polylines(np.zeros((H, W), np.float32), pts, False, 1.0, 1, cv2.LINE_8))
d2, _ = T("polylines LINE_AA on uint8",
          lambda: cv2.polylines(np.zeros((H, W), np.uint8), pts, False, 255, 1, cv2.LINE_AA))

print("\nSUMMARY of achievable savings (ms/frame)")
print("  LUT      : %5.2f -> %5.2f   save %.2f" % (base, ms, base - ms))
print("  composite: %5.2f -> %5.2f   save %.2f" % (c0, c1, c0 - c1))
print("  bloom    : %5.2f -> %5.2f   save %.2f" % (b0, b1, b0 - b1))
print("  draw     : %5.2f -> %5.2f   save %.2f (quality cost)" % (d0, d1, d0 - d1))
