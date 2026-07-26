"""Two structural ideas: work in uint8 (half the memory traffic), and put the
composite on the idle GPU."""
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


def T(label, fn, n_=25):
    try:
        fn()
    except Exception as exc:
        print("  %-40s FAILED: %s" % (label, str(exc)[:40]))
        return 1e9, None
    t = time.perf_counter()
    for _ in range(n_):
        r = fn()
    ms = (time.perf_counter() - t) / n_ * 1000
    print("  %-40s %6.2f ms" % (label, ms))
    return ms, r


print("baseline")
b, _ = T("render_bgra (current)",
         lambda: phosphor.render_bgra(pts, W, H, (vig, grain, lut)))

print("\nA. uint8 PIPELINE (half the memory traffic of float32)")
vig8 = np.clip(vig * (255.0 / (255.0 / phosphor.MAX_I)), 0, 255).astype(np.uint8)


def u8_pipeline():
    beam = np.zeros((H, W), np.uint8)
    cv2.polylines(beam, pts, False, 255, 1, cv2.LINE_AA)
    hw, hh = W // 2, H // 2
    s = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
    q = cv2.resize(s, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
    w_ = cv2.addWeighted(cv2.GaussianBlur(q, (9, 9), 0), 0.40,
                         cv2.GaussianBlur(q, (21, 21), 0), 0.30, 0)
    g = cv2.addWeighted(cv2.GaussianBlur(s, (5, 5), 0), 0.55,
                        cv2.resize(w_, (hw, hh), interpolation=cv2.INTER_LINEAR),
                        1.0, 0)
    glow = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    inten = cv2.add(beam, glow)
    return cv2.LUT(cv2.cvtColor(inten, cv2.COLOR_GRAY2BGRA), lut)


u, _ = T("full uint8 chain", u8_pipeline)

print("\nB. CUDA COMPOSITE (the GPU is completely idle)")
beam = np.zeros((H, W), np.float32)
cv2.polylines(beam, pts, False, 1.0, 1, cv2.LINE_AA)
try:
    gpu_vig = cv2.cuda_GpuMat(); gpu_vig.upload(vig)
    gf = cv2.cuda.createGaussianFilter(cv2.CV_32FC1, cv2.CV_32FC1, (5, 5), 0)

    def cuda_chain():
        g = cv2.cuda_GpuMat(); g.upload(beam)
        gs = cv2.cuda.resize(g, (W // 2, H // 2))
        gb = gf.apply(gs)
        gu = cv2.cuda.resize(gb, (W, H))
        acc = cv2.cuda.addWeighted(g, 1.15, gu, 1.0, 0)
        acc = cv2.cuda.multiply(acc, gpu_vig)
        return acc.download()

    c, _ = T("upload + blur + composite + download", cuda_chain)
    T("  - upload only", lambda: cv2.cuda_GpuMat().upload(beam))
    gm = cv2.cuda_GpuMat(); gm.upload(beam)
    T("  - download only", lambda: gm.download())
except Exception as exc:
    print("  cuda unavailable: %s" % str(exc)[:60])

print("\nC. RESOLUTION TRADE (render smaller, upscale to panel)")
for scale in (0.85, 0.75):
    w2, h2 = int(W * scale), int(H * scale)
    p2 = geometry.build_pts_culled(v, e, n, w2, h2, (-0.045, 0.1, 0.0),
                                   16.0, -0.05, "and", 11.4 * scale)
    st2 = phosphor.build_statics(w2, h2)

    def smaller(w2=w2, h2=h2, p2=p2, st2=st2):
        f = phosphor.render_bgra(p2, w2, h2, st2)
        return cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)

    T("render @%d%% then upscale (%dx%d)" % (scale * 100, w2, h2), smaller)

print("\nbaseline was %.2f ms" % b)
