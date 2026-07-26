"""End-to-end smoke test: does a frame actually render?

This exists because the other tests did not exercise the render path. A stale
function-level `from tekvector import ...` inside geometry.build_pts_culled
survived the refactor, and only broke once the legacy modules were deleted -
after the equivalence test had already passed. The service then crash-looped
at 0 fps with 198 errors, and the frame-loop's own error suppression (first 3
tracebacks only) meant the journal showed nothing useful.

Run this after any structural change.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo import app, geometry, phosphor, rig, speech

W, H = 640, 400
fail = 0

# every module must import with nothing left dangling
import importlib
for m in ("anatomy", "field", "contour", "rig", "geometry", "phosphor",
          "framebuffer", "camera", "speech", "app"):
    try:
        importlib.import_module("tekdromo." + m)
    except Exception as exc:
        print("import tekdromo.%-12s FAIL %s" % (m, exc))
        fail += 1
print("all modules import      : %s" % (fail == 0))

v, e, n, warm = app.load_geometry()
print("geometry                : %d edges (%s)"
      % (len(e), "cached" if warm else "built"))

face = rig.Face()
face.static = (v, e, n)
face._edge_in = {k: rig.Face._inside_mask(face.static, r.box)
                 for k, r in face.regions.items()}
face.express("neutral", blend=0.01)
statics = phosphor.build_statics(W, H)

# render a short run covering rest, speech and an expression change
t0 = time.time()
lit_seen = []
for i in range(12):
    t = i * 0.1
    if i == 6:
        face.express("happy", blend=0.2)
    face.speak(*speech.synthetic(t))
    try:
        vv, ee, nn = face.update(t, 0.1)
        pts = geometry.build_pts_culled(vv, ee, nn, W, H, (-0.045, 0.1, 0.0),
                                        16.0, -0.05, "and", 11.4)
        frame = phosphor.render_bgra(pts, W, H, statics)
        lit_seen.append(int((frame[..., 1] > 40).sum()))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("frame %d FAILED" % i)
        fail += 1
        break

if lit_seen:
    print("frames rendered         : %d, %.0f ms/frame"
          % (len(lit_seen), (time.time() - t0) / len(lit_seen) * 1000))
    print("lit pixels min/max      : %d / %d" % (min(lit_seen), max(lit_seen)))
    if min(lit_seen) < 500:
        print("  *** a frame came out nearly blank ***")
        fail += 1
    if len(set(lit_seen)) == 1:
        print("  *** nothing changed across frames - is anything animating? ***")
        fail += 1
else:
    fail += 1

print("\n%s" % ("SMOKE OK" if not fail else "%d FAILURES" % fail))
sys.exit(1 if fail else 0)
