import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import time
import numpy as np
import cv2
import tekfdl
import tekhead
import tekvector as tv

t = time.perf_counter()
v, e, n = tekfdl.build()
print("build %.1fs -> %d verts %d edges" % (time.perf_counter() - t, len(v), len(e)))

W, H = 1024, 600
outs = []
for ry in (0.0, 0.50):
    pts = tekhead.build_pts_culled(v, e, n, W, H, (-0.03, ry, 0),
                                   16.0, -0.05, "and", fov=11.4)
    outs.append(tv.draw([(p[0], p[1]) for p in pts], W, H))
cv2.imwrite("/home/super/tek_out/fdl.png", np.hstack(outs))

z = cv2.resize(outs[0][110:430, 590:880], None, fx=3, fy=3,
               interpolation=cv2.INTER_NEAREST)
cv2.imwrite("/home/super/tek_out/ear_zoom.png", z)
print("wrote fdl.png and ear_zoom.png")
