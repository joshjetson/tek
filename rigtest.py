import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import time
import numpy as np
import cv2
import tekrig
import tekhead
import tekvector as tv

face = tekrig.Face(verbose=True)
W, H = 500, 700


def shot(name):
    face.express(name, blend=1e-6)
    face._blend = 1.0
    face.controls = dict(tekrig.DEFAULTS)
    face.controls.update(tekrig.EXPRESSIONS[name])
    v, e, n = face.update(0.0)
    pts = tekhead.build_pts_culled(v, e, n, W, H, (-0.03, 0.0, 0),
                                   16.0, -0.05, "and", fov=9.3)
    img = tv.draw([(p[0], p[1]) for p in pts], W, H)
    cv2.putText(img, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (90, 240, 130), 2, cv2.LINE_AA)
    return img


names = ["neutral", "attentive", "thinking", "happy",
         "concerned", "surprised", "confused", "asleep"]
t0 = time.time()
tiles = [shot(n) for n in names]
print("8 expressions rendered in %.1fs (cold cache)" % (time.time() - t0))
cv2.imwrite("/home/super/tek_out/expressions.png",
            np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))

# warm-cache frame cost
face.express("happy", blend=0.01)
face.update(0.0)
t = time.perf_counter()
for i in range(60):
    face.update(i * 0.033, 0.033)
print("warm update: %.2f ms/frame" % ((time.perf_counter() - t) / 60 * 1000))
print("cache (hits, misses):", face.stats())
print("wrote expressions.png")
