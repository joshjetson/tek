import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tek_out")
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import cv2
from tekdromo import app as tekrun
from tekdromo import rig as tekrig
from tekdromo import geometry as tekhead
from tekdromo import phosphor as tv

W, H = 1024, 600
POSE = (-0.045, 0.0, 0.0)


def render(v, e, n):
    pts = tekhead.build_pts_culled(v, e, n, W, H, POSE, 16.0, -0.05, "and", 11.4)
    return tv.render_bgra(pts, W, H, tv.build_statics(W, H))[..., :3]


v, e, n, _ = tekrun.load_geometry()[:3] + (True,)
A = render(v, e, n)                                   # no rig at all

face = tekrig.Face()
face.static = (v, e, n)      # Face punches per-frame now; pre-punching here
                             # removed the edges twice
B = render(*face.update(0.0))                         # rig: punch + regions

cv2.imwrite(os.path.join(OUT, "ab_A_norig.png"), A)
cv2.imwrite(os.path.join(OUT, "ab_B_rig.png"), B)


def lit(i):
    return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) > 40


a, b = lit(A), lit(B)
lost = a & ~b
gain = b & ~a
print("A (no rig) lit=%d   B (rig) lit=%d" % (a.sum(), b.sum()))
print("lost by rig: %d px    gained: %d px" % (lost.sum(), gain.sum()))

# a rectangular void shows as contiguous rows/cols of heavy loss
rows = lost.sum(1)
cols = lost.sum(0)
hot_r = np.where(rows > max(3, rows.max() * 0.30))[0]
hot_c = np.where(cols > max(3, cols.max() * 0.30))[0]
if len(hot_r) and len(hot_c):
    print("heavy-loss band: rows %d..%d   cols %d..%d"
          % (hot_r.min(), hot_r.max(), hot_c.min(), hot_c.max()))

vis = np.zeros((H, W, 3), np.uint8)
vis[..., 1] = np.where(b, 120, 0)
vis[lost] = (0, 0, 255)      # red  = rig removed it
vis[gain] = (255, 120, 0)    # blue = rig added it
# draw the region boxes so the correspondence is unambiguous
from tekvector import project, rotate
R = rotate(np.eye(3, dtype=np.float32), *POSE)
for name, r in tekrig.REGIONS.items():
    x0, x1, y0, y1 = face.regions[name].box
    c = np.array([[x0, y0 + .20, .70], [x1, y0 + .20, .70],
                  [x1, y1 + .20, .70], [x0, y1 + .20, .70]], np.float32)
    q, _ = project(c @ R, W, H, 16.0, 11.4)
    q = q.astype(int)
    cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (0, 255, 255), 1)
    cv2.putText(vis, name, (q[:, 0].min() + 3, q[:, 1].min() + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(os.path.join(OUT, "ab_diff.png"), vis)
print("wrote ab_A_norig / ab_B_rig / ab_diff (red=removed, blue=added, yellow=boxes)")
