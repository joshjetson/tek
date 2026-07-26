import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import cv2
import tekfdl
import tekhead
import tekvector as tv

W, H = 1024, 600
POSE = (-0.045, 0.0, 0.0)


def render(v, e, n):
    pts = tekhead.build_pts_culled(v, e, n, W, H, POSE, 16.0, -0.05, "and", 11.4)
    return tv.draw([(p[0], p[1]) for p in pts], W, H)


def lit(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 40


# lips=False is the DEFAULT in build() -- it punches MOUTH_BOX out and expects
# something else to supply it. lips=True keeps the mouth in the field.
for tag, kw in (("lips_false", dict(lips=False)), ("lips_true", dict(lips=True))):
    v, e, n = tekfdl.build(**kw)
    img = render(v, e, n)
    cv2.imwrite("/home/super/tek_out/diag_%s.png" % tag, img)
    print("%-11s -> %5d edges, %6d lit px" % (tag, len(e), lit(img).sum()))

a = lit(cv2.imread("/home/super/tek_out/diag_lips_true.png"))
b = lit(cv2.imread("/home/super/tek_out/diag_lips_false.png"))
lost = a & ~b
print("\nlost by lips=False: %d px" % lost.sum())
if lost.sum():
    ys, xs = np.where(lost)
    print("  bbox  x %d..%d  y %d..%d" % (xs.min(), xs.max(), ys.min(), ys.max()))

# Where does MOUTH_BOX actually land on screen?
from tekvector import project, rotate
R = rotate(np.eye(3, dtype=np.float32), *POSE)
x0, x1, y0, y1 = tekfdl.MOUTH_BOX
corners = np.array([[x0, y0 + 0.20, 0.6], [x1, y0 + 0.20, 0.6],
                    [x1, y1 + 0.20, 0.6], [x0, y1 + 0.20, 0.6]], np.float32)
q, _ = project(corners @ R, W, H, 16.0, 11.4)
print("\nMOUTH_BOX projects to x %.0f..%.0f  y %.0f..%.0f"
      % (q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()))

# And find any large rectangular void in the lips=False render
img = cv2.imread("/home/super/tek_out/diag_lips_false.png")
m = lit(img).astype(np.uint8)
head = m.copy()
head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
holes = (head > 0) & (m == 0)
nl, lab, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
big = sorted(range(1, nl), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:4]
print("\nlargest unlit areas inside the head:")
for i in big:
    x, y, w, h, area = st[i]
    print("  area=%6d  x %d..%d  y %d..%d  (%dx%d)"
          % (area, x, x + w, y, y + h, w, h))
