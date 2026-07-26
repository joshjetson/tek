import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import cv2
import tekfdl
import tekrun

# 1. does the DISK CACHE match a fresh build?
vc, ec, nc, warm = tekrun.load_geometry()
print("cache: warm=%s  %d verts %d edges" % (warm, len(vc), len(ec)))
vf, ef, nf = tekfdl.build(lips=False)
print("fresh:              %d verts %d edges" % (len(vf), len(ef)))
if len(vc) == len(vf):
    print("  vert max diff: %.6g" % np.abs(vc - vf).max())
    print("  edge identical: %s" % np.array_equal(ec, ef))
else:
    print("  *** CACHE DIFFERS FROM FRESH BUILD ***")

# 2. grab what is actually on the panel right now
with open("/sys/class/graphics/fb0/virtual_size") as f:
    w, h = [int(x) for x in f.read().strip().split(",")]
with open("/sys/class/graphics/fb0/stride") as f:
    stride = int(f.read().strip())
raw = np.fromfile("/dev/fb0", dtype=np.uint8, count=h * stride)
live = raw.reshape(h, stride // 4, 4)[:, :w, :3]

off = cv2.imread("/home/super/tek_out/diag_lips_false.png")
pair = np.hstack([off, live])
cv2.putText(pair, "OFFLINE build", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(pair, "LIVE framebuffer", (w + 12, 26), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (120, 255, 150), 2, cv2.LINE_AA)
cv2.imwrite("/home/super/tek_out/live_vs_offline.png", pair)


def lit(i):
    return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) > 40


print("\nlit px  offline=%d  live=%d" % (lit(off).sum(), lit(live).sum()))
# column and row profiles expose a rectangular void as a flat-bottomed notch
for name, img in (("offline", off), ("live", live)):
    m = lit(img)
    rows = m.sum(1)
    band = np.where(rows > 0)[0]
    print("%s: rows %d..%d" % (name, band.min(), band.max()))
print("wrote live_vs_offline.png")
