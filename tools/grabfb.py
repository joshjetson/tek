import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
# Accept a destination path. It used to hardcode tek_out/screen.png and
# silently ignore any argument passed to it, so `grabfb.py /tmp/x.png` reported
# success and wrote somewhere else entirely.
_DEST = sys.argv[1] if len(sys.argv) > 1 else None
OUT = (os.path.dirname(os.path.abspath(_DEST)) if _DEST else
       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tek_out"))
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import cv2

with open("/sys/class/graphics/fb0/virtual_size") as f:
    w, h = [int(x) for x in f.read().strip().split(",")]
with open("/sys/class/graphics/fb0/stride") as f:
    stride = int(f.read().strip())
raw = np.fromfile("/dev/fb0", dtype=np.uint8, count=h * stride)
img = raw.reshape(h, stride // 4, 4)[:, :w, :3]
cv2.imwrite(_DEST or os.path.join(OUT, "screen.png"), img)
print("captured %dx%d  mean=%.1f  max=%d  nonzero=%.2f%%"
      % (w, h, img.mean(), img.max(), 100.0 * (img.max(2) > 12).mean()))
