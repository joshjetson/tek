import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import time
import cv2
import numpy as np
import tekcam

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(8):
    cap.read()
ok, frame = cap.read()
cap.release()
print("frame:", frame.shape, "mean=%.1f" % frame.mean())
cv2.imwrite("/home/super/tek_out/cam_diag.png", frame)

haar = cv2.CascadeClassifier(tekcam.CASCADE)
lbp = cv2.CascadeClassifier(tekcam.CASCADE_FAST)
print("haar loaded:", not haar.empty(), " lbp loaded:", not lbp.empty())

for scale in (1.0, 0.5):
    if scale == 1.0:
        img = frame
    else:
        img = cv2.resize(frame, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for name, det, params in (("haar", haar, (1.2, 5, 60)),
                              ("haar-loose", haar, (1.15, 3, 30)),
                              ("lbp", lbp, (1.15, 4, 36)),
                              ("lbp-loose", lbp, (1.05, 2, 24))):
        if det.empty():
            continue
        sf, mn, ms = params
        ms = max(12, int(ms * scale))
        t = time.perf_counter()
        f = det.detectMultiScale(g, sf, mn, minSize=(ms, ms))
        ms_t = (time.perf_counter() - t) * 1000
        print("  scale=%.2f %-11s -> %d faces  %.0f ms" % (scale, name, len(f), ms_t))
