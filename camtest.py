import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import time
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit("could not open /dev/video0")

# MJPG lets a UVC cam hit useful resolutions over USB2; YUYV is bandwidth-bound
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("negotiated: %dx%d @ %.0f fps  fourcc=%s"
      % (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
         cap.get(cv2.CAP_PROP_FPS),
         "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))))

for _ in range(5):
    cap.read()                                  # let exposure settle

t = time.perf_counter()
n = 0
frame = None
for _ in range(40):
    ok, f = cap.read()
    if ok:
        frame = f
        n += 1
dt = time.perf_counter() - t
print("captured %d frames in %.2fs -> %.1f fps" % (n, dt, n / dt))

if frame is None:
    raise SystemExit("no frames captured")
print("frame shape:", frame.shape, "mean brightness: %.1f" % frame.mean())
cv2.imwrite("/home/super/tek_out/cam_raw.png", frame)

# face detection with the stock Haar cascade
# cv2.data does not exist in a source build; cascades ship under the
# install prefix instead
xml = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
det = cv2.CascadeClassifier(xml)
print("cascade loaded:", not det.empty())
g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
g = cv2.equalizeHist(g)
t = time.perf_counter()
faces = det.detectMultiScale(g, 1.2, 5, minSize=(60, 60))
print("detect: %.1f ms  faces=%d" % ((time.perf_counter() - t) * 1000, len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cx = (x + w / 2) / frame.shape[1] * 2 - 1
    cy = 1 - (y + h / 2) / frame.shape[0] * 2
    print("  face at gaze_x=%+.2f gaze_y=%+.2f  (size %dpx)" % (cx, cy, w))
cv2.imwrite("/home/super/tek_out/cam_faces.png", frame)
cap.release()
print("wrote cam_raw.png and cam_faces.png")
