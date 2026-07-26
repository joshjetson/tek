import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import time
import tekcam

t0 = time.time()
cam = tekcam.Tracker().start()
fol = tekcam.Follower()
print("tracking for 20s - move around in front of the camera")
print("%6s %8s %8s %8s %7s %7s %7s" %
      ("t", "present", "raw_x", "raw_y", "gaze_x", "gaze_y", "head"))
last = time.time()
seen = 0
while time.time() - t0 < 20:
    now = time.time()
    dt, last = now - last, now
    st = cam.state()
    gx, gy, hy = fol.update(st, dt)
    if st["present"]:
        seen += 1
    if int((now - t0) * 4) % 4 == 0:
        print("%6.1f %8s %8.2f %8.2f %7.2f %7.2f %7.2f"
              % (now - t0, st["present"], st["x"], st["y"], gx, gy, hy))
    time.sleep(0.25)
cam.stop()
print("\ncamera frames: %d   detections: %d   detect rate: %.1f/s"
      % (cam.frames, cam.detections, cam.detections / 20.0))
