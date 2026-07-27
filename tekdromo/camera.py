#!/usr/bin/env python3
"""
tekcam - camera face tracking for TEKDROMO.

Runs detection in a BACKGROUND THREAD, not the render loop. Haar detect costs
~147 ms on this board; doing it inline would drop the display from 41 fps to 7.
A face does not move fast, so detecting at ~6 Hz and interpolating between
detections looks identical and costs the renderer nothing.

Output is deliberately already in the rig's units: target x/y in -1..+1, which
is exactly the range of the gaze_x / gaze_y controls. No mapping layer.

Design notes:
  * Detection runs on a downscaled frame - the cascade cost is roughly linear
    in pixel count and a face 320px wide is still ~100px at quarter scale.
  * Tracking is smoothed with a critically-damped follow rather than a lerp, so
    it settles without overshoot and without the rubber-banding a plain
    exponential gives on fast moves.
  * Losing a face does not immediately reset the gaze - it holds, then drifts
    back to centre. A person stepping briefly out of frame should not make the
    head snap forward.
"""
import os
import threading
import time

import cv2
import numpy as np

LANDMARK_MODEL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "lbfmodel.yaml")

CASCADE = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FAST = "/usr/local/share/opencv4/lbpcascades/lbpcascade_frontalface_improved.xml"


class Tracker:
    """Background face tracker.

        cam = Tracker(); cam.start()
        s = cam.state()      # -> dict(present, x, y, size, age)
    """

    def __init__(self, device=0, width=640, height=480, detect_scale=0.5,
                 interval=0.15, fast=True, mirror=True, landmarks=True):
        self.device = device
        self.size = (width, height)
        self.detect_scale = detect_scale
        self.interval = interval
        self.mirror = mirror
        self.cascade = cv2.CascadeClassifier(CASCADE_FAST if fast else CASCADE)
        if self.cascade.empty():
            self.cascade = cv2.CascadeClassifier(CASCADE)
        self._lock = threading.Lock()
        self._raw = None            # newest detection (x, y, size, t)
        self._t = None
        self._run = False
        self._err = 0
        self.frames = 0
        self.detections = 0
        # 68 facial landmarks, normalised to 0..1 of the frame so nothing
        # downstream needs to know the camera's resolution.
        self.landmarks = None
        self.landmarks_at = 0.0
        self.name = None            # who the recogniser thinks this is
        self.face_crop = None       # greyscale crop, for enrolment
        try:
            from . import recog
            self._recog = recog.Recogniser()
        except Exception:
            self._recog = None
        self._facemark = self._load_facemark() if landmarks else None
        self.last_frame = None
        self.last_faces = 0

    @staticmethod
    def _load_facemark():
        """LBF, not Kazemi. Measured on this board: LBF 10.7 ms per fit,
        Kazemi 124 ms for the same 68 points and the same accuracy budget.
        Downscaling does not help Kazemi - its cost is a fixed number of tree
        traversals, not pixels (124 ms at 640x480, 109 ms at 224x168) - so the
        model choice is the whole difference between affordable and not."""
        if not os.path.exists(LANDMARK_MODEL):
            return None
        try:
            fm = cv2.face.createFacemarkLBF()
            fm.loadModel(LANDMARK_MODEL)
            return fm
        except Exception:
            return None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._run = False

    # -- worker ------------------------------------------------------------
    def _loop(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            # MJPG is required: YUYV at 640x480x30 saturates USB 2.0
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last = 0.0
            while self._run:
                # grab() pulls a frame off the queue WITHOUT decoding it;
                # retrieve() is what costs the JPEG decode. Decoding every
                # frame just to discard it halved the display's framerate
                # (42 -> 21 fps), so only decode on a detection tick.
                if not cap.grab():
                    self._err += 1
                    time.sleep(0.2)
                    if self._err > 40:            # camera unplugged: reopen
                        cap.release()
                        time.sleep(1.0)
                        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                        cap.set(cv2.CAP_PROP_FOURCC,
                                cv2.VideoWriter_fourcc(*"MJPG"))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                        self._err = 0
                    continue
                self._err = 0
                self.frames += 1
                now = time.time()
                if now - last < self.interval:
                    time.sleep(0.004)             # yield: only 4 cores here
                    continue
                last = now
                ok, frame = cap.retrieve()
                if ok:
                    self._detect(frame, now)
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()

    def _detect(self, frame, now):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, None, fx=self.detect_scale, fy=self.detect_scale,
                           interpolation=cv2.INTER_AREA)
        g = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(g, 1.15, 4, minSize=(36, 36))
        # Keep the frame whether or not a face was found. It used to be stored
        # only on a hit, which meant "look at the camera" had nothing to look
        # at in an empty room - and an empty room is a perfectly good answer.
        with self._lock:
            self.last_frame = frame
            self.last_faces = len(faces)
        if len(faces) == 0:
            return
        # biggest face wins - the nearest person is the one being addressed
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        self._fit_landmarks(small, (x, y, fw, fh), now)
        self._identify(g, (x, y, fw, fh))
        if self.detections % 40 == 0:
            self._check_gallery()
        cx = (x + fw * 0.5) / small.shape[1]
        cy = (y + fh * 0.5) / small.shape[0]
        nx = cx * 2.0 - 1.0
        ny = 1.0 - cy * 2.0
        if self.mirror:
            nx = -nx          # camera faces the user, so mirror to match them
        with self._lock:
            self._raw = (nx, ny, fw / small.shape[1], now)
            self.detections += 1

    def _fit_landmarks(self, small, rect, now):
        """68 points for the chosen face, normalised to the frame.

        Fitted on the SAME downscaled image the detector used - the rectangle
        is in that image's coordinates, and handing the fitter a full-size
        frame with small-image boxes silently produces landmarks in the wrong
        place rather than an error.
        """
        if self._facemark is None:
            return
        try:
            ok, shapes = self._facemark.fit(small, np.array([rect]))
        except Exception:
            return
        if not ok or not len(shapes):
            return
        pts = np.array(shapes[0]).reshape(-1, 2).astype(np.float32)
        h, w = small.shape[:2]
        pts[:, 0] /= float(w)
        pts[:, 1] /= float(h)
        if self.mirror:
            pts[:, 0] = 1.0 - pts[:, 0]     # match the gaze convention
        with self._lock:
            self.landmarks = pts
            self.landmarks_at = now

    def _check_gallery(self):
        """Reload if someone has been enrolled since we started.

        Polled rather than signalled: the enrolling process is the CLI, the
        recogniser lives in the display, and a directory mtime is a cheaper
        channel between them than another socket.
        """
        try:
            from . import recog
            m = os.path.getmtime(recog.GALLERY)
        except OSError:
            return
        if m != getattr(self, "_gallery_mtime", None):
            self._gallery_mtime = m
            if self._recog is not None:
                n = self._recog.reload()
                print("recogniser: %d samples, %s"
                      % (n, ", ".join(recog.people()) or "nobody"), flush=True)

    def _identify(self, gray_small, rect):
        """Name the detected face, from the same greyscale the detector used."""
        if self._recog is None:
            return
        x, y, w, h = rect
        crop = gray_small[max(0, y):y + h, max(0, x):x + w]
        if crop.size < 100:
            return
        try:
            name, _dist = self._recog.predict(crop)
        except Exception:
            name = None
        with self._lock:
            self.face_crop = crop
            self.name = name

    def who(self):
        with self._lock:
            return self.name

    def crop(self):
        with self._lock:
            return None if self.face_crop is None else self.face_crop.copy()

    def relearn(self):
        """Pick up newly enrolled faces without a restart."""
        if self._recog is not None:
            return self._recog.reload()
        return 0

    def face_points(self, hold=1.5):
        """Latest landmarks, or None if stale. Never blocks."""
        with self._lock:
            pts, at = self.landmarks, self.landmarks_at
        if pts is None or time.time() - at > hold:
            return None
        return pts

    # -- consumer ----------------------------------------------------------
    def snapshot(self, path):
        """Write the most recent decoded frame to disk, for the Brain to look at.

        Only detection ticks decode a frame (grab/retrieve are split to keep the
        renderer fast), so this is the newest frame that was actually looked at
        - which is the one the event refers to.
        """
        with self._lock:
            frame = self.last_frame
        if frame is None:
            return None
        try:
            d = os.path.dirname(path)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            cv2.imwrite(path, frame)
            return path
        except Exception:
            return None

    def state(self, hold=1.2):
        """present/x/y/size/age. `hold` = seconds a lost face stays 'present'."""
        with self._lock:
            raw = self._raw
        if raw is None:
            return dict(present=False, x=0.0, y=0.0, size=0.0, age=None)
        nx, ny, size, t = raw
        age = time.time() - t
        return dict(present=age < hold, x=nx, y=ny, size=size, age=age)


class Follower:
    """Turns tracker output into smooth rig values.

    Critically damped rather than a plain lerp: settles without overshoot and
    without the rubber-band feel an exponential gives on fast moves. Gaze leads
    and the head follows more slowly, which is what people actually do - the
    eyes get there first.
    """

    def __init__(self, gaze_k=9.0, head_k=2.6, head_gain=0.55):
        self.gx = self.gy = 0.0
        self.vx = self.vy = 0.0
        self.hy = 0.0
        self.gaze_k = gaze_k
        self.head_k = head_k
        self.head_gain = head_gain

    @staticmethod
    def _smooth(p, v, target, smooth_time, dt):
        """Critically damped follow, unconditionally stable for any dt.

        The obvious `a = k^2*(t-p) - 2k*v` with explicit Euler is only stable
        while dt < 1/k, and worse, clamping p without clamping v winds the
        integrator up: velocity keeps growing while position sits on the rail,
        so the gaze pinned itself at -1.00 and never came back. This closed
        form (the standard SmoothDamp) cannot do that.
        """
        omega = 2.0 / max(smooth_time, 1e-4)
        x = omega * dt
        decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = p - target
        temp = (v + omega * change) * dt
        v = (v - omega * temp) * decay
        p = target + (change + temp) * decay
        if p < -1.0 or p > 1.0:          # hitting the limit kills the velocity
            p = min(1.0, max(-1.0, p))   # so it cannot wind up
            v = 0.0
        return p, v

    def update(self, st, dt):
        tx = st["x"] if st["present"] else 0.0
        ty = st["y"] if st["present"] else 0.0
        dt = float(min(max(dt, 1e-3), 0.25))
        self.gx, self.vx = self._smooth(self.gx, self.vx, tx, 2.0 / self.gaze_k, dt)
        self.gy, self.vy = self._smooth(self.gy, self.vy, ty, 2.0 / self.gaze_k, dt)
        # head lags the eyes - people's eyes arrive first, then the head turns
        self.hy += (tx * self.head_gain - self.hy) * min(1.0, self.head_k * dt)
        return self.gx, self.gy, self.hy
