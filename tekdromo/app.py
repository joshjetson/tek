"""
The display application.

Its one job is that the screen never goes blank. Five ways it used to stop:

 1. Long black gap at startup - geometry is disk-cached, keyed by a hash of the
    source files so it self-invalidates. Cold ~4s, warm ~0.02s. The framebuffer
    is painted before any of that, so there is a boot screen instead of black.
 2. An exception killed the process - the frame loop catches per-frame errors,
    leaves the last good frame up, and continues.
 3. systemd's start limit - disabled in the unit ([Unit], not [Service]).
 4. The kernel console blanker - FBIOBLANK re-asserted every 30s.
 5. A wedged model - a watchdog thread repaints the last good frame.

Split out of the old tekrun.py, whose main() had grown to 130 lines doing
setup, camera wiring and rendering all at once.
"""
import argparse
import hashlib
import math
import os
import threading
import time
import traceback

import cv2
import numpy as np

from . import contour, framebuffer, geometry, phosphor, rig, speech

CACHE_DIR = os.path.expanduser("~/.cache/tekdromo")
SRC = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
       for f in ("anatomy.py", "field.py", "contour.py", "rig.py")]


# ---------------------------------------------------------------------------
def _src_hash():
    h = hashlib.sha256()
    for p in SRC:
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


def load_geometry():
    """Disk-cached static geometry; rebuilds itself when the source changes."""
    path = os.path.join(CACHE_DIR, _src_hash() + ".npz")
    if os.path.exists(path):
        try:
            d = np.load(path)
            return d["v"], d["e"], d["n"], True
        except Exception:
            pass                                  # corrupt cache: rebuild
    # lips=True: the RESTING face must be complete. Regions only take over an
    # area once their controls actually move.
    v, e, n = contour.build(lips=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp.npz"                       # savez appends .npz
    np.savez_compressed(tmp, v=v, e=e, n=n)
    os.replace(tmp, path)                         # atomic
    return v, e, n, False


# ---------------------------------------------------------------------------
class Display:
    """Owns the panel, the model and the frame loop."""

    def __init__(self, args):
        self.a = args
        self.running = True
        self.last_frame = None
        self.last_frame_t = 0.0
        self.fd, self.mm, self.screen, self.w, self.h = framebuffer.open_screen()
        framebuffer.unblank(self.fd)
        self._unblanked = time.time()
        self.statics = phosphor.build_statics(self.w, self.h)
        self.banner("INITIALISING")

    # -- setup ------------------------------------------------------------
    def banner(self, msg):
        self.screen[:] = phosphor.erase_bgra(self.w, self.h, 0.25, self.statics)
        img = np.zeros((self.h, self.w, 3), np.uint8)
        cv2.putText(img, "TEKTRONIX 4014", (int(self.w * .10), int(self.h * .44)),
                    cv2.FONT_HERSHEY_SIMPLEX, .62, (40, 150, 70), 2, cv2.LINE_AA)
        cv2.putText(img, msg, (int(self.w * .10), int(self.h * .52)),
                    cv2.FONT_HERSHEY_SIMPLEX, .85, (70, 220, 110), 2, cv2.LINE_AA)
        self.screen[..., :3] = img
        self.screen[..., 3] = 255

    def load(self):
        v, e, n, warm = load_geometry()
        print("geometry %s (%d edges)" % ("cached" if warm else "built", len(e)),
              flush=True)
        self.banner("LOADING RIG")
        self.face = rig.Face()
        # unpunched: Face drops edges per frame, and only for regions whose
        # controls have actually moved
        self.face.static = (v, e, n)
        self.face._edge_in = {nm: rig.Face._inside_mask(self.face.static, r.box)
                              for nm, r in self.face.regions.items()}
        self.face.express("neutral", blend=0.01)

        self.cam = self.follow = None
        if not self.a.no_camera and os.path.exists("/dev/video0"):
            try:
                from . import camera
                self.cam = camera.Tracker().start()
                self.follow = camera.Follower()
                print("camera tracking active", flush=True)
            except Exception:
                traceback.print_exc()
        return self

    # -- watchdog ---------------------------------------------------------
    def _watchdog(self, period=2.0):
        while self.running:
            time.sleep(period)
            if self.last_frame is not None and \
                    time.time() - self.last_frame_t > period:
                try:
                    self.screen[:] = self.last_frame
                except Exception:
                    pass

    # -- per-frame --------------------------------------------------------
    def pose(self, t, dt):
        """Head orientation: idle sway, overridden by whoever is being looked at."""
        idle = self.a.yaw * (0.72 * math.sin(2 * math.pi * t / 9.3)
                             + 0.28 * math.sin(2 * math.pi * t / 3.7))
        rx = -0.045 + 0.030 * math.sin(2 * math.pi * t / 6.1)
        if self.cam is None:
            return rx, idle
        st = self.cam.state()
        gx, gy, hy = self.follow.update(st, dt)
        self.face.set(gaze_x=gx, gaze_y=gy)
        if st["present"] != getattr(self, "_seen", False):
            self.face.express("attentive" if st["present"] else "neutral",
                              blend=0.45)
            self._seen = st["present"]
        w = 1.0 if st["present"] else 0.0         # presence weight
        return rx - 0.10 * gy * w, idle * (1.0 - 0.75 * w) + hy * w

    def run(self):
        threading.Thread(target=self._watchdog, daemon=True).start()
        t0 = prev = tick = time.time()
        frames = errors = 0
        while self.running:
            try:
                now = time.time()
                dt = min(0.1, now - prev)
                prev, t = now, now - t0

                if self.a.talk:
                    self.face.speak(*speech.synthetic(t))
                rx, ry = self.pose(t, dt)
                v, e, n = self.face.update(t, dt)
                pts = geometry.build_pts_culled(v, e, n, self.w, self.h,
                                                (rx, ry, 0.0), self.a.dist,
                                                -0.05, "and", self.a.fov)
                frame = phosphor.render_bgra(pts, self.w, self.h, self.statics)
                self.screen[:] = frame
                self.last_frame, self.last_frame_t = frame, time.time()
                frames += 1
                # Cap the rate. Uncapped we render as fast as the CPU allows,
                # which on a 2GB Nano at MAXN with a USB hub, wifi, bluetooth
                # and a camera attached pushed the input rail hard enough to
                # trip soctherm OC ALARM 0x00000001. Nothing above ~30fps is
                # visible on this panel, so the extra draw buys nothing.
                if self.a.max_fps > 0:
                    slack = (1.0 / self.a.max_fps) - (time.time() - now)
                    if slack > 0:
                        time.sleep(slack)
            except Exception:
                # One bad frame is not fatal - but silently swallowing a
                # PERSISTENT failure is worse. A stale import once had this
                # loop crash-looping at 0 fps with 198 errors and nothing
                # useful in the journal, because only the first 3 tracebacks
                # were printed. Re-report periodically so a stuck loop is
                # always visible.
                errors += 1
                if errors <= 3 or errors % 200 == 0:
                    traceback.print_exc()
                    print("frame error #%d - still failing" % errors, flush=True)
                time.sleep(0.05)

            now = time.time()
            if now - tick >= 10.0:
                print("%.1f fps  errors=%d" % (frames / (now - tick), errors),
                      flush=True)
                frames, tick = 0, now
            if now - self._unblanked > 30.0:
                framebuffer.unblank(self.fd)
                self._unblanked = now

    def close(self):
        self.running = False
        if self.cam is not None:
            self.cam.stop()
        try:
            self.screen[:] = 0
        finally:
            del self.screen
            self.mm.close()
            os.close(self.fd)
        print("exited cleanly", flush=True)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="tekdromo")
    ap.add_argument("--yaw", type=float, default=0.30)
    ap.add_argument("--dist", type=float, default=16.0)
    ap.add_argument("--fov", type=float, default=11.4)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--max-fps", type=float, default=30.0,
                    help="frame cap. Rendering flat-out burns power for no "
                         "visible benefit and this board has thin current "
                         "headroom (see the OC ALARM note in app.run).")
    ap.add_argument("--no-talk", dest="talk", action="store_false",
                    help="leave the mouth at rest")
    a = ap.parse_args(argv)

    import signal
    d = Display(a).load()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: setattr(d, "running", False))
    try:
        d.run()
    finally:
        d.close()
