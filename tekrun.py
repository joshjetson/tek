#!/usr/bin/env python3
"""
tekrun - the display runner. Its one job is that the screen NEVER goes blank.

Five distinct ways the picture used to stop, and what handles each:

 1. Long black gap at startup.  The field build takes ~5 s. Geometry is now
    cached to disk (keyed by a hash of the source files, so it rebuilds by
    itself when the code changes) - a warm start is well under a second. The
    framebuffer is also opened and painted BEFORE any of that, so there is a
    boot screen instead of black.

 2. An exception in the frame loop killed the process. The loop now catches
    per-frame errors, leaves the last good frame on screen, logs once, and
    keeps going. One bad frame is not a reason to stop.

 3. systemd's start limit. Restart=always still gave up permanently after 5
    failures in 300 s. StartLimitIntervalSec=0 removes the limit - it will now
    retry forever.

 4. The kernel console blanker switching the panel off (consoleblank). An
    FBIOBLANK unblank is re-asserted every 30 s.

 5. A stall in the model code hanging the loop. A watchdog thread repaints the
    last good frame if the main loop misses its deadline, so the picture stays
    live even if something upstream is wedged.
"""
import argparse
import hashlib
import math
import mmap
import os
import signal
import sys
import threading
import time
import traceback

import cv2
import numpy as np

from tekfb import build_statics, erase_bgra, fb_info, render_bgra, unblank

CACHE_DIR = "/home/super/.cache/tekface"
_running = True
_last_frame = None
_last_frame_t = 0.0


def _bye(*_):
    global _running
    _running = False


def _src_hash(paths):
    h = hashlib.sha256()
    for p in paths:
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


def load_geometry(rig=True):
    """Disk-cached static geometry. Rebuilds automatically when source changes."""
    srcs = ["/home/super/tekfdl.py", "/home/super/tekrig.py"]
    key = _src_hash(srcs) + ("-rig" if rig else "-plain") + "-lips"
    path = os.path.join(CACHE_DIR, key + ".npz")
    if os.path.exists(path):
        try:
            d = np.load(path)
            return d["v"], d["e"], d["n"], True
        except Exception:
            pass                                   # corrupt cache: just rebuild
    import tekfdl
    # lips=True: the resting face must be COMPLETE. Regions only take
    # over an area once their controls actually move.
    v, e, n = tekfdl.build(lips=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # savez appends .npz unless the name already ends in it, so name the temp
    # file explicitly or the rename below has nothing to rename.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, v=v, e=e, n=n)
    os.replace(tmp, path)                          # atomic: no half-written cache
    return v, e, n, False


def boot_screen(screen, w, h, statics, msg):
    """Something on the panel immediately, so startup is never black."""
    screen[:] = erase_bgra(w, h, 0.25, statics)
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, msg, (int(w * 0.10), int(h * 0.52)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (70, 220, 110), 2, cv2.LINE_AA)
    cv2.putText(img, "TEKTRONIX 4014", (int(w * 0.10), int(h * 0.44)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 150, 70), 2, cv2.LINE_AA)
    screen[..., :3] = img
    screen[..., 3] = 255


def watchdog(screen, period=2.0):
    """If the main loop misses its deadline, keep repainting the last good
    frame. A wedged model must not become a blank screen."""
    while _running:
        time.sleep(period)
        if _last_frame is None:
            continue
        if time.time() - _last_frame_t > period:
            try:
                screen[:] = _last_frame
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw", type=float, default=0.30)
    ap.add_argument("--dist", type=float, default=16.0)
    ap.add_argument("--fov", type=float, default=11.4)
    ap.add_argument("--no-rig", action="store_true",
                    help="static face, no expression rig")
    ap.add_argument("--no-camera", action="store_true",
                    help="disable face tracking")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    w, h, bpp, stride = fb_info()
    fd = os.open("/dev/fb0", os.O_RDWR)
    mm = mmap.mmap(fd, h * stride, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    screen = np.frombuffer(mm, dtype=np.uint8).reshape(h, w, 4)
    unblank(fd)
    last_unblank = time.time()

    statics = build_statics(w, h)
    boot_screen(screen, w, h, statics, "INITIALISING")
    print("framebuffer %dx%d" % (w, h), flush=True)

    import tekhead
    v, e, n, warm = load_geometry(rig=not a.no_rig)
    print("geometry %s (%d edges)" % ("from cache" if warm else "built",
                                      len(e)), flush=True)

    face = None
    if not a.no_rig:
        boot_screen(screen, w, h, statics, "LOADING RIG")
        import tekrig
        face = tekrig.Face()
        # Hand it the cached geometry UNPUNCHED - Face drops edges per frame,
        # and only for regions whose controls have actually moved.
        face.static = (v, e, n)
        face._edge_in = {n_: tekrig.Face._inside_mask(face.static, r.box)
                         for n_, r in face.regions.items()}
        face.express("neutral", blend=0.01)

    # --- camera face tracking -------------------------------------------
    cam = fol = None
    if not a.no_camera and os.path.exists("/dev/video0"):
        try:
            import tekcam
            cam = tekcam.Tracker().start()
            fol = tekcam.Follower()
            print("camera tracking active", flush=True)
        except Exception:
            traceback.print_exc()
            print("camera unavailable - idling", flush=True)
            cam = fol = None
    else:
        print("no camera", flush=True)

    global _last_frame, _last_frame_t
    threading.Thread(target=watchdog, args=(screen,), daemon=True).start()

    t0 = time.time()
    frames, tick, errors = 0, time.time(), 0
    prev = time.time()
    present_prev = False
    while _running:
        try:
            now0 = time.time()
            dt = min(0.1, now0 - prev)
            prev = now0
            t = now0 - t0

            # idle sway: two periods that never line up, so it does not read
            # as a machine sweeping
            idle = a.yaw * (0.72 * math.sin(2 * math.pi * t / 9.3)
                            + 0.28 * math.sin(2 * math.pi * t / 3.7))
            ry = idle
            rx = -0.045 + 0.030 * math.sin(2 * math.pi * t / 6.1)

            if cam is not None:
                st = cam.state()
                gx, gy, hy = fol.update(st, dt)
                if face is not None:
                    face.set(gaze_x=gx, gaze_y=gy)
                    if st["present"] != present_prev:
                        face.express("attentive" if st["present"] else "neutral",
                                     blend=0.45)
                        present_prev = st["present"]
                # when someone is there the head turns toward them and the idle
                # sway is suppressed; when they leave it drifts back to idling
                pw = 1.0 if st["present"] else 0.0     # NOT `w` - that is the
                ry = idle * (1.0 - 0.75 * pw) + hy * pw  # framebuffer width
                rx += -0.10 * gy * pw

            if face is not None:
                vv, ee, nn = face.update(t, dt)
            else:
                vv, ee, nn = v, e, n
            pts = tekhead.build_pts_culled(vv, ee, nn, w, h, (rx, ry, 0.0),
                                           a.dist, -0.05, "and", a.fov)
            frame = render_bgra(pts, w, h, statics)
            screen[:] = frame
            _last_frame, _last_frame_t = frame, time.time()
            frames += 1
        except Exception:
            # One bad frame must not take the display down.
            errors += 1
            if errors <= 3:
                traceback.print_exc()
                print("frame error #%d - continuing" % errors, flush=True)
            time.sleep(0.05)

        now = time.time()
        if now - tick >= 10.0:
            print("%.1f fps  errors=%d" % (frames / (now - tick), errors),
                  flush=True)
            frames, tick = 0, now
        if now - last_unblank > 30.0:
            unblank(fd)
            last_unblank = now

    if cam is not None:
        cam.stop()

    screen[:] = 0
    del screen
    mm.close()
    os.close(fd)
    print("exited cleanly", flush=True)


if __name__ == "__main__":
    main()
