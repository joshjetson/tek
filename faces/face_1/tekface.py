#!/usr/bin/env python3
"""
tekface - the assistant's face. Bald head on the storage-tube display, idling.

The motion is deliberately understated: two sine waves of different, mutually
irrational-ish periods for the yaw, plus a slower shallow nod. A single sine
reads as a machine sweeping; two out-of-phase ones read as somebody idly
present. Nothing here needs facial expression to work.

Runs straight on /dev/fb0 - no X.
"""
import argparse
import math
import os
import time

import numpy as np

import tekhead
import tekanat
from tekfb import (build_statics, erase_bgra, fb_info, render_bgra, unblank,
                   _bye)
import signal
import mmap
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw", type=float, default=0.30,
                    help="peak left/right turn in radians (0.30 ~ 17 deg)")
    ap.add_argument("--dist", type=float, default=3.15)
    ap.add_argument("--model", default="anat",
                    choices=("anat", "mannequin", "face"),
                    help="mannequin = dense featureless form (free3d reference); "
                         "face = the coarser head with eyes/nose/mouth")
    ap.add_argument("--fov", type=float, default=None)
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the initial beam-plot reveal")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    w, h, bpp, stride = fb_info()
    print(f"framebuffer {w}x{h} {bpp}bpp", flush=True)
    fd = os.open("/dev/fb0", os.O_RDWR)
    mm = mmap.mmap(fd, h * stride, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    screen = np.frombuffer(mm, dtype=np.uint8).reshape(h, w, 4)
    unblank(fd)
    last_unblank = time.time()

    statics = build_statics(w, h)
    if a.model == "anat":
        verts, edges, normals = tekanat.anatomical_head()
        dist = a.dist if a.dist != 3.15 else 12.0
        fov = a.fov if a.fov else 7.4 * min(w, h) / 820.0
    elif a.model == "mannequin":
        verts, edges, normals = tekhead.mannequin_model()
        # long lens: far camera + wide fov fills the frame while staying
        # near-orthographic, which is how the reference render looks.
        dist = a.dist if a.dist != 3.15 else 12.0
        fov = a.fov if a.fov else 8.6 * min(w, h) / 900.0
    else:
        verts, edges, normals = tekhead.head_model()
        dist, fov = a.dist, (a.fov or 1.35)
    print(f"{a.model}: {len(verts)} verts, {len(edges)} edges", flush=True)

    talking = (a.model == "anat")

    def pose(rx, ry, t=None):
        v, e_, n_ = verts, edges, normals
        if talking and t is not None:
            # Mouth is rebuilt every frame (~6 ms) and appended to the static
            # head, rather than rebuilding the whole head (~32 ms).
            op, rnd = tekanat.speech_params(t)
            mv, mn, me = tekanat.mouth_geometry(op, rnd, 1.15, len(verts))
            v = np.concatenate([verts, mv])
            n_ = np.concatenate([normals, mn])
            e_ = np.concatenate([edges, me])
        return tekhead.build_pts_culled(v, e_, n_, w, h,
                                        (rx, ry, 0.0), dist, -0.02, "and", fov)

    try:
        # Arrive the way a 4014 would: erase, then let the beam lay the face in.
        for k in range(7):
            screen[:] = erase_bgra(w, h, k / 6.0, statics)
            time.sleep(0.04)
        if not a.no_plot:
            for i in range(60):
                n = int(len(edges) * (i + 1) / 60)
                screen[:] = render_bgra(pose(-0.05, 0.0, 0.0)[:n], w, h, statics)

        t0, frames, tick = time.time(), 0, time.time()
        while True:
            import tekfb
            if not tekfb._running:
                break
            t = time.time() - t0

            # Two periods that do not line up, so the loop never feels like one.
            ry = a.yaw * (0.72 * math.sin(2 * math.pi * t / 9.3) +
                          0.28 * math.sin(2 * math.pi * t / 3.7))
            rx = -0.045 + 0.030 * math.sin(2 * math.pi * t / 6.1)

            screen[:] = render_bgra(pose(rx, ry, t), w, h, statics)
            frames += 1

            if time.time() - tick >= 10.0:
                print(f"{frames / (time.time() - tick):.1f} fps  yaw={ry:+.2f}",
                      flush=True)
                frames, tick = 0, time.time()
            if time.time() - last_unblank > 60:
                unblank(fd)
                last_unblank = time.time()
    finally:
        screen[:] = 0
        del screen
        mm.close()
        os.close(fd)
        print("exited cleanly", flush=True)


if __name__ == "__main__":
    main()
