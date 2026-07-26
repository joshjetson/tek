#!/usr/bin/env python3
"""
tekfb - Tektronix 4014 storage tube straight to /dev/fb0. No X, no desktop.

Why this is faster than the cv2.imshow version:
  * imshow pushed a 2.7 MB image through the X protocol socket every frame
    (~44 MB/s at 16 fps) just to get it on screen. Here the framebuffer is
    mmap'd and frames are written into it directly - a memcpy.
  * The panel is BGRA with no row padding (stride == xres*4), which is exactly
    OpenCV's BGRA byte order, so there is no pixel conversion at all. The
    phosphor LUT emits 4-channel output directly.
  * Killing X/LXDE also hands back ~100 MB of RAM and its share of the CPU.

Ctrl-C or SIGTERM exits cleanly and blanks the screen.
"""
import argparse
import fcntl
import math
import mmap
import os
import signal
import struct
import time

import cv2
import numpy as np

from tekvector import MODELS, PHOSPHOR_BGR, SCREEN_TINT, build_pts

MAX_I = 2.0
FBIOBLANK = 0x4611
FB_BLANK_UNBLANK = 0
_running = True


def unblank(fd):
    """The kernel console blanker (consoleblank=600 here) switches the display
    off after 10 min with no console keypress. We keep drawing happily into a
    dark panel. Re-assert unblank periodically so the display cannot die under
    us regardless of how the console is configured."""
    try:
        fcntl.ioctl(fd, FBIOBLANK, FB_BLANK_UNBLANK)
    except OSError:
        pass


def _bye(*_):
    global _running
    _running = False


def fb_info(dev="/dev/fb0"):
    with open(dev, "rb") as f:
        u = struct.unpack("<40I", fcntl.ioctl(f, 0x4600, b"\0" * 160)[:160])
    xres, yres, bpp = u[0], u[1], u[6]
    with open("/sys/class/graphics/fb0/stride") as f:
        stride = int(f.read().strip())
    return xres, yres, bpp, stride


def build_statics(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - .5) * 2) ** 2 + ((yy / h - .5) * 2) ** 2)
    # Pre-fold the float->uint8 scale into the vignette so the composite does
    # one multiply instead of a multiply plus a separate scaling pass.
    vig = (np.clip(1.12 - 0.30 * r ** 2, 0, 1) * (255.0 / MAX_I)).astype(np.float32)
    # Double-height grain: each frame takes a *view* at a random row offset.
    # np.roll was copying 2.4 MB per frame; a view copies nothing.
    grain = (np.random.normal(0, 0.014, (2 * h, w)) * (255.0 / MAX_I)).astype(np.float32)
    # 4-channel LUT: intensity -> BGRA, with phosphor colour, white-core
    # saturation and screen tint all baked in. Alpha pinned opaque.
    lut = np.zeros((1, 256, 4), np.uint8)
    for i in range(256):
        t = i / 255.0 * MAX_I
        c = t * PHOSPHOR_BGR + max(t - 1.0, 0.0) * 0.55 + SCREEN_TINT
        lut[0, i, :3] = np.clip(c, 0, 1) * 255
        lut[0, i, 3] = 255
    return vig, grain, lut


def render_bgra(pts, w, h, statics, intensity=1.0):
    vig, grain, lut = statics
    beam = np.zeros((h, w), dtype=np.float32)
    if len(pts):
        cv2.polylines(beam, pts, False, intensity, 1, cv2.LINE_AA)

    # Bloom as a CPU pyramid. Measured against CUDA at this resolution:
    #   3x CUDA blur (half-res) ... 29.3 ms
    #   pure CPU     (half-res) ... 14.2 ms
    #   CPU pyramid             ...  6.8 ms   <- this
    # The GPU wins on big convolutions at 1080p, but at 512x300 the kernel
    # launch overhead swamps it and NEON on the A57s is simply faster. Doing
    # the wide blurs at quarter res costs nothing visually - they are blurs.
    hw, hh = w // 2, h // 2
    small = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
    quart = cv2.resize(small, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA)
    wide = cv2.GaussianBlur(quart, (9, 9), 0) * 0.40 + \
           cv2.GaussianBlur(quart, (21, 21), 0) * 0.30
    glow_s = cv2.GaussianBlur(small, (5, 5), 0) * 0.55 + \
             cv2.resize(wide, (hw, hh), interpolation=cv2.INTER_LINEAR)
    glow = cv2.resize(glow_s, (w, h), interpolation=cv2.INTER_LINEAR)

    off = np.random.randint(0, h)
    inten = cv2.add(cv2.multiply(beam, 1.15), glow)
    inten = cv2.add(cv2.multiply(inten, vig), grain[off:off + h])
    idx = cv2.convertScaleAbs(inten)          # scale already folded into vig
    return cv2.LUT(cv2.cvtColor(idx, cv2.COLOR_GRAY2BGRA), lut)


def erase_bgra(w, h, k, statics):
    a = math.exp(-3.2 * k)
    c = np.clip(PHOSPHOR_BGR * (0.85 * a) + SCREEN_TINT, 0, 1) * 255
    f = np.empty((h, w, 4), np.uint8)
    f[..., 0], f[..., 1], f[..., 2], f[..., 3] = c[0], c[1], c[2], 255
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="apple,torus,sphere,cube")
    ap.add_argument("--spin-frames", type=int, default=200)
    ap.add_argument("--plot-frames", type=int, default=90)
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    w, h, bpp, stride = fb_info()
    assert bpp == 32 and stride == w * 4, f"unexpected fb: {bpp}bpp stride={stride}"
    print(f"framebuffer {w}x{h} {bpp}bpp stride={stride}", flush=True)

    fd = os.open("/dev/fb0", os.O_RDWR)
    mm = mmap.mmap(fd, h * stride, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    screen = np.frombuffer(mm, dtype=np.uint8).reshape(h, w, 4)

    unblank(fd)
    last_unblank = time.time()

    statics = build_statics(w, h)
    names = [m.strip() for m in a.models.split(",") if m.strip() in MODELS]
    dist = {"apple": 3.4, "torus": 3.0, "sphere": 3.2, "cube": 3.6}

    try:
        while _running:
            for name in names:
                if not _running:
                    break
                verts, edges = MODELS[name]()
                d = dist.get(name, 3.4)

                if time.time() - last_unblank > 60:
                    unblank(fd)
                    last_unblank = time.time()

                for k in range(7):
                    screen[:] = erase_bgra(w, h, k / 6.0, statics)
                    time.sleep(0.045)

                n_plot = min(len(edges), a.plot_frames)
                for i in range(n_plot):
                    if not _running:
                        break
                    n = int(len(edges) * (i + 1) / n_plot)
                    pts = build_pts(verts, edges[:n], w, h, (-0.20, 0.55, 0), d)
                    screen[:] = render_bgra(pts, w, h, statics)

                t0, fr = time.time(), 0
                for i in range(a.spin_frames):
                    if not _running:
                        break
                    ry = 0.55 + 2 * math.pi * i / a.spin_frames
                    pts = build_pts(verts, edges, w, h, (-0.20, ry, 0), d)
                    screen[:] = render_bgra(pts, w, h, statics)
                    fr += 1
                if fr:
                    print(f"{name}: {fr / (time.time() - t0):.1f} fps", flush=True)
    finally:
        screen[:] = 0
        del screen
        mm.close()
        os.close(fd)
        print("exited cleanly", flush=True)


if __name__ == "__main__":
    main()
