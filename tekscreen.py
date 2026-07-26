#!/usr/bin/env python3
"""
tekscreen - live Tektronix 4014 storage tube on the Nano's actual display.

Cycles: erase flash -> beam plots the model stroke by stroke -> model rotates.

Speed notes vs the still renderer (which ran ~0.76 s/frame):
  * bloom is computed at half resolution and upscaled. Blur cost scales with
    pixel count, so this is ~4x cheaper and visually near-identical.
  * the vignette was being rebuilt with np.mgrid every single frame. Cached.
  * grain is a small pre-generated tile, rolled each frame, instead of a fresh
    1280x720x3 normal draw.

Press q or ESC to quit.
"""
import argparse
import math
import time

import cv2
import numpy as np

from tekvector import MODELS, PHOSPHOR_BGR, SCREEN_TINT, build_segments, _blur

_cache = {}
MAX_I = 2.0          # intensity range packed into the 0..255 LUT index


def _statics(w, h):
    """Vignette, grain tile and phosphor LUT, all built once.

    Profiling showed the old per-frame composite cost 140 ms - it was doing ~8
    full-frame 3-channel float passes, which is memory-bandwidth bound on this
    board. Two changes fix it:
      * vignette and grain are single-channel, so fold them into the intensity
        map *before* expanding to colour (1/3 the memory traffic).
      * phosphor colour, white-core saturation and screen tint all depend only
        on intensity, so they collapse into a 256-entry lookup table.
    """
    if (w, h) in _cache:
        return _cache[(w, h)]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - .5) * 2) ** 2 + ((yy / h - .5) * 2) ** 2)
    vig = np.clip(1.12 - 0.30 * r ** 2, 0, 1).astype(np.float32)
    grain = np.random.normal(0, 0.014, (h, w)).astype(np.float32)

    lut = np.zeros((1, 256, 3), np.uint8)
    for i in range(256):
        t = i / 255.0 * MAX_I
        c = t * PHOSPHOR_BGR + max(t - 1.0, 0.0) * 0.55 + SCREEN_TINT
        lut[0, i] = np.clip(c, 0, 1) * 255

    _cache[(w, h)] = (vig, grain, lut)
    return _cache[(w, h)]


def render(segments, w, h, intensities=None):
    beam = np.zeros((h, w), dtype=np.float32)
    uniform = intensities is None or float(np.ptp(intensities)) < 1e-6
    if uniform and segments:
        # One batched call instead of a Python loop: 2.9 ms vs 50 ms.
        pts = np.rint(np.asarray(segments, dtype=np.float32)).astype(np.int32)
        cv2.polylines(beam, pts, False,
                      float(intensities[0]) if intensities is not None else 1.0,
                      1, cv2.LINE_AA)
    else:
        for i, (p0, p1) in enumerate(segments):
            cv2.line(beam, (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(p1[0])), int(round(p1[1]))),
                     float(intensities[i]), 1, cv2.LINE_AA)

    # Bloom at half res, then back up.
    hw, hh = w // 2, h // 2
    small = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
    glow_s = _blur(small, 5) * 0.55 + _blur(small, 15) * 0.40 + _blur(small, 31) * 0.30
    glow = cv2.resize(glow_s, (w, h), interpolation=cv2.INTER_LINEAR)

    vig, grain, lut = _statics(w, h)
    inten = cv2.add(cv2.multiply(beam, 1.15), glow)
    inten = cv2.add(cv2.multiply(inten, vig), np.roll(grain, np.random.randint(0, h), 0))
    idx = cv2.convertScaleAbs(inten, alpha=255.0 / MAX_I)   # float -> uint8, clamped
    return cv2.LUT(cv2.cvtColor(idx, cv2.COLOR_GRAY2BGR), lut)


def erase_flash(w, h, k):
    """A real 4014 could not erase one line - it flooded the whole screen with
    a bright green flash and started over. Best quirk of the format."""
    a = math.exp(-3.2 * k)
    f = np.zeros((h, w, 3), np.float32)
    f += (PHOSPHOR_BGR * (0.85 * a))[None, None, :]
    f += SCREEN_TINT[None, None, :]
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-W", "--width", type=int, default=1280)
    ap.add_argument("-H", "--height", type=int, default=720)
    ap.add_argument("--models", default="apple,torus,sphere,cube")
    ap.add_argument("--spin-frames", type=int, default=150)
    ap.add_argument("--windowed", action="store_true")
    a = ap.parse_args()

    win = "tek"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    if not a.windowed:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    names = [m.strip() for m in a.models.split(",") if m.strip() in MODELS]
    dist = {"apple": 3.4, "torus": 3.0, "sphere": 3.2, "cube": 3.6}

    def show(img):
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        return k not in (ord('q'), 27)

    try:
        while True:
            for name in names:
                verts, edges = MODELS[name]()
                d = dist.get(name, 3.4)

                for k in range(7):                       # erase
                    if not show(erase_flash(a.width, a.height, k / 6.0)):
                        return

                n_plot = min(len(edges), 90)             # plot it in
                for i in range(n_plot):
                    n = int(len(edges) * (i + 1) / n_plot)
                    segs, ints = build_segments(verts, edges[:n], a.width,
                                                a.height, (-0.20, 0.55, 0), d)
                    if not show(render(segs, a.width, a.height, ints)):
                        return

                t0, fr = time.time(), 0                  # spin
                for i in range(a.spin_frames):
                    ry = 0.55 + 2 * math.pi * i / a.spin_frames
                    segs, ints = build_segments(verts, edges, a.width,
                                                a.height, (-0.20, ry, 0), d)
                    if not show(render(segs, a.width, a.height, ints)):
                        return
                    fr += 1
                print(f"{name}: {fr / (time.time() - t0):.1f} fps", flush=True)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
