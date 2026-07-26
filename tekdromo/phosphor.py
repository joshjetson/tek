"""
Storage-tube look: phosphor colour, beam bloom, and the intensity->BGRA LUT.

ONE implementation. This maths previously existed twice - once in tekvector for
still renders and again in tekfb for the live path - which is precisely the kind
of duplication that lets two renderers drift apart.

Authenticity notes that constrain the code:
  * No scanlines. A DVST does not scan; adding them is the classic fake tell.
  * Constant beam intensity - stored charge is binary-ish.
  * Bloom is a CPU pyramid, NOT CUDA. Measured at 1024x600: CUDA 29ms,
    CPU pyramid 6.8ms. Kernel-launch overhead dominates at this size.
"""
import math

import cv2
import numpy as np

# The 4010's storage tube read as a slightly yellowed green, not pure #00FF00.
PHOSPHOR_BGR = np.array([0.30, 1.00, 0.45], dtype=np.float32)
SCREEN_TINT = np.array([0.010, 0.028, 0.014], dtype=np.float32)
MAX_I = 2.0                      # intensity range packed into the 0..255 LUT


def build_statics(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - .5) * 2) ** 2 + ((yy / h - .5) * 2) ** 2)
    # Pre-fold the float->uint8 scale into the vignette so the composite does
    # one multiply instead of a multiply plus a separate scaling pass.
    vig = (np.clip(1.12 - 0.30 * r ** 2, 0, 1) * (255.0 / MAX_I)).astype(np.float32)
    # uint8 copy of the vignette, as a 0..255 gain applied with cv2.multiply's
    # built-in scale. The whole uint8 path exists because the pipeline is
    # memory-bandwidth bound on LPDDR4: measured 29.5ms in float32 vs 17.4ms
    # in uint8 at 1024x600, for the same picture.
    vig_u8 = np.clip(np.clip(1.12 - 0.30 * r ** 2, 0, 1) * 255, 0, 255).astype(np.uint8)
    # Double-height grain: each frame takes a *view* at a random row offset.
    # np.roll was copying 2.4 MB per frame; a view copies nothing.
    grain = (np.random.normal(0, 0.014, (2 * h, w)) * (255.0 / MAX_I)).astype(np.float32)
    grain_u8 = np.clip(np.random.normal(0, 0.014, (2 * h, w)) * 255 + 128,
                       0, 255).astype(np.uint8)
    # 4-channel LUT: intensity -> BGRA, with phosphor colour, white-core
    # saturation and screen tint all baked in. Alpha pinned opaque.
    lut = np.zeros((1, 256, 4), np.uint8)
    for i in range(256):
        t = i / 255.0 * MAX_I
        c = t * PHOSPHOR_BGR + max(t - 1.0, 0.0) * 0.55 + SCREEN_TINT
        lut[0, i, :3] = np.clip(c, 0, 1) * 255
        lut[0, i, 3] = 255
    return vig, grain, lut, vig_u8, grain_u8

def render_bgra(pts, w, h, statics, intensity=1.0):
    """Vectors -> a BGRA frame ready to memcpy into the panel.

    Runs entirely in uint8. The pipeline is memory-bandwidth bound, not compute
    bound, so halving the bytes per pass is the single biggest win available:
    29.5ms -> 17.4ms at 1024x600 for a visually identical picture.

    Things that were measured and did NOT help, so nobody repeats them:
      numpy fancy-index instead of cv2.LUT ... 25.1ms vs 7.1ms  (3.5x WORSE)
      np.take instead of cv2.LUT           ...  8.6ms vs 7.1ms
      box blur instead of gaussian         ...  no change
      LINE_8 instead of LINE_AA            ...  no change
      CUDA composite                       ... 39.1ms, 12.4ms of it in
                                               upload+download alone
    OpenCV's NEON paths are already at the limit here.
    """
    vig, grain, lut, vig_u8, grain_u8 = statics
    # Beam at 127 = intensity 1.0, so bloom can add another 128 before the LUT
    # clips - that headroom is what produces the white saturated core.
    beam = np.zeros((h, w), dtype=np.uint8)
    if len(pts):
        cv2.polylines(beam, pts, False, int(127 * intensity), 1, cv2.LINE_AA)

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
    wide = cv2.addWeighted(cv2.GaussianBlur(quart, (9, 9), 0), 0.40,
                           cv2.GaussianBlur(quart, (21, 21), 0), 0.30, 0)
    glow_s = cv2.addWeighted(cv2.GaussianBlur(small, (5, 5), 0), 0.55,
                             cv2.resize(wide, (hw, hh),
                                        interpolation=cv2.INTER_LINEAR), 1.0, 0)
    glow = cv2.resize(glow_s, (w, h), interpolation=cv2.INTER_LINEAR)

    off = np.random.randint(0, h)
    inten = cv2.addWeighted(beam, 1.15, glow, 1.0, 0)     # saturating uint8 add
    inten = cv2.multiply(inten, vig_u8, scale=1.0 / 255.0)  # vignette as a gain
    # addWeighted does add-and-offset in ONE full-frame pass; add() then
    # subtract(128) was two, and at 614k pixels every pass costs ~1.5ms.
    inten = cv2.addWeighted(inten, 1.0, grain_u8[off:off + h], 1.0, -128.0)
    return cv2.LUT(cv2.cvtColor(inten, cv2.COLOR_GRAY2BGRA), lut)

def erase_bgra(w, h, k, statics):
    a = math.exp(-3.2 * k)
    c = np.clip(PHOSPHOR_BGR * (0.85 * a) + SCREEN_TINT, 0, 1) * 255
    f = np.empty((h, w, 4), np.uint8)
    f[..., 0], f[..., 1], f[..., 2], f[..., 3] = c[0], c[1], c[2], 255
    return f
