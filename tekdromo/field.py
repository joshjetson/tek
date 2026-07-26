"""
The FDL surface equation: one scalar height field for the whole head.

    z(x,y) = skull + forehead + brow + nose + cheeks + lips + chin
             - eyes - philtrum - nostrils

The neck is unioned in with max(), which also gives correct jaw-over-neck
occlusion for free - whichever surface is nearer the viewer wins.

The ear is NOT here: it protrudes sideways, and a z(x,y) height field cannot
express that. It has its own lateral field in anatomy.ear_field.
"""
import numpy as np

from .anatomy import (BROW_POLY, CHEEK, CHIN, EYE, FOREHEAD, LIP_LINE,
                      LOWER_LIP, NOSE_CENTRE, NOSE_TIP, NOSE_WIDTH, NOSTRIL,
                      PHILTRUM, UPPER_LIP, _blob, _ridge, sil_w, skull_base)


MOUTH_BOX = (-0.40, 0.40, -0.58, -0.11)


def lip_field(X, Y, openness=0.0, rounding=0.0):
    """Animated lip terms. openness 0..1, rounding -1 (spread) .. +1 (pursed)."""
    o = float(np.clip(openness, 0.0, 1.0))
    r = float(np.clip(rounding, -1.0, 1.0))
    wide = 1.0 - 0.30 * max(r, 0.0) + 0.15 * max(-r, 0.0)
    up = dict(UPPER_LIP); lo = dict(LOWER_LIP); ln = dict(LIP_LINE)
    up["rx2"] = UPPER_LIP["rx2"] * wide * wide
    lo["rx2"] = LOWER_LIP["rx2"] * wide * wide
    ln["rx2"] = LIP_LINE["rx2"] * wide * wide
    # jaw drop: the lower lip travels far further than the upper
    up["cy"] = UPPER_LIP["cy"] + 0.028 * o
    lo["cy"] = LOWER_LIP["cy"] - 0.150 * o
    ln["cy"] = LIP_LINE["cy"] - 0.052 * o
    ln["a"] = LIP_LINE["a"] - 0.30 * o          # aperture deepens as it opens
    ln["ry2"] = LIP_LINE["ry2"] + 0.0090 * o
    return _blob(X, Y, up) + _blob(X, Y, lo) + _blob(X, Y, ln)


# NECK, as part of the SAME height field, so its contours are level sets of the
# same surface as the face and run continuously off the jaw. It was previously
# a ring-and-meridian cylinder - a different renderer inside a contour drawing,
# which is exactly why it read as bolted on.
_NKF_Y = np.array([-0.26, -0.50, -0.74, -0.98, -1.18, -1.34, -1.42])
_NKF_W = np.array([0.312, 0.318, 0.332, 0.360, 0.412, 0.482, 0.540])
_NKF_Z = np.array([-0.150, -0.142, -0.132, -0.118, -0.100, -0.082, -0.070])

def _nk(y, tab):
    return np.interp(y, _NKF_Y[::-1], tab[::-1])


def neck_surface(X, Y):
    """Front of the neck as z(x,y), including the sternocleidomastoid ridges so
    they appear as contour deflections rather than drawn-on lines."""
    w = np.maximum(_nk(Y, _NKF_W), 1e-6)
    t = np.clip(np.abs(X) / w, 0.0, 1.0)
    z = _nk(Y, _NKF_Z) + w * 1.04 * np.sqrt(np.maximum(1.0 - t * t, 0.0))
    for sgn in (-1.0, 1.0):
        tt = np.clip((Y + 0.30) / (-0.86), 0.0, 1.0)
        cx = sgn * (0.40 + (0.05 - 0.40) * tt)
        z = z + 0.050 * np.exp(-((X - cx) ** 2) / 0.0090
                               - ((Y + 0.73) ** 2) / 0.170)
    return z


def neck_mask(X, Y):
    return (np.abs(X) <= _nk(Y, _NKF_W)) & (Y <= -0.26) & (Y >= -1.42)

def zfield(X, Y, lips=True, openness=0.0, rounding=0.0):
    """The FDL surface equation."""
    z = skull_base(X, Y)
    z = z + _blob(X, Y, FOREHEAD)
    z = z + _ridge(X, Y, BROW_POLY, 0.105, 0.085, mirror=True)
    z = z + _ridge(X, Y, list(zip([p[0] for p in NOSE_CENTRE],
                                  [p[1] for p in NOSE_CENTRE])),
                   0.125, NOSE_WIDTH)
    z = z + _blob(X, Y, NOSE_TIP)
    z = z + _blob(X, Y, CHEEK, mirror=True)
    if lips:
        z = z + lip_field(X, Y, openness, rounding)
    z = z + _blob(X, Y, CHIN)
    z = z + _blob(X, Y, EYE, mirror=True)          # a is negative
    z = z + _blob(X, Y, PHILTRUM)
    z = z + _blob(X, Y, NOSTRIL, mirror=True)
    # Union with the neck. At any (x,y) the visible surface is whichever is
    # nearer the viewer, so the jaw occludes the neck for free - no separate
    # hidden-vertex test needed.
    face_ok = (np.abs(X) <= sil_w(Y)) & (Y <= 1.140) & (Y >= -0.855)
    return np.maximum(np.where(face_ok, z, -9.0),
                      np.where(neck_mask(X, Y), neck_surface(X, Y), -9.0))


def head_mask(X, Y):
    """Skull silhouette plus the neck - contours are clipped to this."""
    e = (np.abs(X) <= sil_w(Y)) & (Y <= 1.140) & (Y >= -0.855)
    return e | neck_mask(X, Y)
