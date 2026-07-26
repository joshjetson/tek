"""
The head's measured shape and the FDL region constants.

Pure data plus the field primitives that evaluate it. No contouring, no
rendering - those live in contour.py and phosphor.py.

Sources for the numbers (see docs/TEKDROMO.md):
  * Loomis: a head is a sphere with the sides cut FLAT. An ellipse cannot be a
    head, so the silhouette is an explicit landmark profile.
  * Widest point is the zygomatic arch, below the eye line - not the cranium.
  * Five eyes wide; mouth two eyes wide; vertical thirds.
  * The jaw turns at the gonial angle: a slope break, not a smooth taper.
  * Ears span brow line to nose base, tilted 18 degrees.
"""
import math

import numpy as np


_SIL_Y = np.array([1.140, 1.100, 1.040, 0.960, 0.860, 0.740, 0.620, 0.480,
                   0.340, 0.200, 0.060, -0.080, -0.200, -0.320, -0.440,
                   -0.540, -0.640, -0.730, -0.790, -0.830, -0.855])
_SIL_W = np.array([0.000, 0.185, 0.315, 0.425, 0.512, 0.578, 0.616, 0.634,
                   0.650, 0.662, 0.670, 0.660, 0.632, 0.592, 0.540,
                   0.468, 0.378, 0.276, 0.166, 0.062, 0.000])
#                   ^flat side plane: .616 -> .670 over most of the face^
#                                     widest at y=.06 (zygomatic arch)
#                                          gonial break at y=-.44 ^

# depth (how far the face projects) against height
_DEP_Y = np.array([1.140, 0.900, 0.620, 0.430, 0.200, 0.000, -0.300,
                   -0.600, -0.800, -0.855])
_DEP_Z = np.array([0.060, 0.470, 0.690, 0.745, 0.720, 0.742, 0.712,
                   0.620, 0.410, 0.180])

# Cross-section exponent. 2.0 = elliptical (round). Higher flattens the front
# plane and turns the sides more sharply - the Loomis front-plane/side-plane
# split. 2.6 reads as structured without going faceted.
CROSS_N = 2.6

def sil_w(y):
    return np.interp(y, _SIL_Y[::-1], _SIL_W[::-1])


def sil_depth(y):
    return np.interp(y, _DEP_Y[::-1], _DEP_Z[::-1])


def skull_base(X, Y):
    """Superelliptic cross-section of width sil_w(y) and depth sil_depth(y)."""
    w = np.maximum(sil_w(Y), 1e-6)
    t = np.clip(np.abs(X) / w, 0.0, 1.0)
    return sil_depth(Y) * np.power(np.maximum(1.0 - t ** CROSS_N, 0.0),
                                   1.0 / CROSS_N)

FOREHEAD = dict(a=0.22, cx=0.0, cy=0.62, rx2=0.18, ry2=0.09)
EYE = dict(a=-0.19, cx=0.315, cy=0.22, rx2=0.018, ry2=0.008)
NOSE_TIP = dict(a=0.17, cx=0.0, cy=-0.02, rx2=0.0121, ry2=0.0100)
NOSTRIL = dict(a=-0.075, cx=0.05, cy=-0.08, rx2=0.0016, ry2=0.0004)
CHEEK = dict(a=0.105, cx=0.34, cy=0.00, rx2=0.0784, ry2=0.1156)
PHILTRUM = dict(a=-0.050, cx=0.0, cy=-0.22, rx2=0.0009, ry2=0.0144)
CHIN = dict(a=0.095, cx=0.0, cy=-0.63, rx2=0.0324, ry2=0.0196)
UPPER_LIP = dict(a=0.070, cx=0.0, cy=-0.275, rx2=0.0626, ry2=0.0016)
LOWER_LIP = dict(a=0.082, cx=0.0, cy=-0.385, rx2=0.0542, ry2=0.0021)
LIP_LINE = dict(a=-0.052, cx=0.0, cy=-0.330, rx2=0.0688, ry2=0.00035)

BROW_POLY = [(-0.34, 0.36), (-0.26, 0.40), (-0.18, 0.42), (-0.08, 0.43), (0.0, 0.43)]
NOSE_CENTRE = [(0.0, 0.43), (0.0, 0.28), (0.0, 0.15), (0.0, 0.04), (-0.01, -0.06)]
NOSE_WIDTH = [0.07, 0.08, 0.09, 0.10, 0.12]
# cx widened from the spec's 0.28: a wider inter-eye gap is wanted here
EYE_OPEN = dict(cx=0.315, cy=0.20, rx=0.13, ry=0.04, rot=-6.0)
JAW = [(-0.44, -0.14), (-0.42, -0.28), (-0.36, -0.48), (-0.22, -0.68)]
# widened to the canonical two eye-widths (the spec's values gave 1.38)
UPPER_LIP_CURVE = [(-0.260, -0.30), (-0.173, -0.27), (-0.072, -0.26), (0.0, -0.25)]
LOWER_LIP_CURVE = [(-0.231, -0.37), (-0.144, -0.39), (0.0, -0.40)]
# NECK, from anatomy references:
#  * it attaches BEHIND the jaw, not centred under the chin - the chin
#    overhangs it. So the cylinder axis is set back in z.
#  * the sternocleidomastoids run from the mastoid (behind the ear) down to
#    the sternal notch, forming a V in front view.
#  * it flares at the base into the trapezius.
_NECK_Y = np.array([-0.34, -0.55, -0.75, -0.94, -1.12, -1.26, -1.34])
_NECK_R = np.array([0.310, 0.315, 0.326, 0.346, 0.386, 0.452, 0.522])
_NECK_Z = np.array([-0.150, -0.142, -0.132, -0.120, -0.104, -0.086, -0.074])
MASTOID = (0.400, -0.300, -0.190)      # behind the ear
STERNAL = (0.052, -1.160, 0.224)       # sternal notch

def _blob(X, Y, s, mirror=False):
    g = s["a"] * np.exp(-((X - s["cx"]) ** 2) / s["rx2"]
                        - ((Y - s["cy"]) ** 2) / s["ry2"])
    if mirror:
        g = g + s["a"] * np.exp(-((X + s["cx"]) ** 2) / s["rx2"]
                                - ((Y - s["cy"]) ** 2) / s["ry2"])
    return g


def _ridge(X, Y, pts, amp, half_w, mirror=False):
    """Sum of gaussians walked along a polyline - the FDL 'ridge' surface.

    Sample centres are gathered first and evaluated as one broadcast, rather
    than one full-grid exponential per sample: the naive form cost ~108 grid
    exps and dominated the 7 s build.
    """
    P = np.array(pts, float)
    cxs, cys, ws = [], [], []
    for i in range(len(P) - 1):
        w = half_w[i] if np.ndim(half_w) else half_w
        for t in np.linspace(0, 1, 7):
            c = P[i] + t * (P[i + 1] - P[i])
            cxs.append(c[0]); cys.append(c[1]); ws.append(w)
            if mirror:
                cxs.append(-c[0]); cys.append(c[1]); ws.append(w)
    cx = np.array(cxs)[:, None]; cy = np.array(cys)[:, None]
    w = np.array(ws)[:, None]
    xf = X.ravel()[None, :]; yf = Y.ravel()[None, :]
    g = amp * np.exp(-((xf - cx) ** 2) / (w * w) - ((yf - cy) ** 2) / (w * w * 1.6))
    return g.max(0).reshape(X.shape)

# EAR canon: top aligns with the BROW line, bottom with the NOSE BASE, and it
# tilts 15-20 deg with the top leaning toward the back of the head. In FDL
# coords that is y = +0.43 down to -0.185.
EAR_TOP, EAR_BOT, EAR_TILT = 0.430, -0.185, 18.0
# helix outline in (along-ear, across-ear) normalised local coords
_EAR_OUT = [(1.00, 0.02), (0.88, 0.46), (0.55, 0.72), (-0.10, 0.62),
            (-0.72, 0.42), (-1.00, 0.05), (-0.92, -0.30), (-0.25, -0.82),
            (0.35, -1.00), (0.82, -0.85)]
_EAR_CONCHA = [(0.42, -0.30), (0.15, -0.02), (-0.22, 0.10), (-0.48, -0.06),
               (-0.40, -0.42), (0.02, -0.58), (0.34, -0.54)]
_EAR_ANTIHELIX = [(0.66, -0.60), (0.38, -0.44), (0.10, -0.34), (-0.20, -0.36),
                  (-0.44, -0.50)]

def ear_field(U, V):
    """Ear relief as protrusion in X, on the side plane, in local (u,v).

    A z(x,y) height field CANNOT express an ear: it protrudes SIDEWAYS, not
    forward. So the ear gets its own field - x as a function of (y,z) - and is
    contoured exactly like the face. Its lines are then level sets of sideways
    protrusion, the direct lateral analogue of the face's contours. That is
    what makes it read as the same object rather than a bolted-on mesh.
    """
    r = np.sqrt(U ** 2 + (V * 1.12) ** 2)
    # body of the ear: a mound inside the outline, fading past the rim
    h = 0.112 * np.clip(1.0 - (r / 1.02) ** 2.4, 0.0, 1.0) ** 0.75
    # helix: raised rim just inside the edge, strongest toward the back
    h = h + 0.050 * np.exp(-((r - 0.86) ** 2) / 0.012) * (0.55 + 0.45 * (0.5 - 0.5 * V))
    # antihelix ridge, inboard of the rim
    h = h + 0.025 * np.exp(-((r - 0.50) ** 2) / 0.020)
    # concha - a genuine bowl, forward and slightly low
    h = h - 0.086 * np.exp(-((U + 0.02) ** 2) / 0.075 - ((V + 0.10) ** 2) / 0.055)
    # tragus, in front of the canal
    h = h + 0.030 * np.exp(-((U - 0.02) ** 2) / 0.010 - ((V - 0.36) ** 2) / 0.007)
    return h
