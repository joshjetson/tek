#!/usr/bin/env python3
"""
tekfdl - Face Description Language v0.1 -> Tektronix vectors.

A different architecture from face_1. Instead of a lat/long mesh with feature
curves drawn on top, the face is an IMPLICIT HEIGHT FIELD:

    z(x,y) = skull + forehead + brow + nose + cheeks + lips + chin
             - eyes - philtrum - nostrils

built from the FDL regions, then sliced into iso-contours:

    for z = .95 down to -.25 step -.05:
        intersect surface, extract closed contours, simplify, emit vectors

That is why this approach is worth trying: contour lines *automatically* flow
around every feature, because they are level sets of the actual surface. All
the overlay tricks in face_1 were trying to fake exactly this and could not -
a nose ridge drawn on top always read as a decal.

FDL convention: normalised x,y in [-1,1], origin between the eyes at (0,0).
Gaussian denominators are radius**2, which matches the worked equations in the
spec (skull rx=.74 -> x^2/.65, eye rx=.13 -> (x-.28)^2/.018).
"""
import math

import numpy as np

# --------------------------------------------------------------------------
# REGIONS  (amplitude, centre, radii)  - straight from the FDL spec
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ANATOMICAL SKULL  (replaces the ellipse - an ellipse cannot be a head)
#
# From the reference material:
#  * Loomis: the head is a sphere with the SIDES CUT FLAT, because the temporal
#    bone runs vertically down the side of the skull. Those flat side planes are
#    what a smooth oval can never have.
#  * The widest point of the head is the ZYGOMATIC ARCH (cheekbones), not the
#    cranium - so the silhouette must bulge low, not high.
#  * "Five eyes wide" zygomatic-to-zygomatic: eye width == inter-eye gap ==
#    outer-corner-to-side. With the FDL eye (rx .13, centre .28) that puts the
#    side of the head at |x| ~= .67 at eye level.
#  * The jaw turns at the GONIAL ANGLE - a distinct slope break, not a smooth
#    taper. Above it the mandible barely narrows; below it drops away fast.
#
# half-width of the head silhouette against height
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


def _superblob(X, Y, s, n):
    """Like _blob but with a superelliptic falloff, so the level sets square
    off instead of staying perfectly oval."""
    ax = np.abs(X - s["cx"]) / math.sqrt(s["rx2"])
    ay = np.abs(Y - s["cy"]) / math.sqrt(s["ry2"])
    return s["a"] * np.exp(-(ax ** n + ay ** n))


# The mouth is re-contoured every frame inside this box while the rest of the
# face stays static. Because the field is identical either side of the border,
# contour lines meet the boundary at the same points and read as continuous.
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


# --------------------------------------------------------------------------
# Contour generator - literally the loop from the spec
# --------------------------------------------------------------------------
def _march(mask, gx, gy, min_raw=12, min_pts=6, eps=1.15):
    """Closed iso-contours of a boolean field, in world coords.

    The thresholds must be relaxed for small region boxes. Inside a box the
    iso-lines are short and nearly straight, so approxPolyDP reduces them to
    2-4 points and the default min_pts=6 discards them - which punched a black
    band across the brow and eyes once the rig started supplying those areas.
    Whole-head contours are big closed loops and always survive, so the bug
    only appeared at region scale.
    """
    import cv2
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < min_raw:
            continue
        c = cv2.approxPolyDP(c, eps, True)        # simplify
        if len(c) < min_pts:
            continue
        p = c[:, 0, :].astype(float)
        out.append(np.stack([gx[p[:, 0].astype(int).clip(0, len(gx) - 1)],
                             gy[p[:, 1].astype(int).clip(0, len(gy) - 1)]], 1))
    return out


# The one grid definition. Regions MUST sample the same points or their
# contours land at slightly different coordinates and a rectangular seam shows
# up where they meet the static face.
GRID_RES = 360
GRID_X = (-1.02, 1.02)
GRID_Y = (1.18, -1.48)


def static_grid(res=GRID_RES):
    return (np.linspace(GRID_X[0], GRID_X[1], res),
            np.linspace(GRID_Y[0], GRID_Y[1], res))


def build(res=GRID_RES, lips=False, z_top=0.95, z_bot=-0.25, z_step=0.05,
          depth=1.05, back=True):
    """Returns (verts, edges, normals)."""
    gx, gy = static_grid(res)
    X, Y = np.meshgrid(gx, gy)
    Z = zfield(X, Y, lips=lips)
    M = head_mask(X, Y)
    Zm = np.where(M, Z, -9.0)

    # surface normal from the gradient of the height field, for culling
    gzy, gzx = np.gradient(Z, gy[1] - gy[0], gx[1] - gx[0])

    verts, edges, norms = [], [], []

    def emit(poly, zlev, nz_from_grad=True, closed=True):
        base = len(verts)
        for (px, py) in poly:
            ix = int(np.clip((px - gx[0]) / (gx[1] - gx[0]), 0, res - 1))
            iy = int(np.clip((py - gy[0]) / (gy[1] - gy[0]), 0, res - 1))
            n = np.array([-gzx[iy, ix], -gzy[iy, ix], 1.0]) if nz_from_grad \
                else np.array([0.0, 0.0, 1.0])
            n /= max(np.linalg.norm(n), 1e-9)
            verts.append((px, py, zlev * depth))
            norms.append(tuple(n))
        k = np.arange(len(poly) - 1) + base
        edges.extend(np.stack([k, k + 1], 1).tolist())
        if closed:
            edges.append([base + len(poly) - 1, base])

    levels = np.arange(z_top, z_bot - 1e-9, -z_step)
    bx0, bx1, by0, by1 = MOUTH_BOX
    for lev in levels:
        for poly in _march(Zm >= lev, gx, gy):
            # Only carve out the mouth box when the lips are NOT in the field
            # (i.e. something else will supply them). With lips=True the face
            # is complete and punching a hole here would leave it mouthless.
            inside = ((poly[:, 0] > bx0) & (poly[:, 0] < bx1)
                      & (poly[:, 1] > by0) & (poly[:, 1] < by1)) if not lips \
                else np.zeros(len(poly), bool)
            if not inside.any():
                emit(poly, float(lev))
                continue
            run = []
            for pt, ins in zip(np.vstack([poly, poly[:1]]),
                               np.concatenate([inside, inside[:1]])):
                if ins:
                    if len(run) > 3:
                        emit(np.array(run), float(lev), closed=False)
                    run = []
                else:
                    run.append(pt)
            if len(run) > 3:
                emit(np.array(run), float(lev), closed=False)

    # ---- explicit feature curves, in the spec's render order ---------------
    def curve(pts, zfn, close=False):
        base = len(verts)
        for (px, py) in pts:
            verts.append((px, py, zfn(px, py) * depth))
            norms.append((0.0, 0.0, 1.0))
        k = np.arange(len(pts) - 1) + base
        edges.extend(np.stack([k, k + 1], 1).tolist())
        if close:
            edges.append([base + len(pts) - 1, base])

    def zat(px, py):
        return float(zfield(np.array([[px]]), np.array([[py]]))[0, 0]) + 0.012

    # eye openings - ellipse with the specified -6 degree rotation
    for s in (-1, 1):
        th = math.radians(EYE_OPEN["rot"] * s)
        pts = []
        for k in range(48):
            a = 2 * math.pi * k / 48
            ex, ey = EYE_OPEN["rx"] * math.cos(a), EYE_OPEN["ry"] * math.sin(a)
            pts.append((s * EYE_OPEN["cx"] + ex * math.cos(th) - ey * math.sin(th),
                        EYE_OPEN["cy"] + ex * math.sin(th) + ey * math.cos(th)))
        curve(pts, zat, close=True)

    # lips: mirror the half-curves given in the spec
    up = UPPER_LIP_CURVE + [(-p[0], p[1]) for p in reversed(UPPER_LIP_CURVE[:-1])]
    lo = LOWER_LIP_CURVE + [(-p[0], p[1]) for p in reversed(LOWER_LIP_CURVE[:-1])]
    curve(_resample(up, 26), zat)
    curve(_resample(lo, 22), zat)
    curve(_resample(up, 26)[:1] + _resample(lo, 22)[:1], zat)   # corner tie
    for s in (-1, 1):
        curve(_resample([(s * p[0], p[1]) for p in JAW], 20), zat)

    if back:
        _add_back(verts, edges, norms, depth)
    # NOTE: no _add_neck here. The neck is part of the height field now, so it
    # is contoured with the face. The old ring-and-meridian version is kept
    # below only for reference and is no longer called.
    _add_ears(verts, edges, norms, depth)

    V = np.array(verts, np.float32)
    # Portrait framing: the model now runs from the crown (y=1.14) to the
    # trapezius (y=-1.62). Lifting it puts the head in the upper two-thirds
    # with the neck running to the bottom edge, rather than centring on the
    # whole head+neck and shrinking the face.
    V[:, 1] += 0.20
    return V, np.array(edges, np.int32), np.array(norms, np.float32)


def _resample(pts, n):
    P = np.array(pts, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, d[-1], n)
    return list(zip(np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])))


def _add_back(verts, edges, norms, depth):
    """Back of the skull as a coarse ellipsoid, so the head has volume when it
    turns. The FDL height field only describes the front relief."""
    rings, n_lon = 9, 22
    grid = {}
    for i in range(rings):
        v = math.pi * (i + 0.5) / rings
        for j in range(n_lon):
            u = math.pi * j / (n_lon - 1)          # back half only
            grid[(i, j)] = len(verts)
            yy = 0.18 + 0.96 * math.cos(v)
            x = float(sil_w(np.array(yy))) * math.sin(v) * math.cos(u) / max(
                math.sin(v), 1e-3) if False else \
                float(sil_w(np.array(yy))) * math.cos(u)
            z = -0.78 * math.sin(v) * math.sin(u) * depth
            verts.append((x, yy, z))
            nn = np.array([x / 0.74, (yy - 0.18) / 0.96, z / 0.78])
            nn /= max(np.linalg.norm(nn), 1e-9)
            norms.append(tuple(nn))
    for i in range(rings):
        for j in range(n_lon):
            if j < n_lon - 1:
                edges.append([grid[(i, j)], grid[(i, j + 1)]])
            if i < rings - 1:
                edges.append([grid[(i, j)], grid[(i + 1, j)]])


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


def _add_ears(verts, edges, norms, depth, res=130, step=0.022):
    """Contour the ear field and lay the loops onto the side of the head."""
    yc = 0.5 * (EAR_TOP + EAR_BOT)
    L = 0.5 * (EAR_TOP - EAR_BOT)
    Wd = 0.190
    zc = -0.10
    t = math.radians(EAR_TILT)

    gu = np.linspace(-1.28, 1.28, res)
    gv = np.linspace(1.28, -1.28, res)
    U, V = np.meshgrid(gu, gv)
    Hf = ear_field(U, V)
    Hm = np.where(np.sqrt(U ** 2 + (V * 1.12) ** 2) <= 1.14, Hf, -9.0)

    for side in (-1, 1):
        for lev in np.arange(0.098, -0.070, -step):
            for poly in _march(Hm >= float(lev), gu, gv):
                if len(poly) < 5:
                    continue
                base = len(verts)
                uu, vv = poly[:, 0], poly[:, 1]
                yy = yc + uu * L * math.cos(t) + vv * Wd * math.sin(t)
                zz = zc - uu * L * math.sin(t) + vv * Wd * math.cos(t)
                xw = sil_w(yy)
                for k in range(len(poly)):
                    verts.append((side * (xw[k] - 0.030 + float(lev)),
                                  float(yy[k]), float(zz[k]) * depth))
                    norms.append((side * 0.74, 0.0, 0.67))
                kk = np.arange(len(poly) - 1) + base
                edges.extend(np.stack([kk, kk + 1], 1).tolist())
                edges.append([base + len(poly) - 1, base])


def _add_neck(verts, edges, norms, depth):
    """Tapered cylinder set BACK from the chin, plus the SCM V.

    Vertices that fall inside the head silhouette AND behind the face surface
    are dropped: there is no depth buffer, so without that test the front of
    the neck draws straight through the jaw.
    """
    n_lon = 20
    grid, hidden = {}, set()
    for i, (yy, r, z0) in enumerate(zip(_NECK_Y, _NECK_R, _NECK_Z)):
        for j in range(n_lon):
            u = 2 * math.pi * j / n_lon
            x = r * math.sin(u)
            z = (z0 + r * 1.04 * math.cos(u)) * depth
            idx = len(verts)
            grid[(i, j)] = idx
            verts.append((x, yy, z))
            nn = np.array([math.sin(u), 0.12, math.cos(u)])
            nn /= np.linalg.norm(nn)
            norms.append(tuple(nn))
            # occlusion: inside the head outline and behind its surface?
            if abs(x) <= float(sil_w(np.array(yy))) and yy <= 1.14:
                zs = float(zfield(np.array([[x]]), np.array([[yy]]))[0, 0]) * depth
                if z < zs:
                    hidden.add(idx)
    for i in range(len(_NECK_Y)):
        for j in range(n_lon):
            a, b = grid[(i, j)], grid[(i, (j + 1) % n_lon)]
            if a not in hidden and b not in hidden:
                edges.append([a, b])
            if i < len(_NECK_Y) - 1:
                c = grid[(i + 1, j)]
                if a not in hidden and c not in hidden:
                    edges.append([a, c])

    # sternocleidomastoid: mastoid -> sternal notch, bowed slightly outward
    for side in (-1, 1):
        base = len(verts)
        pts = []
        for k in range(14):
            t = k / 13.0
            bow = math.sin(math.pi * t) * 0.055
            x = side * ((1 - t) * MASTOID[0] + t * STERNAL[0] + bow)
            y = (1 - t) * MASTOID[1] + t * STERNAL[1]
            z = ((1 - t) * MASTOID[2] + t * STERNAL[2] + bow * 0.5) * depth
            pts.append((x, y, z))
        for (x, y, z) in pts:
            verts.append((x, y, z))
            nn = np.array([x, 0.10, max(z, 0.05)])
            nn /= np.linalg.norm(nn)
            norms.append(tuple(nn))
        for k in range(len(pts) - 1):
            if not (abs(pts[k][0]) <= float(sil_w(np.array(pts[k][1])))
                    and pts[k][2] < float(zfield(np.array([[pts[k][0]]]),
                                                 np.array([[pts[k][1]]]))[0, 0]) * depth):
                edges.append([base + k, base + k + 1])

    # submandibular line - the underside of the jaw sitting on the neck
    for side in (-1, 1):
        base = len(verts)
        pts = [(side * 0.540, -0.440), (side * 0.470, -0.560),
               (side * 0.360, -0.680), (side * 0.235, -0.780),
               (side * 0.105, -0.838), (0.0, -0.855)]
        for (x, y) in pts:
            z = float(zfield(np.array([[x]]), np.array([[y]]))[0, 0]) * depth - 0.02
            verts.append((x, y, z))
            norms.append((0.0, -0.45, 0.89))
        for k in range(len(pts) - 1):
            edges.append([base + k, base + k + 1])


# --------------------------------------------------------------------------
# Animated mouth: re-contour ONLY the mouth box each frame.
# Rebuilding the whole field costs 4.8 s; the box is ~5% of the grid, and the
# field outside it is untouched, so contours still meet the border correctly.
# --------------------------------------------------------------------------
_MCACHE = {}


def _mouth_grid(res_box=112, depth=1.05):
    key = (res_box, depth)
    if key in _MCACHE:
        return _MCACHE[key]
    bx0, bx1, by0, by1 = MOUTH_BOX
    gx = np.linspace(bx0, bx1, res_box)
    gy = np.linspace(by1, by0, int(res_box * (by1 - by0) / (bx1 - bx0)))
    X, Y = np.meshgrid(gx, gy)
    base = zfield(X, Y, lips=False)          # everything except the lips
    mask = np.abs(X) <= sil_w(Y)
    _MCACHE[key] = (gx, gy, X, Y, base, mask)
    return _MCACHE[key]


def mouth_geometry(openness=0.0, rounding=0.0, base_index=0, depth=1.05,
                   z_top=0.95, z_bot=-0.25, z_step=0.05):
    """Returns (verts, edges, normals) for the mouth region at this pose."""
    gx, gy, X, Y, base, mask = _mouth_grid(depth=depth)
    Z = base + lip_field(X, Y, openness, rounding)
    Zm = np.where(mask, Z, -9.0)
    gzy, gzx = np.gradient(Z, gy[1] - gy[0], gx[1] - gx[0])

    ny, nx = Z.shape
    P, N, E = [], [], []
    n_pts = 0
    # Only sweep levels that actually cut this box. The mouth occupies a narrow
    # band of z, so iterating all 25 global levels was ~4x wasted work.
    lo = max(z_bot, math.floor(Zm[mask].min() / z_step) * z_step)
    hi = min(z_top, math.ceil(Z[mask].max() / z_step) * z_step)
    for lev in np.arange(hi, lo - 1e-9, -z_step):
        for poly in _march(Zm >= float(lev), gx, gy):
            # drop the run that hugs the box border - that edge belongs to the
            # static geometry, and keeping it would draw a rectangle
            ix = np.clip(((poly[:, 0] - gx[0]) / (gx[1] - gx[0])).astype(int), 0, nx - 1)
            iy = np.clip(((poly[:, 1] - gy[0]) / (gy[1] - gy[0])).astype(int), 0, ny - 1)
            edge = (ix <= 1) | (ix >= nx - 2) | (iy <= 1) | (iy >= ny - 2)
            keep = np.where(~edge)[0]
            if len(keep) < 4:
                continue
            for run_i in np.split(keep, np.where(np.diff(keep) != 1)[0] + 1):
                if len(run_i) > 3:
                    _emit_run(poly, ix, iy, run_i, lev, gzx, gzy,
                              depth, base_index + n_pts, P, N, E)
                    n_pts += len(run_i)
    if not P:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32),
                np.zeros((0, 3), np.float32))
    return (np.concatenate(P).astype(np.float32),
            np.concatenate(E).astype(np.int32),
            np.concatenate(N).astype(np.float32))


def _emit_run(poly, ix, iy, run_i, lev, gzx, gzy, depth, base, P, N, E):
    idx = run_i
    pts = poly[idx]
    nn = np.stack([-gzx[iy[idx], ix[idx]], -gzy[iy[idx], ix[idx]],
                   np.ones(len(idx))], 1)
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    P.append(np.stack([pts[:, 0], pts[:, 1] + 0.20,
                       np.full(len(idx), float(lev) * depth)], 1))
    N.append(nn)
    k = np.arange(len(idx) - 1) + base
    E.append(np.stack([k, k + 1], 1))


def speech_params(t):
    """Syllable bursts with pauses - a steady cycle reads as a puppet."""
    word = 2.35
    ph = (t % word) / word
    if ph < 0.10:
        gate = 0.5 - 0.5 * math.cos(math.pi * ph / 0.10)
    elif ph < 0.62:
        gate = 1.0
    else:
        gate = max(0.0, 1.0 - (ph - 0.62) / 0.10)
    f = 4.1 + 0.8 * math.sin(t * 0.63)
    env = abs(math.sin(math.pi * f * t)) ** 0.65
    amp = 0.52 + 0.30 * math.sin(t * 2.17) + 0.16 * math.sin(t * 5.31)
    return gate * env * max(0.15, amp), \
        0.55 * math.sin(t * 1.43) + 0.25 * math.sin(t * 3.7)


# --------------------------------------------------------------------------
# Pose table. Re-contouring costs ~28 ms/frame, which would cap the display at
# ~20 fps. But the mouth is a 2-parameter pose, so every reachable shape can be
# contoured once at startup and simply looked up at runtime. Build costs about
# 2 s; per-frame cost drops to an array slice.
# --------------------------------------------------------------------------
_POSES = None
_POSE_O = 12          # openness steps
_POSE_R = 5           # rounding steps


def build_pose_table(base_index, depth=1.05, verbose=False):
    global _POSES
    tbl = []
    for i in range(_POSE_O):
        row = []
        for j in range(_POSE_R):
            o = i / (_POSE_O - 1)
            r = -1.0 + 2.0 * j / (_POSE_R - 1)
            row.append(mouth_geometry(o, r, base_index, depth))
        tbl.append(row)
        if verbose:
            print(f"  pose row {i + 1}/{_POSE_O}", flush=True)
    _POSES = tbl
    return tbl


def mouth_pose(openness, rounding):
    """Nearest cached pose. 12 openness steps is finer than a ~4 Hz syllable
    rate can resolve, so the quantisation is not visible."""
    i = int(round(np.clip(openness, 0, 1) * (_POSE_O - 1)))
    j = int(round((np.clip(rounding, -1, 1) + 1.0) * 0.5 * (_POSE_R - 1)))
    return _POSES[i][j]
