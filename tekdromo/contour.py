"""
The contour generator - literally the loop from the FDL spec:

    for z = .95 down to -.25 step -.05:
        intersect the surface, extract closed contours, simplify, emit vectors

This is why the whole approach works: contours are LEVEL SETS of the real
surface, so they flow around every feature for free. face_1 drew feature curves
onto an undeformed mesh and they always read as decals.
"""
import math

import numpy as np

from .anatomy import (EAR_BOT, EAR_TILT, EAR_TOP, EYE_OPEN, JAW,
                      LOWER_LIP_CURVE, UPPER_LIP_CURVE, ear_field, sil_w)
from .field import MOUTH_BOX, head_mask, zfield


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
