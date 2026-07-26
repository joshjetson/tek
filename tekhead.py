#!/usr/bin/env python3
"""
tekhead - a smooth bald head for the Tektronix vector display.

Built as a stack of horizontal cross-sections rather than a surface of
revolution, because a head is not rotationally symmetric: the face is flatter
than the cranium, so every ring needs its own front depth and back depth.
Each cross-section is an egg shape:

    x = wx * sin(phi)
    z = (front ? zf : zb) * cos(phi)

continuous at phi = pi/2 where cos(phi) is 0, so the front/back join is seamless.

Two things make it read as a face rather than an egg:

  * BACK-FACE CULLING. Without it the rear of the skull draws straight through
    the face and the feature lines vanish into the noise. Each vertex carries an
    approximate outward normal; an edge is kept only if at least one end faces
    the camera (keeping one end preserves the silhouette).
  * A SPARSE MESH. The features have to out-weigh the wireframe, so the mesh is
    deliberately coarse and the eyes/nose/mouth are drawn large.
"""
import math

import numpy as np

# y, half-width, front depth, back depth.  y: -1 = chin, +1 = crown.
# Dr-Manhattan-ish: big smooth cranium, strong cheekbones, defined jaw, no hair.
_PROFILE = np.array([
    (-1.00, 0.232, 0.330, 0.292),   # chin (rounded, not pointed)
    (-0.90, 0.336, 0.422, 0.432),
    (-0.78, 0.432, 0.478, 0.560),   # jaw corner
    (-0.62, 0.522, 0.512, 0.660),
    (-0.46, 0.576, 0.538, 0.712),   # mouth
    (-0.28, 0.612, 0.552, 0.750),   # nose base
    (-0.10, 0.640, 0.548, 0.778),   # cheekbone
    (0.10, 0.652, 0.534, 0.792),    # eyes
    (0.28, 0.648, 0.542, 0.796),    # brow
    (0.46, 0.645, 0.532, 0.788),    # forehead
    (0.62, 0.622, 0.500, 0.760),
    (0.76, 0.566, 0.448, 0.700),
    (0.88, 0.462, 0.360, 0.590),    # crown
    (0.96, 0.330, 0.250, 0.412),
    # A small cap ring, NOT a pole: converging every meridian on one point
    # produces a starburst artefact right on top of the head.
    (1.005, 0.155, 0.120, 0.180),
], dtype=np.float32)

# Underside of the jaw, so the head is a closed volume rather than an open tube.
_UNDER = np.array([
    (-1.00, 0.232, 0.330, 0.292),
    (-1.07, 0.150, 0.238, 0.228),
    # small cap ring, not a pole - same starburst reason as the crown
    (-1.115, 0.088, 0.130, 0.126),
], dtype=np.float32)

# Ellipsoid semi-axes used to approximate outward normals for culling.
_AX, _AY, _AZ = 0.65, 1.05, 0.80


def _tab(y):
    return _UNDER if y < -1.0 else _PROFILE


def _wx(y):
    t = _tab(y); return float(np.interp(y, t[:, 0], t[:, 1]))


def _zf(y):
    t = _tab(y); return float(np.interp(y, t[:, 0], t[:, 2]))


def _zb(y):
    t = _tab(y); return float(np.interp(y, t[:, 0], t[:, 3]))


def surf(y, phi, swell=1.0):
    """Point on the head surface. phi=0 is straight ahead (+Z)."""
    c, s = math.cos(phi), math.sin(phi)
    z = (_zf(y) if c >= 0 else _zb(y)) * c
    return (_wx(y) * s * swell, y, z * swell)


def head_model(n_lon=16, n_rings=12, scale=1.30):
    """Returns (verts, edges, normals)."""
    verts, edges = [], []

    def add(p):
        verts.append((p[0] * scale, p[1] * scale, p[2] * scale))
        return len(verts) - 1

    # ---- cranium / face mesh --------------------------------------------
    ys = np.linspace(-1.115, 1.005, n_rings)
    grid = {}
    for i, y in enumerate(ys):
        if _wx(y) < 1e-4:
            idx = add(surf(y, 0.0))
            for j in range(n_lon):
                grid[(i, j)] = idx
            continue
        for j in range(n_lon):
            grid[(i, j)] = add(surf(y, 2 * math.pi * j / n_lon))

    for i in range(n_rings):
        for j in range(n_lon):
            a, b = grid[(i, j)], grid[(i, (j + 1) % n_lon)]
            if a != b:
                edges.append((a, b))
            if i < n_rings - 1:
                c = grid[(i + 1, j)]
                if a != c:
                    edges.append((a, c))

    PROUD = 1.030

    def loop(pts, close=True):
        idx = [add(p) for p in pts]
        for k in range(len(idx) - 1):
            edges.append((idx[k], idx[k + 1]))
        if close:
            edges.append((idx[-1], idx[0]))
        return idx

    # ---- eyes -------------------------------------------------------------
    for side in (-1, 1):
        phi_c, y_c = side * 0.425, 0.115
        pts = []
        for k in range(22):
            t = 2 * math.pi * k / 22
            dphi = 0.275 * math.cos(t)
            # squared-off sine -> pointed inner and outer corners
            dy = 0.115 * math.sin(t) * (abs(math.sin(t)) ** 0.35)
            pts.append(surf(y_c + dy, phi_c + dphi, PROUD))
        loop(pts)
        # iris only - Dr Manhattan's eyes have no visible pupil
        loop([surf(y_c + 0.062 * math.sin(2 * math.pi * k / 14),
                   phi_c + 0.098 * math.cos(2 * math.pi * k / 14), PROUD + 0.010)
              for k in range(14)])

    # ---- brow ridge --------------------------------------------------------
    for side in (-1, 1):
        pts = []
        for k in range(11):
            u = k / 10.0
            phi = side * (0.155 + 0.560 * u)        # spans a real arc
            pts.append(surf(0.268 + 0.052 * math.sin(math.pi * u), phi, PROUD))
        loop(pts, close=False)

    # ---- nose --------------------------------------------------------------
    bridge = [(0.0, 0.255, _zf(0.255) + 0.020),
              (0.0, 0.140, _zf(0.140) + 0.062),
              (0.0, 0.015, _zf(0.015) + 0.112),
              (0.0, -0.105, _zf(-0.105) + 0.168),
              (0.0, -0.196, _zf(-0.196) + 0.205)]      # tip
    bi = loop(bridge, close=False)
    base_i = add((0.0, -0.262, _zf(-0.262) + 0.078))
    edges.append((bi[-1], base_i))
    for side in (-1, 1):
        wing = [(side * 0.062, -0.200, _zf(-0.200) + 0.168),
                (side * 0.118, -0.238, _zf(-0.238) + 0.112),
                (side * 0.100, -0.284, _zf(-0.284) + 0.062)]
        wi = loop(wing, close=False)
        edges.append((bi[-1], wi[0]))
        edges.append((wi[-1], base_i))

    # ---- mouth --------------------------------------------------------------
    upper, lower = [], []
    for k in range(15):
        u = k / 14.0
        phi = (u - 0.5) * 0.86
        bow = math.sin(math.pi * u)
        upper.append(surf(-0.412 - 0.030 * bow, phi, PROUD))
        lower.append(surf(-0.470 + 0.026 * bow, phi, PROUD))
    ui, li = loop(upper, close=False), loop(lower, close=False)
    edges.append((ui[0], li[0]))
    edges.append((ui[-1], li[-1]))

    # ---- ears ---------------------------------------------------------------
    for side in (-1, 1):
        pts = []
        for k in range(12):
            t = 2 * math.pi * k / 12
            p = surf(-0.02 + 0.155 * math.sin(t),
                     side * (math.pi / 2 + 0.11 * math.cos(t) - 0.14), 1.0)
            pts.append((p[0] * 1.055, p[1], p[2] * 1.055))
        loop(pts)

    # ---- neck ---------------------------------------------------------------
    prev = None
    for y in (-0.92, -1.10, -1.30, -1.52):
        ring = []
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            ring.append(add((0.285 * math.sin(phi), y,
                             0.235 * math.cos(phi) - 0.055)))
        for j in range(n_lon):
            edges.append((ring[j], ring[(j + 1) % n_lon]))
        if prev:
            edges.extend((prev[j], ring[j]) for j in range(n_lon))
        prev = ring

    v = np.array(verts, np.float32)
    e = np.array(edges, np.int32)
    # Ellipsoid-gradient normals: good enough for culling on a convex-ish head,
    # and far cheaper than real per-face normals.
    n = np.stack([v[:, 0] / _AX ** 2, v[:, 1] / _AY ** 2, v[:, 2] / _AZ ** 2], 1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return v, e, n.astype(np.float32)


def head(**kw):
    v, e, _ = head_model(**kw)
    return v, e


def build_pts_culled(verts, edges, normals, w, h, rot, dist=3.0, eps=-0.10,
                     mode="and", fov=1.35):
    """Project, dropping edges whose both ends face away from the camera.

    Keeping an edge when *either* end faces us preserves the silhouette, which
    is what gives the head its outline. eps slightly negative keeps a little of
    the terminator so the shape does not look sheared off at the edges.
    """
    from tekvector import project, rotate
    R = rotate(np.eye(3, dtype=np.float32), *rot)
    v = verts @ R
    n = normals @ R
    p2, _ = project(v, w, h, dist, fov)
    vis = n[:, 2] > eps
    # "or" keeps the silhouette but leaves ragged stubs poking past the outline
    # where an edge straddles the terminator; "and" is clean.
    keep = (vis[edges[:, 0]] & vis[edges[:, 1]]) if mode == "and" \
        else (vis[edges[:, 0]] | vis[edges[:, 1]])
    return np.rint(p2[edges[keep]]).astype(np.int32)


def _register():
    import tekvector
    tekvector.MODELS["head"] = head


_register()


# ---------------------------------------------------------------------------
# Mannequin form, matching the free3d reference: a tall smooth ovoid that
# tapers to a narrow neck and then flares into a wide display base. The
# reference is essentially a surface of revolution with a very fine quad mesh
# and no facial features at all.
# ---------------------------------------------------------------------------
# (y, half-width). Depth is derived from width - the reference is near
# rotationally symmetric, just slightly deeper than wide like a real skull.
_MANQ = np.array([
    # Smooth dome. Ends on a small cap ring, not a pole - converging 48
    # meridians on one point makes a bright nipple on the crown.
    (0.995, 0.082), (0.980, 0.132), (0.960, 0.196), (0.930, 0.272),
    (0.890, 0.358), (0.840, 0.444), (0.780, 0.524), (0.710, 0.596),
    (0.630, 0.658), (0.530, 0.710), (0.420, 0.744), (0.300, 0.762),
    (0.170, 0.768), (0.040, 0.764),                      # widest plateau
    (-0.090, 0.752), (-0.220, 0.734), (-0.350, 0.710), (-0.470, 0.680),
    (-0.590, 0.644), (-0.700, 0.602), (-0.800, 0.556), (-0.890, 0.506),
    (-0.960, 0.456), (-1.015, 0.412), (-1.055, 0.386), (-1.080, 0.376),
    # neck pinch, then the base flares out and down. Left OPEN at the rim -
    # the reference is a hollow display stand, not a closed bowl.
    (-1.110, 0.392), (-1.155, 0.446), (-1.215, 0.522), (-1.290, 0.604),
    (-1.375, 0.678), (-1.470, 0.734), (-1.565, 0.768), (-1.655, 0.784),
    (-1.720, 0.790),
], dtype=np.float32)


def _mw(y):
    return float(np.interp(y, _MANQ[::-1, 0], _MANQ[::-1, 1]))


def mannequin_model(n_lon=48, n_rings=46, scale=1.05, depth=1.06):
    """Dense featureless mannequin head. Returns (verts, edges, normals)."""
    verts, edges, grid = [], [], {}

    ys = np.linspace(_MANQ[-1, 0], _MANQ[0, 0], n_rings)
    for i, y in enumerate(ys):
        r = _mw(y)
        if r < 1e-4:
            grid[(i, 0)] = len(verts)
            verts.append((0.0, y * scale, 0.0))
            for j in range(1, n_lon):
                grid[(i, j)] = grid[(i, 0)]
            continue
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            grid[(i, j)] = len(verts)
            verts.append((r * math.sin(phi) * scale,
                          y * scale,
                          r * math.cos(phi) * depth * scale))

    for i in range(n_rings):
        for j in range(n_lon):
            a, b = grid[(i, j)], grid[(i, (j + 1) % n_lon)]
            if a != b:
                edges.append((a, b))
            if i < n_rings - 1:
                c = grid[(i + 1, j)]
                if a != c:
                    edges.append((a, c))

    v = np.array(verts, np.float32)
    # Centre vertically: the form runs from the base rim to the crown, so its
    # midpoint is well below the origin and it would frame off-centre.
    v[:, 1] -= 0.5 * (v[:, 1].max() + v[:, 1].min())
    e = np.array(edges, np.int32)
    # True surface normal for a surface of revolution: cross of the along-profile
    # tangent and the around-ring tangent. Much better than the ellipsoid
    # approximation once the form has a concave pinch at the neck.
    n = np.zeros_like(v)
    for i in range(n_rings):
        for j in range(n_lon):
            a = grid[(i, j)]
            i0, i1 = max(i - 1, 0), min(i + 1, n_rings - 1)
            du = v[grid[(i1, j)]] - v[grid[(i0, j)]]
            dv = v[grid[(i, (j + 1) % n_lon)]] - v[grid[(i, (j - 1) % n_lon)]]
            nn = np.cross(dv, du)
            if np.linalg.norm(nn) > 1e-9:
                n[a] = nn / np.linalg.norm(nn)
    # point them outward
    flip = np.einsum("ij,ij->i", n, v - np.array([0, v[:, 1].mean(), 0])) < 0
    n[flip] *= -1
    return v, e, n.astype(np.float32)


tekvector_registered = False


def _register_manq():
    import tekvector
    tekvector.MODELS["mannequin"] = lambda **kw: mannequin_model(**kw)[:2]


_register_manq()
