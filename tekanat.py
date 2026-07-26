#!/usr/bin/env python3
"""
tekanat - an anatomical human head for the vector display.

The mannequin was a surface of revolution. A real face cannot be: the eye
sockets go IN, the nose comes OUT, the lips sit proud of a groove. So this
builds a dense base ovoid and then pushes the surface around with a set of
localised anatomical displacements.

That distinction matters for wireframe specifically. Drawing eye/nose/mouth
lines onto a smooth surface (what the first head did) reads as a doodle on an
egg. Displacing the actual surface makes the horizontal contour rings dip into
the sockets and bulge over the nose - and those deflected rings are what the
eye reads as a face, exactly as in the reference render.

Normals are recomputed AFTER displacement, so back-face culling correctly hides
the inside of the sockets and the far side of the nose.

Landmarks use the standard canon: eyes at half the head height, nose base
one third up from the chin, mouth at about a fifth.
"""
import math

import numpy as np

# y, half-width, front depth, back depth.  y: -1.06 chin, +1.0 crown.
_PROF = np.array([
    # Widths MEASURED from the free3d reference front view (silhouette
    # extracted by saturation segmentation, normalised by crown->chin).
    # The ear zone (v 0.19-0.50) is interpolated out: the reference
    # silhouette there is EAR, not skull - fitting it would have baked a
    # false 'jaw corner' into the head. Ears are added separately.
    # Depths are NOT measurable (the profile view's back of skull is
    # clipped by the image border), so they use anatomical ratios:
    # depth ~1.23x width, split ~42/58 front/back.
    ( 1.000, 0.072, 0.061, 0.083),
    ( 0.980, 0.220, 0.189, 0.257),
    ( 0.920, 0.392, 0.344, 0.477),
    ( 0.840, 0.502, 0.455, 0.646),
    ( 0.720, 0.616, 0.582, 0.838),
    ( 0.600, 0.686, 0.672, 0.974),
    ( 0.480, 0.722, 0.719, 1.034),
    ( 0.360, 0.736, 0.745, 1.063),
    ( 0.240, 0.738, 0.754, 1.067),
    ( 0.120, 0.736, 0.755, 1.058),
    ( 0.000, 0.728, 0.750, 1.041),
    (-0.160, 0.694, 0.711, 0.963),
    (-0.280, 0.668, 0.682, 0.906),
    (-0.400, 0.643, 0.656, 0.840),
    (-0.520, 0.617, 0.630, 0.775),
    (-0.640, 0.592, 0.601, 0.702),
    (-0.680, 0.568, 0.575, 0.654),
    (-0.760, 0.548, 0.550, 0.594),
    (-0.840, 0.502, 0.497, 0.502),
    (-0.920, 0.458, 0.444, 0.412),
    (-1.000, 0.400, 0.380, 0.320),
    (-1.035, 0.330, 0.370, 0.330),
    (-1.075, 0.245, 0.300, 0.260),
    (-1.110, 0.120, 0.185, 0.150),
], dtype=np.float32)

# --- nose profile: amplitude and half-width of the ridge, bridge -> base ----
# Nose, measured: nasion v=0.450 (y=-0.100) to base v=0.215 (y=-0.570),
# so 0.235H long; alar width 0.180H. Was 1.36x too long and 0.73x too narrow -
# a long narrow nose is one of the strongest masculine cues.
_NY = np.array([-0.590, -0.570, -0.548, -0.522, -0.484, -0.430, -0.368,
                -0.300, -0.232, -0.160, -0.080])
_NA = np.array([0.000, 0.062, 0.112, 0.136, 0.128, 0.104, 0.078,
                0.052, 0.030, 0.011, 0.000])
_NW = np.array([0.132, 0.152, 0.148, 0.134, 0.118, 0.102, 0.088,
                0.080, 0.077, 0.082, 0.096])


# Landmark heights MEASURED from the reference (darkness minima along the
# facial midline and through the eye column), converted to model y = -1 + 2v:
#     eye line v=0.474 -> -0.052   nostrils v=0.207 -> -0.586
#     mouth    v=0.097 -> -0.806
# The features were originally placed by canon and sat too high and too
# compressed. Rather than re-tune every Gaussian, map the vertical axis: the
# tuned feature shapes stay, they just land where the reference puts them.
# Features are specified DIRECTLY at measured positions now. The old->new
# remap layer was compressing the lower face by 0.34x, which is what flattened
# the lips to 42% of the reference height.
_Y_OLD = np.array([-2.0, 2.0])
_Y_NEW = np.array([-2.0, 2.0])


def _to_old(y):
    """new model space -> the space the feature formulas were tuned in"""
    return np.interp(y, _Y_NEW, _Y_OLD)


def _to_new(y):
    return np.interp(y, _Y_OLD, _Y_NEW)


def _col(y, i):
    return np.interp(y, _PROF[::-1, 0], _PROF[::-1, i])


def _g(d):
    return np.exp(-d * d)


def _base(Y, PHI):
    C, S = np.cos(PHI), np.sin(PHI)
    wx, zf, zb = _col(Y, 1), _col(Y, 2), _col(Y, 3)
    return np.stack([wx * S, Y, np.where(C >= 0, zf, zb) * C], -1)


def _radial(P):
    """Cheap outward direction: away from the vertical axis, biased forward."""
    d = P.copy()
    d[..., 1] = 0.0
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-9)


def surface(Y, PHI, lift=0.0):
    """Point on the DISPLACED head surface for arbitrary (y, phi)."""
    P = _base(Y, PHI)
    X = P[..., 0]
    C = np.cos(PHI)
    front = np.clip(C, 0.0, 1.0) ** 1.3
    ax = np.abs(X)
    D = _disp(X, Y, ax, front)
    P = P + _radial(P) * D[..., None]
    P[..., 2] += _nose(X, Y, ax, front)
    if np.ndim(lift) or lift:
        L = np.asarray(lift, float)
        P = P + _radial(P) * (L[..., None] if np.ndim(L) else L)
    return P


def phi_for_x(y, x):
    """Azimuth that lands at a given x on the ring at height y - lets features
    be specified in natural face coordinates instead of angles."""
    return np.arcsin(np.clip(x / np.maximum(_col(y, 1), 1e-6), -1.0, 1.0))


def _disp(X, Y, ax, front):
    """Centres below are measured off a normalised grid overlaid on the
    reference (origin = chin on the midline, unit = crown-to-chin height):
        eye   v=0.445  nose base v=0.215  mouth v=0.105
    converted with y = -1 + 2v."""
    Y = _to_old(Y)
    D = np.zeros_like(Y)
    D -= 0.062 * front * _g((ax - 0.370) / 0.190) * _g((Y + 0.139) / 0.105)
    D += 0.040 * front * _g((ax - 0.370) / 0.130) * _g((Y + 0.139) / 0.066)
    D -= 0.020 * front * _g((ax - 0.370) / 0.155) * _g((Y + 0.055) / 0.034)
    D += 0.030 * front * _g((ax - 0.430) / 0.200) * _g((Y + 0.300) / 0.170)
    D -= 0.026 * front * _g((ax - 0.640) / 0.150) * _g((Y - 0.190) / 0.180)
    lipw = _g(X / 0.262)                                              # mouth
    D += 0.040 * front * lipw * _g((Y + 0.812) / 0.048)               # upper lip
    D += 0.046 * front * lipw * _g((Y + 0.898) / 0.058)               # lower lip (fuller)
    D -= 0.030 * front * _g(X / 0.300) * _g((Y + 0.855) / 0.022)      # lip line
    D -= 0.019 * front * _g(X / 0.042) * _g((Y + 0.720) / 0.062)      # philtrum
    D += 0.032 * front * _g(X / 0.165) * _g((Y + 0.945) / 0.075)      # chin
    D += 0.020 * _g((ax - 0.470) / 0.140) * _g((Y + 0.830) / 0.110)   # jaw
    return D


def _nose(X, Y, ax, front):
    """Pushed along +Z, not along the normal - a nose sticks forward."""
    Y = _to_old(Y)
    nose = np.interp(Y, _NY, _NA) * _g(X / np.interp(Y, _NY, _NW)) * front
    nose += 0.040 * front * _g((ax - 0.132) / 0.060) * _g((Y + 0.556) / 0.042)
    return nose


# (cx, cy, rx, ry, amplitude) - repulsion centres in face coords, new-space y.
# The mesh is redistributed AROUND these, which is what makes contour lines
# arch over the brow and bow around the eyes and mouth.
_WARP = (
    (-0.370, -0.139, 0.255, 0.145, 0.185),   # left eye
    (0.370, -0.139, 0.255, 0.145, 0.185),    # right eye
    (0.000, -0.855, 0.330, 0.150, 0.155),    # mouth
)


def _warp_params(Y, PHI):
    """Slide vertices ALONG the surface so the mesh flows around the features.

    This is the piece that was missing. The anatomical displacement is mostly
    along +Z, and in a near-orthographic front view Z motion barely changes
    where a vertex projects - so the grid stayed dead straight through the face
    no matter how strong the displacement got. Deflecting the lines requires
    moving vertices *tangentially*, in x and y.

    Each feature repels the parameterisation radially, with the push peaking at
    the feature boundary (d = 1) and decaying either side:

        m(d) = A * d * exp((1 - d^2) / 2)

    so the centre stays put, the ring of mesh just outside it gets pushed
    outward, and the effect dies away smoothly. The result is contour lines
    that bow around the eye and arch over the brow instead of cutting through.
    """
    C = np.cos(PHI)
    front = np.clip(C, 0.0, 1.0) ** 1.2
    x = _col(Y, 1) * np.sin(PHI)
    y = Y.copy()
    for cx, cy, rx, ry, A in _WARP:
        u = (x - cx) / rx
        v = (y - cy) / ry
        d = np.sqrt(u * u + v * v) + 1e-6
        m = A * d * np.exp((1.0 - d * d) / 2.0) * front
        x = x + m * (u / d) * rx
        y = y + m * (v / d) * ry
    wx2 = np.maximum(_col(y, 1), 1e-6)
    phi2 = np.arcsin(np.clip(x / wx2, -1.0, 1.0))
    # arcsin only covers the front half; mirror it for the back of the skull
    phi2 = np.where(C >= 0, phi2, np.pi - phi2)
    return y, phi2


def anatomical_head(n_lon=54, n_rings=52, scale=1.15, ears=True):
    """Returns (verts, edges, normals)."""
    ys = np.linspace(_PROF[-1, 0], _PROF[0, 0], n_rings)
    phis = np.linspace(0.0, 2 * math.pi, n_lon, endpoint=False)
    Y = np.repeat(ys[:, None], n_lon, 1)
    PHI = np.repeat(phis[None, :], n_rings, 0)

    def normals_of(P):
        du = np.empty_like(P)
        du[1:-1] = P[2:] - P[:-2]
        du[0], du[-1] = P[1] - P[0], P[-1] - P[-2]
        dv = np.roll(P, -1, 1) - np.roll(P, 1, 1)
        N = np.cross(dv, du)
        n = np.linalg.norm(N, axis=-1, keepdims=True)
        N = N / np.maximum(n, 1e-9)
        axis = P.copy(); axis[..., 1] = 0.0
        flip = np.einsum("ijk,ijk->ij", N, axis) < 0
        N[flip] *= -1
        return N

    Yw, PHIw = _warp_params(Y, PHI)
    P = surface(Yw, PHIw)
    N = normals_of(P)

    # ---- flatten grid to verts/edges -------------------------------------
    verts = (P.reshape(-1, 3) * scale).astype(np.float32)
    norms = N.reshape(-1, 3).astype(np.float32)
    idx = np.arange(n_rings * n_lon).reshape(n_rings, n_lon)

    e_ring = np.stack([idx, np.roll(idx, -1, 1)], -1).reshape(-1, 2)
    e_mer = np.stack([idx[:-1], idx[1:]], -1).reshape(-1, 2)
    edges = np.concatenate([e_ring, e_mer]).astype(np.int32)

    verts, norms, edges = _add_neck(verts, norms, edges, scale, n_lon)
    verts, norms, edges = _add_features(verts, norms, edges, scale)
    if ears:
        verts, norms, edges = _add_ears(verts, norms, edges, scale)
    return verts, edges, norms


def _add_neck(verts, norms, edges, scale, n_lon):
    """The reference has a substantial neck; the earlier head had none at all.
    Half-width 0.170H just below the chin, measured from the front view."""
    nv, nn, ne = [], [], []
    base = len(verts)
    prev = None
    for y, r in ((-0.96, 0.352), (-1.10, 0.344), (-1.30, 0.350), (-1.52, 0.372)):
        ring = []
        for j in range(n_lon // 2):
            phi = 2 * math.pi * j / (n_lon // 2)
            ring.append(base + len(nv))
            nv.append((r * math.sin(phi) * scale, y * scale,
                       (r * 0.86 * math.cos(phi) - 0.10) * scale))
            nn.append((math.sin(phi), 0.0, math.cos(phi)))
        m = len(ring)
        for j in range(m):
            ne.append((ring[j], ring[(j + 1) % m]))
        if prev:
            ne.extend((prev[j], ring[j]) for j in range(m))
        prev = ring
    return (np.concatenate([verts, np.array(nv, np.float32)]),
            np.concatenate([norms, np.array(nn, np.float32)]),
            np.concatenate([edges, np.array(ne, np.int32)]))



# Nose ridge half-width, measured off a 0.01H grid on the reference: narrow at
# the bridge, splaying toward the base. Model y via v -> y = (v-0.107)/0.475-0.838
# Curved, not linear: the ridge stays narrow from the bridge down to about
# v=0.32 and only then flares to the wings. A straight taper reads as a tent.
# Continues up past the eye line so the lines blend into the glabella instead
# of stopping dead - the reference has no hard termination at the top.
_RIDGE_Y = np.array([-0.640, -0.610, -0.575, -0.540, -0.505, -0.470, -0.430,
                     -0.385, -0.335, -0.280, -0.220, -0.155, -0.085, -0.010,
                     0.070])
# Re-read: the dorsum between the eye sockets is 0.055H half-width, not the
# 0.030H I first took (that was the specular highlight, not the full ridge).
# So the flare from bridge to wings is only ~1.7x. The earlier 3.2x pinched it
# into a spike.
_RIDGE_W = np.array([0.190, 0.188, 0.182, 0.174, 0.164, 0.154, 0.145,
                     0.136, 0.126, 0.117, 0.109, 0.101, 0.095, 0.091,
                     0.090])


def _ridge_w(y):
    return np.interp(y, _RIDGE_Y, _RIDGE_W)


def _add_features(verts, norms, edges, scale):
    """Anatomical outline curves lying ON the displaced surface.

    Necessary because a near-orthographic FRONT view barely shows the
    displacement at all: the features are pushed along +Z, and Z motion
    hardly changes where anything projects head-on. The reference render gets
    away with it because it is shaded. With no shading the lines themselves
    have to carry the anatomy, so these loops trace the eye rims, nostrils and
    lips directly. In profile the displacement already does the work.
    """
    nv, nn, ne = [], [], []
    base = len(verts)

    def curve(pts_xy, close=False, lift=0.016):
        """pts_xy: list of (x, y) in face coords -> a loop on the surface."""
        ys = _to_new(np.array([p[1] for p in pts_xy], np.float64))
        xs = np.array([p[0] for p in pts_xy], np.float64)
        phis = phi_for_x(ys, xs)
        P = surface(ys, phis, lift=lift)
        idx = []
        for q in P:
            idx.append(base + len(nv))
            nv.append((q[0] * scale, q[1] * scale, q[2] * scale))
            d = np.array([q[0], 0.0, q[2]])
            d /= max(np.linalg.norm(d), 1e-9)
            nn.append(tuple(d))
        for k in range(len(idx) - 1):
            ne.append((idx[k], idx[k + 1]))
        if close:
            ne.append((idx[-1], idx[0]))
        return idx

    for side in (-1, 1):
        # EYE, measured off a 0.01H grid over the reference:
        #   inner corner (0.088H, 0.428H)   outer corner (0.246H, 0.450H)
        # so the outer corner sits 0.022H HIGHER - a real canthal tilt of ~8
        # degrees, not a level football. The lids are asymmetric too: the upper
        # arches 0.032H with its apex 37% along from the inner corner, the lower
        # drops only 0.023H with its nadir at 53%. That asymmetry is what makes
        # it read as an eye. Spaced slightly further from the nose as asked.
        W = 0.316                      # width, model units (0.158H)
        x_in = side * 0.212            # inner corner, pushed out from the nose
        y_in = -0.161
        rise = 0.044                   # outer corner higher than inner
        up_a, lo_a = 0.064, 0.046      # lid amplitudes
        # skew exponents put the apex/nadir at 37% / 53% along the lid
        p_up, p_lo = 0.697, 1.091

        def lid(t, upper):
            x = x_in + side * t * W
            y = y_in + t * rise
            sh = math.sin(math.pi * (t ** (p_up if upper else p_lo)))
            return (x, y + (up_a * sh if upper else -lo_a * sh))

        pts = [lid(k / 15.0, True) for k in range(16)]
        pts += [lid(1.0 - k / 15.0, False) for k in range(1, 15)]
        curve(pts, close=True)

        # upper lid fold (a real crease in the reference, distinct from a brow)
        curve([lid(k / 12.0, True)[0:1] + (lid(k / 12.0, True)[1] + 0.052,)
               for k in range(13)], lift=0.014)

        # nasolabial fold: barely present in the reference, so keep it short
        curve([(side * (0.126 + 0.062 * (k / 8) ** 0.8),
                -0.612 - 0.150 * (k / 8)) for k in range(9)], lift=0.007)
        curve([(side * (0.048 + 0.062 * (1 + math.cos(math.pi + math.pi * k / 9))),
                -0.566 + 0.034 * math.sin(math.pi * k / 9)) for k in range(10)],
              lift=0.020)
        # side of the nose bridge
        curve([(side * (0.042 + 0.062 * (k / 9) ** 1.7),
                -0.098 - 0.480 * (k / 9)) for k in range(10)], lift=0.012)

    # nose tip underside
    curve([(0.112 * math.cos(math.pi * k / 10), -0.545 - 0.030 * math.sin(math.pi * k / 10))
           for k in range(11)], lift=0.022)

    # (mouth is generated per-frame by mouth_geometry)

    verts = np.concatenate([verts, np.array(nv, np.float32)])
    norms = np.concatenate([norms, np.array(nn, np.float32)])
    edges = np.concatenate([edges, np.array(ne, np.int32)])
    return verts, norms, edges


def _add_ears(verts, norms, edges, scale):
    """Ear as an outline lying in the Y-Z plane against the side of the skull:
    an outer helix oval, an inner concha, and a lobe. The earlier version built
    rings in the wrong plane and rendered as flat tabs sticking straight out.
    """
    nv, nn, ne = [], [], []
    base = len(verts)

    def curve(side, pts_yz, close=False, out=0.0):
        # NOTE: pts_yz carries an optional third element = extra outward flare.
        # Without varying x across the ear the whole outline sits in one Y-Z
        # plane and projects to a bare vertical line when seen head-on.
        idx = []
        for pt in pts_yz:
            yy, zz = pt[0], pt[1]
            out_i = out + (pt[2] if len(pt) > 2 else 0.0)
            # Ears must protrude BEYOND the skull: the reference silhouette at
            # ear level is 0.416H versus a 0.335H skull, so they set the widest
            # point of the head. The previous 0.965 factor put them inside the
            # surface, where they never showed at all.
            xx = side * (_col(np.array(yy), 1) + out_i)
            idx.append(base + len(nv))
            nv.append((float(xx) * scale, yy * scale, zz * scale))
            nn.append((float(side) * 0.82, 0.0, 0.57))
        for k in range(len(idx) - 1):
            ne.append((idx[k], idx[k + 1]))
        if close:
            ne.append((idx[-1], idx[0]))

    for side in (-1, 1):
        # outer helix: tall oval, tipped back slightly, widest at the top
        helix = []
        for k in range(22):
            t = 2 * math.pi * k / 22
            yy = -0.320 + 0.276 * math.sin(t)
            taper = 0.62 + 0.38 * (0.5 + 0.5 * math.sin(t))     # narrower lobe
            zz = -0.105 + 0.135 * taper * math.cos(t) - 0.055 * math.sin(t)
            helix.append((yy, zz, 0.052 * (0.5 - 0.5 * math.cos(t))))
        curve(side, helix, close=True, out=0.150)
        # concha
        conc = []
        for k in range(16):
            t = 2 * math.pi * k / 16
            conc.append((-0.315 + 0.150 * math.sin(t),
                         -0.108 + 0.066 * math.cos(t)))
        curve(side, conc, close=True, out=0.090)
        # tragus / lobe tick
        curve(side, [(-0.520, -0.120), (-0.560, -0.090), (-0.550, -0.055)],
              out=0.105)

    verts = np.concatenate([verts, np.array(nv, np.float32)])
    norms = np.concatenate([norms, np.array(nn, np.float32)])
    edges = np.concatenate([edges, np.array(ne, np.int32)])
    return verts, norms, edges


def _register():
    import tekvector
    tekvector.MODELS["anat"] = lambda **kw: anatomical_head(**kw)[:2]


_register()


# ---------------------------------------------------------------------------
# Talking mouth. Rebuilt every frame from two parameters:
#   openness  0 = closed, 1 = wide open (jaw dropped)
#   rounding -1 = spread wide (an "ee"), +1 = pursed round (an "oo")
# Only ~120 vertices, so regenerating per frame is far cheaper than rebuilding
# the whole head (which takes ~32 ms).
# ---------------------------------------------------------------------------
_MU = np.array([-1.00, -0.94, -0.85, -0.72, -0.58, -0.45, -0.32, -0.244,
                -0.14, -0.07, 0.0, 0.07, 0.14, 0.244, 0.32, 0.45, 0.58,
                0.72, 0.85, 0.94, 1.00])
_M_UPPER = np.array([-0.859, -0.850, -0.838, -0.826, -0.812, -0.800,
                     -0.789, -0.783, -0.786, -0.791, -0.794, -0.791,
                     -0.786, -0.783, -0.789, -0.800, -0.812, -0.826,
                     -0.838, -0.850, -0.859])
_M_LINE = np.array([-0.859, -0.857, -0.855, -0.853, -0.852, -0.852,
                    -0.851, -0.851, -0.851, -0.851, -0.851, -0.851,
                    -0.851, -0.851, -0.851, -0.852, -0.852, -0.853,
                    -0.855, -0.857, -0.859])
_M_LOWER = np.array([-0.859, -0.869, -0.882, -0.896, -0.908, -0.917,
                     -0.925, -0.929, -0.935, -0.939, -0.941, -0.939,
                     -0.935, -0.929, -0.925, -0.917, -0.908, -0.896,
                     -0.882, -0.869, -0.859])


def mouth_geometry(openness=0.0, rounding=0.0, scale=1.15, base_index=0):
    """Returns (verts, norms, edges) for the mouth at this pose."""
    o = float(np.clip(openness, 0.0, 1.0))
    r = float(np.clip(rounding, -1.0, 1.0))

    # Rounding narrows and thickens ("oo"); spreading widens and flattens ("ee").
    half = 0.230 * (1.0 - 0.26 * max(r, 0.0) + 0.13 * max(-r, 0.0))
    # A real jaw drop moves the lower lip much further than the upper.
    up_rise = 0.042 * o
    lo_drop = 0.235 * o
    # Inner (wet) edges part as the mouth opens.
    in_up = 0.078 * o
    in_lo = 0.132 * o

    uu = np.linspace(-0.955, 0.955, 33)
    nv, ne, nn = [], [], []

    # Collect every curve, then evaluate the surface ONCE. Seven separate
    # surface() calls cost 19 ms/frame because each recomputes the whole
    # displacement field (12 gaussians + interpolations) for a handful of points.
    _X, _Y, _L, _SEG = [], [], [], []
    count = [0]

    def add(xs, ys, lift, closed=False):
        # Fully vectorised: the earlier per-point loop with np.linalg.norm on
        # 120 one-element arrays cost 40 ms/frame, all Python overhead.
        ys = np.asarray(ys, float)
        xs = np.asarray(xs, float)
        idx0 = base_index + count[0]
        n = len(ys)
        _X.append(xs if xs.shape == ys.shape else np.full(n, float(xs)))
        _Y.append(ys)
        _L.append(np.full(n, float(lift)))
        k = np.arange(n - 1)
        seg = np.stack([idx0 + k, idx0 + k + 1], 1)
        if closed:
            seg = np.vstack([seg, [[idx0 + n - 1, idx0]]])
        _SEG.append(seg)
        count[0] += n
        return idx0

    xs = uu * half
    upper = np.interp(uu, _MU, _M_UPPER) + up_rise
    line = np.interp(uu, _MU, _M_LINE)
    lower = np.interp(uu, _MU, _M_LOWER) - lo_drop

    add(xs, upper, 0.019)
    add(xs, lower, 0.019)
    if o < 0.04:
        add(xs, line, 0.013)                       # closed: a single lip line
    else:
        # Opening: inner lip edges bounding a dark aperture. Taper to the
        # corners so the opening is a lens, not a rectangle.
        taper = np.sqrt(np.maximum(0.0, 1.0 - uu ** 2))
        add(xs, line + in_up * taper, 0.011)
        add(xs, line - in_lo * taper, 0.011)
        if o > 0.30:                               # hint of the teeth line
            add(xs * 0.86, line + in_up * 0.55 * np.sqrt(
                np.maximum(0.0, 1.0 - (uu / 0.86) ** 2)) + 0.004, 0.006)

    # philtrum, and a chin crease that drops with the jaw so the whole lower
    # face reads as moving rather than just the lips
    for side in (-1, 1):
        k = np.arange(9) / 8.0
        add(side * (0.040 + 0.014 * k ** 1.6), -0.655 - 0.132 * k, 0.017)
    k = np.linspace(-1, 1, 15)
    add(k * 0.150, -0.985 - 0.055 * o - 0.018 * np.cos(math.pi * k * 0.5), 0.010)

    X = np.concatenate(_X); Y = np.concatenate(_Y); L = np.concatenate(_L)
    P = surface(Y, phi_for_x(Y, X), lift=L)
    d = P.copy()
    d[:, 1] = 0.0
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return ((P * scale).astype(np.float32), d.astype(np.float32),
            np.concatenate(_SEG).astype(np.int32))


def speech_params(t):
    """A syllable-ish drive signal. Bursts of 3-6 syllables separated by
    pauses, with the vowel shape wandering, which reads as speech far better
    than a steady open/close cycle."""
    word = 2.35
    ph = (t % word) / word
    gate = 0.5 - 0.5 * math.cos(math.pi * min(ph / 0.10, 1.0)) if ph < 0.10 else (
        1.0 if ph < 0.62 else max(0.0, 1.0 - (ph - 0.62) / 0.10))
    f = 4.1 + 0.8 * math.sin(t * 0.63)
    env = abs(math.sin(math.pi * f * t)) ** 0.65
    amp = 0.52 + 0.30 * math.sin(t * 2.17) + 0.16 * math.sin(t * 5.31)
    return gate * env * max(0.15, amp), 0.55 * math.sin(t * 1.43) + 0.25 * math.sin(t * 3.7)
