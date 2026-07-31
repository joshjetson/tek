#!/usr/bin/env python3
"""
tekrig - expression rig over base face v1 (tekfdl).

ARCHITECTURE
------------
Four layers, each with one job. Nothing below knows about anything above.

    EXPRESSIONS   named presets -> control values          ("thinking", "happy")
        |
    CONTROLS      ~12 named scalars, the ONLY animation state
        |
    REGIONS       a bbox + a field function + which controls touch it
        |
    FIELD/CACHE   re-contour one region, memoised by quantised controls

Why this shape:

* The face is an implicit field, so an expression is just "different numbers in
  the field equation". There are no blendshapes to author and no rig to skin -
  a smile is the lip blobs' centres moving, which the contour generator turns
  into correctly-flowing lines for free.

* Rebuilding the whole field costs ~4 s. But an expression only disturbs a small
  part of it, so each region owns a BOX and only that box is re-contoured. The
  face outside the box is untouched, so contours still meet the border exactly.
  This is the mouth trick from base v1, generalised.

* Every region goes through the SAME contour/cache/compose pipeline. Only the
  field maths differs per region - that is the DRY line. Adding a feature means
  writing one field function and one REGIONS entry; the caching, blending,
  occlusion and compositing are inherited.

* Controls are quantised and geometry is memoised, so a blend between two
  expressions costs a dict lookup after the first pass over that path.

Adding an expression is one line in EXPRESSIONS. Adding a control is one line
in CONTROLS plus a term in the relevant region's field function.
"""
import math
import time
from collections import OrderedDict

import numpy as np

from . import contour as C
from . import field as FLD
from .anatomy import BROW_POLY, EYE, _ridge
from .field import lip_field

# ---------------------------------------------------------------------------
# CONTROLS - the entire animation state of the face.
#   name: (lo, hi, default, quantisation steps)
# Steps set the cache granularity: finer = smoother but more misses. Anything
# the eye cannot resolve at ~40 fps is wasted precision.
# ---------------------------------------------------------------------------
CONTROLS = {
    "brow_raise":  (0.0, 1.0, 0.0, 7),    # both brows up (surprise, attention)
    "brow_furrow": (0.0, 1.0, 0.0, 7),    # inner brows down+together (thought)
    "eye_open":    (0.0, 1.0, 1.0, 7),    # 0 = lids shut, drives blink
    "eye_squint":  (0.0, 1.0, 0.0, 5),    # lower lid up (smile, scrutiny)
    "gaze_x":      (-1.0, 1.0, 0.0, 7),
    "gaze_y":      (-1.0, 1.0, 0.0, 5),
    "mouth_open":  (0.0, 1.0, 0.0, 11),   # jaw drop, drives speech
    "mouth_round": (-1.0, 1.0, 0.0, 5),   # -1 spread "ee" .. +1 pursed "oo"
    "smile":       (-1.0, 1.0, 0.0, 9),   # negative = frown
    "nose_flare":  (0.0, 1.0, 0.0, 3),
    # The third eye: how far the star is raised out of the forehead, and where
    # it has rotated to. See field_third_eye.
    "eye3":        (0.0, 1.0, 0.0, 5),
    # Spin phase, 0..1 across ONE period of a five-pointed star - which is 72
    # degrees, not 360, because the star is its own rotation by symmetry.
    # Twelve steps is 6 degrees apiece: fine enough to read as turning, coarse
    # enough that the whole revolution is twelve cache entries rather than a
    # new contour every frame.
    "eye3_spin":   (0.0, 1.0, 0.0, 12),
}

DEFAULTS = {k: v[2] for k, v in CONTROLS.items()}


def quantise(controls, names=None):
    """Snap controls to their declared cache grid. Returns a hashable key.

    THE quantisation. The step counts in CONTROLS exist precisely so that a
    continuously-varying control (speech driving mouth_open, a camera driving
    gaze) collapses onto a small set of reachable poses that can be cached.

    Region.key used to ignore this and key on round(value * 1000) instead -
    a thousand steps - so an animating control missed the cache on essentially
    every frame. Measured: mouth hit rate 2%, each miss ~53ms, 98% of frames
    over budget. That was the stutter.
    """
    out = []
    for name in (sorted(CONTROLS) if names is None else names):
        lo, hi, _, steps = CONTROLS[name]
        v = float(np.clip(controls.get(name, DEFAULTS[name]), lo, hi))
        out.append(int(round((v - lo) / (hi - lo) * (steps - 1))))
    return tuple(out)


def dequantise(key, names=None):
    """Inverse of quantise: grid index -> control value."""
    out = {}
    for i, name in enumerate(sorted(CONTROLS) if names is None else names):
        lo, hi, _, steps = CONTROLS[name]
        out[name] = lo + (hi - lo) * key[i] / (steps - 1)
    return out


# ---------------------------------------------------------------------------
# REGION FIELD FUNCTIONS
# Each returns the field contribution for its own area given the controls.
# These are the ONLY places expression maths lives.
# ---------------------------------------------------------------------------
def field_brows(X, Y, c):
    raise_, furrow = c["brow_raise"], c["brow_furrow"]
    pts = [(x, y + 0.075 * raise_ - 0.045 * furrow * (1.0 - abs(x) / 0.34))
           for (x, y) in BROW_POLY]
    z = _ridge(X, Y, pts, 0.105 + 0.030 * furrow, 0.085, mirror=True)
    # furrow also pinches the glabella into a vertical crease
    z = z - 0.055 * furrow * np.exp(-(X ** 2) / 0.0016 - ((Y - 0.36) ** 2) / 0.020)
    return z


# Where the star sits and how big it is, in model coordinates - the same space
# the rest of the face is defined in, which is the point of doing it this way.
# amp and lobe were swept against the render, and both matter more than they
# look. The head is sliced at 0.05 in z, so a bump of 0.075 crosses barely one
# contour and the five points never resolve - it reads as a ripple on the
# forehead, not a star. 0.26 crosses five or six, which is what makes the shape
# legible as contour lines rather than as a smudge. lobe below about 0.5 rounds
# the points off; above 0.7 they go spiky and the slices tangle.
EYE3 = dict(cx=0.0, cy=0.72, r=0.11, amp=0.26, lobe=0.60)


def field_third_eye(X, Y, c):
    """A five-pointed star raised out of the forehead.

    Drawn as a BUMP IN THE FIELD rather than as an outline over the top, which
    is the whole reason it looks like it belongs. The head is an implicit
    height field sliced into iso-contours, so anything added to the field gets
    sliced by the same knife: the contour lines bend around the star exactly
    the way they bend around the nose and the brow, for free. An outline drawn
    afterwards is a decal, and reads as one - which is the same mistake an
    earlier version of the whole face made, drawing feature curves onto an
    undeformed mesh.

    The star is a radially modulated gaussian. The radius at angle theta is
    r(theta) = r * (1 + lobe*cos(5*(theta - phase))), so five points fall out of
    the cosine rather than being constructed from line segments, and every
    slice of it is smooth.
    """
    a = c["eye3"]
    if a <= 0.0:
        return np.zeros_like(X)
    dx, dy = X - EYE3["cx"], Y - EYE3["cy"]
    d2 = dx * dx + dy * dy
    th = np.arctan2(dy, dx)
    phase = c["eye3_spin"] * (2.0 * np.pi / 5.0)
    r = EYE3["r"] * (1.0 + EYE3["lobe"] * np.cos(5.0 * (th - phase)))
    # r can go small at the concave points; floor it so the exponent stays sane.
    r = np.maximum(r, EYE3["r"] * 0.25)
    return a * EYE3["amp"] * np.exp(-d2 / (r * r))


def field_eyes(X, Y, c):
    """Socket, lid aperture and gaze. eye_open 0 shuts the lids."""
    op, sq = c["eye_open"], c["eye_squint"]
    gx, gy = c["gaze_x"], c["gaze_y"]
    z = np.zeros_like(X)
    for sgn in (-1.0, 1.0):
        cx = sgn * EYE["cx"]
        cy = EYE["cy"]
        # socket
        z = z + EYE["a"] * np.exp(-((X - cx) ** 2) / EYE["rx2"]
                                    - ((Y - cy) ** 2) / EYE["ry2"])
        # eyeball: rises as the lids close, so a shut eye is a smooth dome
        z = z + (0.055 + 0.045 * (1.0 - op)) * np.exp(
            -((X - cx) ** 2) / 0.0130 - ((Y - cy) ** 2) / 0.0075)
        # iris as a shallow dimple, moved by gaze
        z = z - 0.030 * op * np.exp(
            -((X - cx - 0.055 * gx) ** 2) / 0.0022
            - ((Y - cy - 0.035 * gy) ** 2) / 0.0016)
        # lower lid rises with squint
        z = z + 0.040 * sq * np.exp(-((X - cx) ** 2) / 0.0180
                                    - ((Y - cy + 0.075) ** 2) / 0.0022)
    return z


def field_mouth(X, Y, c):
    """Lips. Reuses base v1's lip_field then adds the smile shear."""
    z = lip_field(X, Y, c["mouth_open"], c["mouth_round"])
    sm = c["smile"]
    if abs(sm) > 1e-3:
        # corners lift (or drop) and the whole lip mass shears with them
        for sgn in (-1.0, 1.0):
            z = z + 0.055 * sm * np.exp(-((X - sgn * 0.235) ** 2) / 0.0110
                                        - ((Y + 0.330 - 0.055 * sm) ** 2) / 0.0060)
        # cheek fullness that always comes with a real smile
        for sgn in (-1.0, 1.0):
            z = z + 0.045 * max(sm, 0.0) * np.exp(
                -((X - sgn * 0.360) ** 2) / 0.0300 - ((Y + 0.130) ** 2) / 0.0180)
    return z


def field_nose(X, Y, c):
    fl = c["nose_flare"]
    return 0.030 * fl * np.exp(-((np.abs(X) - 0.115) ** 2) / 0.0022
                               - ((Y + 0.075) ** 2) / 0.0020)


# ---------------------------------------------------------------------------
# REGIONS - box, field function, and which controls disturb it.
# The box is what makes partial re-contouring possible.
# ---------------------------------------------------------------------------
REGIONS = OrderedDict((
    ("brows", dict(box=(-0.52, 0.52, 0.20, 0.60), fn=field_brows,
                   controls=("brow_raise", "brow_furrow"))),
    ("eyes",  dict(box=(-0.52, 0.52, 0.02, 0.40), fn=field_eyes,
                   controls=("eye_open", "eye_squint", "gaze_x", "gaze_y"))),
    ("nose",  dict(box=(-0.26, 0.26, -0.20, 0.06), fn=field_nose,
                   controls=("nose_flare",))),
    ("mouth", dict(box=(-0.46, 0.46, -0.60, -0.08), fn=field_mouth,
                   controls=("mouth_open", "mouth_round", "smile"))),
    # Sits above the brows box (which ends at 0.60) and does not overlap it,
    # so the two never re-contour each other's territory.
    ("eye3",  dict(box=(-0.22, 0.22, 0.58, 0.88), fn=field_third_eye,
                   controls=("eye3", "eye3_spin"))),
))


# ---------------------------------------------------------------------------
# EXPRESSIONS - the whole vocabulary, as data. One line each.
# ---------------------------------------------------------------------------
EXPRESSIONS = {
    "neutral":    {},
    "attentive":  dict(brow_raise=0.45, eye_open=1.0),
    "listening":  dict(brow_raise=0.25, eye_open=1.0, gaze_y=0.10),
    "thinking":   dict(brow_furrow=0.55, eye_squint=0.30, gaze_x=-0.55,
                       gaze_y=0.45, mouth_round=0.25),
    "speaking":   dict(brow_raise=0.15),          # mouth driven by audio
    "happy":      dict(smile=0.85, eye_squint=0.55, brow_raise=0.20),
    "amused":     dict(smile=0.55, eye_squint=0.35, brow_raise=0.35),
    "concerned":  dict(brow_furrow=0.65, smile=-0.35, eye_open=0.90),
    "confused":   dict(brow_furrow=0.40, brow_raise=0.30, mouth_round=0.35,
                       gaze_x=0.35),
    "surprised":  dict(brow_raise=1.0, eye_open=1.0, mouth_open=0.45,
                       mouth_round=0.55),
    "asleep":     dict(eye_open=0.0, brow_raise=0.0, smile=0.08),
}


# ---------------------------------------------------------------------------
# CACHE - one shared memo for every region.
# ---------------------------------------------------------------------------
class _Cache:
    def __init__(self, limit=400):
        self.d = OrderedDict()
        self.limit = limit
        self.hits = self.misses = 0

    def get(self, key, build):
        if key in self.d:
            self.hits += 1
            self.d.move_to_end(key)
            return self.d[key]
        self.misses += 1
        v = build()
        self.d[key] = v
        if len(self.d) > self.limit:
            self.d.popitem(last=False)
        return v


class Region:
    """Owns one box of the face. Everything below here is shared by all
    regions - only the field function above differs. This is the DRY line:
    adding a feature costs a field function and a REGIONS entry, and inherits
    contouring, caching, occlusion and compositing."""

    def __init__(self, name, spec, res=None, depth=1.05, z_step=0.05):
        self.name = name
        self.box = spec["box"]
        self.fn = spec["fn"]
        self.controls = spec["controls"]
        self.depth = depth
        self.z_step = z_step
        x0, x1, y0, y1 = self.box
        # Sample the EXACT SAME POINTS as the static build, not merely the same
        # density. Matching density still put samples at different coordinates,
        # so the region's contours did not meet the surrounding ones and the
        # box outline showed as a rectangle over the face. Slicing the static
        # grid makes them identical where the field is identical, so the seam
        # cannot exist.
        SGX, SGY = C.static_grid()
        ix = np.where((SGX >= x0) & (SGX <= x1))[0]
        iy = np.where((SGY <= y1) & (SGY >= y0))[0]
        self.gx = SGX[ix[0]:ix[-1] + 1]
        self.gy = SGY[iy[0]:iy[-1] + 1]
        self.box = (float(self.gx[0]), float(self.gx[-1]),
                    float(self.gy[-1]), float(self.gy[0]))
        X, Y = np.meshgrid(self.gx, self.gy)
        self.X, self.Y = X, Y
        # everything in this box EXCEPT what this region owns
        self.base = FLD.zfield(X, Y, lips=False) - self.fn(X, Y, DEFAULTS)
        self.mask = np.abs(X) <= FLD.sil_w(Y)
        self.cache = _Cache()

    def is_active(self, controls, tol=1e-3):
        """True only if this region's controls differ from rest.

        At rest the static face is left completely alone, so it renders exactly
        as the plain build - no seam at all. Re-contouring an unchanged region
        can only ever reproduce the static lines approximately, never exactly,
        so the right move is not to do it.
        """
        return any(abs(float(controls.get(c, DEFAULTS[c])) - DEFAULTS[c]) > tol
                   for c in self.controls)

    def key(self, controls):
        """Cache key: this region's controls, on their declared grid.

        Only the controls this region cares about, so a smile does not
        invalidate the brows' cache.
        """
        return quantise(controls, self.controls)

    def geometry(self, controls, base_index):
        k = self.key(controls)
        # Contour at the SNAPPED values, not the raw ones. Otherwise the cached
        # geometry for a key depends on whichever exact controls happened to
        # miss first, and the cache stops being reproducible.
        snapped = dict(controls)
        snapped.update(dequantise(k, self.controls))
        raw = self.cache.get(k, lambda: self._contour(snapped))
        v, e, n = raw
        return v, (e + base_index if len(e) else e), n

    def _contour(self, controls):
        Z = self.base + self.fn(self.X, self.Y, controls)
        Zm = np.where(self.mask, Z, -9.0)
        gzy, gzx = np.gradient(Z, self.gy[1] - self.gy[0], self.gx[1] - self.gx[0])
        ny, nx = Z.shape
        P, N, E = [], [], []
        npts = 0
        lo = math.floor(Zm[self.mask].min() / self.z_step) * self.z_step
        hi = math.ceil(Z[self.mask].max() / self.z_step) * self.z_step
        for lev in np.arange(hi, lo - 1e-9, -self.z_step):
            # permissive thresholds: region contours are short and
            # nearly straight, and the whole-head defaults bin them
            # eps must be SMALLER than the static build's 1.15, not equal to
            # it. approxPolyDP is path-dependent: a short clipped run collapses
            # far harder than the long contour it was cut from, so matching eps
            # made regions supply fewer edges than they punched out - and that
            # deficit WAS the black band. 0.30 was found by sweeping until
            # holecheck.py reported zero holes across every expression and
            # every blend state (worst supply/punch ratio x1.45). Going lower
            # adds edges without closing anything further.
            for poly in C._march(Zm >= float(lev), self.gx, self.gy,
                                 min_raw=3, min_pts=2, eps=0.30):
                ix = np.clip(((poly[:, 0] - self.gx[0])
                              / (self.gx[1] - self.gx[0])).astype(int), 0, nx - 1)
                iy = np.clip(((poly[:, 1] - self.gy[0])
                              / (self.gy[1] - self.gy[0])).astype(int), 0, ny - 1)
                # runs touching the box border belong to the static face
                edge = (ix <= 1) | (ix >= nx - 2) | (iy <= 1) | (iy >= ny - 2)
                keep = np.where(~edge)[0]
                if len(keep) < 2:
                    continue
                for run in np.split(keep, np.where(np.diff(keep) != 1)[0] + 1):
                    if len(run) < 2:      # a 2-point run is a valid segment
                        continue
                    pts = poly[run]
                    nn = np.stack([-gzx[iy[run], ix[run]],
                                   -gzy[iy[run], ix[run]], np.ones(len(run))], 1)
                    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
                    P.append(np.stack([pts[:, 0], pts[:, 1] + 0.20,
                                       np.full(len(run), float(lev) * self.depth)], 1))
                    N.append(nn)
                    k = np.arange(len(run) - 1) + npts
                    E.append(np.stack([k, k + 1], 1))
                    npts += len(run)
        if not P:
            return (np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32),
                    np.zeros((0, 3), np.float32))
        return (np.concatenate(P).astype(np.float32),
                np.concatenate(E).astype(np.int32),
                np.concatenate(N).astype(np.float32))


class Face:
    """The public surface. Everything else is an implementation detail.

        face = Face()
        face.express("thinking")
        face.speak(envelope)             # or face.set(mouth_open=...)
        v, e, n = face.update(t)
    """

    def __init__(self, static=None, verbose=False):
        """`static` is the (verts, edges, normals) the rig composites onto.

        Pass it in. Building it here costs 4.5s, and every caller already has a
        disk-cached copy - the old signature built one anyway and had it thrown
        away immediately, which was the single largest chunk of startup time.
        """
        t0 = time.time()
        # static geometry: skull, ears, neck, back - nothing expressive
        self.static = C.build(lips=True) if static is None else static
        self.regions = OrderedDict(
            (n, Region(n, s)) for n, s in REGIONS.items())
        self._edge_in = {n: self._inside_mask(self.static, r.box)
                         for n, r in self.regions.items()}
        self.controls = dict(DEFAULTS)
        self._target = dict(DEFAULTS)
        self._blend = 0.0
        self._blend_t = 0.30
        self._next_blink = 2.0
        self._blink_t = -1.0
        self.external = set()
        self._from = dict(DEFAULTS)
        self.expression = "neutral"
        if verbose:
            print("Face ready in %.1fs (%d static edges, %d regions)"
                  % (time.time() - t0, len(self.static[1]), len(self.regions)))

    @staticmethod
    def _inside_mask(geom, box):
        """Which static edges fall inside this box (precomputed once)."""
        v, e, _ = geom
        mid = 0.5 * (v[e[:, 0]] + v[e[:, 1]])
        x0, x1, y0, y1 = box
        return ((mid[:, 0] > x0) & (mid[:, 0] < x1)
                & (mid[:, 1] > y0 + 0.20) & (mid[:, 1] < y1 + 0.20))

    @staticmethod
    def _punch(geom, boxes):
        """Kept for callers that want a statically punched copy."""
        v, e, n = geom
        inside = np.zeros(len(e), bool)
        for b in boxes:
            inside |= Face._inside_mask((v, e, n), b)
        return v, e[~inside], n

    # -- control -----------------------------------------------------------
    def set(self, **kw):
        """Drive a control directly. Marks it EXTERNALLY OWNED, so expression
        blending will not fight it - gaze from the camera and mouth from speech
        are inputs, not moods, and must not be dragged back to a preset."""
        for k, val in kw.items():
            if k not in CONTROLS:
                raise KeyError("unknown control %r" % k)
            lo, hi, _, _ = CONTROLS[k]
            self.controls[k] = float(np.clip(val, lo, hi))
            self._target[k] = self.controls[k]
            self.external.add(k)

    def release(self, *names):
        """Hand a control back to expression control."""
        for n in names:
            self.external.discard(n)

    def express(self, name, blend=0.30):
        if name not in EXPRESSIONS:
            raise KeyError("unknown expression %r" % name)
        self._from = dict(self.controls)
        self._target = dict(DEFAULTS)
        self._target.update(EXPRESSIONS[name])
        self._blend = 0.0
        self._blend_t = max(blend, 1e-3)
        self.expression = name

    def speak(self, openness, rounding=0.0):
        """Drive the mouth directly - from a speech envelope or audio RMS."""
        self.controls["mouth_open"] = float(np.clip(openness, 0, 1))
        self.controls["mouth_round"] = float(np.clip(rounding, -1, 1))

    # -- per-frame ---------------------------------------------------------
    def update(self, t, dt=0.033):
        self._tick(t, dt)
        v, e, n = self.static
        active = [nm for nm, r in self.regions.items()
                  if r.is_active(self.controls)]
        if active:
            drop = np.zeros(len(e), bool)
            for nm in active:
                drop |= self._edge_in[nm]
            e = e[~drop]
        vs, es, ns = [v], [e], [n]
        base = len(v)
        for nm in active:
            rv, re, rn = self.regions[nm].geometry(self.controls, base)
            if len(rv):
                vs.append(rv); es.append(re); ns.append(rn)
                base += len(rv)
        return (np.concatenate(vs), np.concatenate(es), np.concatenate(ns))

    def _tick(self, t, dt):
        # expression blend, smoothstepped
        if self._blend < 1.0:
            self._blend = min(1.0, self._blend + dt / self._blend_t)
            s = self._blend * self._blend * (3 - 2 * self._blend)
            for k in CONTROLS:
                if k in self.external:      # sensor/speech driven - hands off
                    continue
                a = self._from.get(k, DEFAULTS[k])
                b = self._target.get(k, DEFAULTS[k])
                self.controls[k] = a + (b - a) * s
        # blink runs independently of expression - it is a reflex, not a mood
        if self._blink_t < 0 and t >= self._next_blink:
            self._blink_t = t
        if self._blink_t >= 0:
            u = (t - self._blink_t) / 0.13
            if u >= 1.0:
                self._blink_t = -1.0
                self._next_blink = t + 2.2 + 4.0 * abs(math.sin(t * 1.7))
            else:
                shut = math.sin(math.pi * u) ** 0.6
                self.controls["eye_open"] = min(
                    self.controls["eye_open"], 1.0 - shut)

    def warm(self, verbose=False):
        """Pre-contour every pose the idle face can reach.

        A cold cache miss costs ~51ms - four frames' worth - so the first time
        a new mouth or eye pose appears the picture visibly hitches. Since the
        controls are quantised, the reachable set during speech and blinking is
        small and finite, so it can simply be enumerated up front.

        Measured: mouth 11 openness x 5 rounding steps, eyes 7 openness steps.
        Roughly 60 poses, a few seconds once, and no runtime spikes afterwards.
        """
        import itertools
        t0 = time.time()
        n = 0
        for name, r in self.regions.items():
            grids = []
            for c in r.controls:
                lo, hi, dflt, steps = CONTROLS[c]
                # speech and blink only drive these; the rest stay at rest
                if c in ("mouth_open", "mouth_round", "eye_open"):
                    grids.append([lo + (hi - lo) * i / (steps - 1)
                                  for i in range(steps)])
                else:
                    grids.append([dflt])
            for combo in itertools.product(*grids):
                ctl = dict(DEFAULTS)
                ctl.update(dict(zip(r.controls, combo)))
                if r.is_active(ctl):
                    r.geometry(ctl, 0)
                    n += 1
        if verbose:
            print("warmed %d poses in %.1fs" % (n, time.time() - t0), flush=True)
        return n

    def stats(self):
        return {n: (r.cache.hits, r.cache.misses)
                for n, r in self.regions.items()}
