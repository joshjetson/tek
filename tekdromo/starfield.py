"""
Amber star backdrop.

Same technique as the face: real vector strokes, run through the same bloom and
the same phosphor LUT - just with the amber colour instead of green. Nothing
here is a sprite or a blitted dot; every star is drawn with cv2.polylines
exactly like a contour line, so the two layers cannot drift apart in look.

Two decisions worth knowing:

* The stars are STATIC. The camera never moves - only the head rotates - so a
  distant backdrop has no parallax to show. That means the whole layer can be
  rendered once at startup and composited each frame, which costs one pass
  instead of a second full render pipeline (~20ms saved per frame).

* Distance is conveyed by SIZE and BRIGHTNESS, not by perspective. Faint
  single-pixel ticks read as far away; a handful of brighter four-armed stars
  give the eye something to fix on. Real vector displays drew stars exactly
  this way because a point is not a thing a beam can draw - it has to be a
  short stroke.
"""
import numpy as np
import cv2

from . import phosphor

# Field geometry. Stars are placed in normalised screen space rather than 3D:
# with a fixed camera and no parallax, projecting them would produce the same
# picture at more cost.
N_STARS = 340
SEED = 0x7EC
TWINKLE_PHASES = 6          # pre-rendered variants, cycled slowly
CLEAR_RADIUS = 0.30         # keep the densest stars off the head's silhouette


def _positions(w, h, rng):
    """Star positions, magnitudes and a twinkle group for each."""
    x = rng.uniform(-1.0, 1.0, N_STARS)
    y = rng.uniform(-1.0, 1.0, N_STARS)
    # Thin the field near the middle so the backdrop does not fight the face.
    r = np.sqrt((x * 0.62) ** 2 + y ** 2)
    keep = rng.uniform(0, 1, N_STARS) < np.clip(r / CLEAR_RADIUS, 0.12, 1.0)
    x, y = x[keep], y[keep]
    # Magnitude: mostly faint, a few bright. A uniform field looks synthetic.
    mag = rng.power(0.45, len(x))
    grp = rng.randint(0, TWINKLE_PHASES, len(x))
    px = ((x * 0.5 + 0.5) * w).astype(np.int32)
    py = ((y * 0.5 + 0.5) * h).astype(np.int32)
    return px, py, mag, grp


def _draw(beam, px, py, mag, grp, phase):
    """Stroke every star. Brighter stars get more arms - the same way a vector
    terminal would have drawn them."""
    for i in range(len(px)):
        x, y, m = int(px[i]), int(py[i]), float(mag[i])
        # slow twinkle: one group at a time dips
        if grp[i] == phase:
            m *= 0.55
        v = int(np.clip(28 + 150 * m, 0, 255))
        if m > 0.80:                      # bright: four-armed star
            a = 3
            cv2.line(beam, (x - a, y), (x + a, y), v, 1, cv2.LINE_AA)
            cv2.line(beam, (x, y - a), (x, y + a), v, 1, cv2.LINE_AA)
            cv2.line(beam, (x - 2, y - 2), (x + 2, y + 2), v // 2, 1, cv2.LINE_AA)
            cv2.line(beam, (x - 2, y + 2), (x + 2, y - 2), v // 2, 1, cv2.LINE_AA)
        elif m > 0.45:                    # medium: a small plus
            cv2.line(beam, (x - 1, y), (x + 1, y), v, 1, cv2.LINE_AA)
            cv2.line(beam, (x, y - 1), (x, y + 1), v, 1, cv2.LINE_AA)
        else:                             # distant: a single tick
            cv2.line(beam, (x, y), (x, y), v, 1, cv2.LINE_AA)


def build(w, h, seed=SEED):
    """Pre-render every twinkle phase as a finished BGRA layer.

    Costs a few hundred ms once. Compositing one of these per frame is a single
    full-frame pass; running the star layer through the render pipeline live
    would cost as much again as the face does.
    """
    rng = np.random.RandomState(seed)
    px, py, mag, grp = _positions(w, h, rng)
    lut = phosphor.make_lut(phosphor.AMBER_BGR, tint=np.zeros(3, np.float32))
    hw, hh = w // 2, h // 2
    frames = []
    for phase in range(TWINKLE_PHASES):
        beam = np.zeros((h, w), np.uint8)
        _draw(beam, px, py, mag, grp, phase)
        # Same bloom shape as the face, but tighter: a distant point source
        # should not smear as much as a near one.
        small = cv2.resize(beam, (hw, hh), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(small, (7, 7), 0),
                          (w, h), interpolation=cv2.INTER_LINEAR)
        inten = cv2.addWeighted(beam, 1.0, glow, 0.75, 0)
        frames.append(cv2.LUT(cv2.cvtColor(inten, cv2.COLOR_GRAY2BGRA), lut))
    return frames


class Backdrop:
    """Holds the pre-rendered layers and composites the current one."""

    def __init__(self, w, h, seed=SEED, period=1.9):
        self.frames = build(w, h, seed)
        self.period = period

    def under(self, frame, t):
        """Composite the face OVER the stars.

        cv2.max, not add: the face's LUT already contributes the screen tint, so
        adding would double it and lift the black level. max() also keeps a
        bright face line from being pushed to white by a star behind it.
        """
        stars = self.frames[int(t / self.period) % len(self.frames)]
        return cv2.max(frame, stars)
