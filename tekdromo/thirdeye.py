# -*- coding: utf-8 -*-
"""
A star on the forehead, spinning while the channel is open.

The radio protocol makes turn-taking explicit for the EAR. This makes it
explicit for the person: on a half-duplex channel both ends have to agree
whose turn it is, and until now the only feedback that TEK was listening was
that it eventually answered. Silence meant "thinking", "did not hear you", and
"channel closed" all at once.

Speaking already has an animation - the mouth moves, driven by the same PCM
reaching the speaker - so this is the other half of the same idea. Spinning
means the channel is open and it is your turn.

It is drawn in MODEL space and projected with the head's own transform, not
placed on screen. A mark that stayed put while the head turned would read as a
HUD element floating in front of the face rather than as part of it, and the
whole point is that it belongs to the face.
"""
import math

import numpy as np

# Where it sits, in the head's own coordinates. Above the brow, on the midline,
# and slightly proud of the surface so it is never swallowed by a contour line
# passing through the same place.
# y was swept against the rendered head rather than guessed: 0.62 put the star
# on the bridge of the nose, 1.10 pushed it over the crown, and 0.95 sits on
# the forehead centred above the brow.
POS = (0.0, 0.95, 0.46)

# Radius of the star, in the same units. Big enough to read across a room at
# 1024x600, small enough not to compete with the face it sits on.
R_OUT = 0.11
R_IN = 0.045

POINTS = 5

# Revolutions per second. Slow: a fast spin reads as an alarm, and this is
# meant to say "go ahead", not "something is wrong".
SPIN_HZ = 0.22

# The twinkle. Amplitude is applied to the star's SIZE rather than its
# brightness, because a storage tube has constant beam intensity - there is no
# dimming on a real 4014, and faking it with fewer segments is the classic
# tell. Scaling is a thing a vector display can genuinely do.
TWINKLE_HZ = 1.6
TWINKLE = 0.22          # +/- this fraction of the radius


def _star(t):
    """The star as (N, 3) model-space points, ready to be rotated by the head."""
    spin = 2.0 * math.pi * SPIN_HZ * t
    scale = 1.0 + TWINKLE * math.sin(2.0 * math.pi * TWINKLE_HZ * t)
    pts = []
    for i in range(POINTS * 2):
        a = spin + i * math.pi / POINTS
        r = (R_OUT if i % 2 == 0 else R_IN) * scale
        pts.append((POS[0] + r * math.cos(a),
                    POS[1] + r * math.sin(a),
                    POS[2]))
    return np.array(pts, dtype=np.float32)


def segments(t, w, h, rot, dist, fov, geometry):
    """(N, 2, 2) int32 segments, in the same form the head and HUD emit.

    Takes `geometry` rather than importing it, so this module stays free of the
    render stack and can be unit-tested on its own.
    """
    v = _star(t)
    n = len(v)
    # A closed outline: each point to the next, last back to first.
    edges = np.array([(i, (i + 1) % n) for i in range(n)], dtype=np.int32)
    # Facing straight out of the forehead. Culling would otherwise drop the
    # whole star the moment the head turns, which is exactly when it is most
    # useful to see.
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
    return geometry.build_pts_culled(v, edges, normals, w, h, rot,
                                     dist=dist, eps=-1.0, mode="or", fov=fov)
