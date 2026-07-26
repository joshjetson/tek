"""
Transform, perspective projection, and back-face culling.

Extracted from the old tekhead.py, which was 338 lines of which the runtime
used 17 - the rest was face_1's head model and a mannequin, both superseded.
The name was wrong too: this is projection, not a head.
"""
import math

import numpy as np


def rotate(v, rx, ry, rz):
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return v @ (Rz @ Ry @ Rx).T

def project(v, w, h, dist=4.0, fov=1.35):
    z = v[:, 2] + dist
    z = np.maximum(z, 1e-3)
    f = (min(w, h) * 0.5) * fov
    return np.stack([w * 0.5 + f * v[:, 0] / z,
                     h * 0.5 - f * v[:, 1] / z], axis=1), z

def build_pts_culled(verts, edges, normals, w, h, rot, dist=3.0, eps=-0.10,
                     mode="and", fov=1.35):
    """Project, dropping edges whose both ends face away from the camera.

    Keeping an edge when *either* end faces us preserves the silhouette, which
    is what gives the head its outline. eps slightly negative keeps a little of
    the terminator so the shape does not look sheared off at the edges.
    """
    # (this used to be a function-level `from tekvector import ...`; both
    # functions live in this module now)
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
