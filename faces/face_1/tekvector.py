#!/usr/bin/env python3
"""
tekvector - Tektronix 4010/4014 storage-tube vector graphics on a Jetson Nano.

A true vector pipeline: models are vertices + EDGES (never faces). Nothing is
ever filled or shaded, because a Direct-View Storage Tube physically could not
do either - an electron beam walked from point to point and the phosphor held
the charge.

Two authenticity details that most "retro CRT" filters get wrong:

  * No scanlines. A DVST is not a raster display. There is no scan, so there
    are no scanlines. Adding them is the single most common tell of a fake.
  * Constant intensity. Stored charge is binary-ish: a line is either burned
    into the phosphor or it isn't. So no depth-cued brightness by default
    (--depth-cue turns it on anyway, which reads more like a refreshed vector
    display such as an Asteroids cabinet).

The glow is done with multi-scale Gaussian blur on the GPU - which is exactly
the workload the CUDA build wins at (~4x over CPU at this size).
"""
import argparse
import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Phosphor. The 4010's storage tube read as a slightly yellowed green, not the
# pure #00FF00 that "terminal green" usually implies.
# ---------------------------------------------------------------------------
PHOSPHOR_BGR = np.array([0.30, 1.00, 0.45], dtype=np.float32)   # B, G, R
SCREEN_TINT  = np.array([0.010, 0.028, 0.014], dtype=np.float32)


# ---------------------------------------------------------------------------
# Geometry. A model is (verts Nx3, edges Mx2).
# ---------------------------------------------------------------------------
def cube():
    v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                 dtype=np.float32)
    e = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.count_nonzero(v[i] != v[j]) == 1]
    return v * 0.8, np.array(e, dtype=np.int32)


def sphere(n_lat=14, n_lon=24):
    verts, edges, grid = [], [], {}
    for i in range(n_lat + 1):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            grid[(i, j)] = len(verts)
            verts.append((math.sin(theta) * math.cos(phi),
                          math.cos(theta),
                          math.sin(theta) * math.sin(phi)))
    for i in range(n_lat + 1):
        for j in range(n_lon):
            if 0 < i < n_lat:                       # parallels
                edges.append((grid[(i, j)], grid[(i, (j + 1) % n_lon)]))
            if i < n_lat:                           # meridians
                edges.append((grid[(i, j)], grid[(i + 1, j)]))
    return np.array(verts, dtype=np.float32), np.array(edges, dtype=np.int32)


def torus(n_u=24, n_v=14, R=0.75, r=0.3):
    verts, grid = [], {}
    for i in range(n_u):
        u = 2 * math.pi * i / n_u
        for j in range(n_v):
            v = 2 * math.pi * j / n_v
            grid[(i, j)] = len(verts)
            verts.append(((R + r * math.cos(v)) * math.cos(u),
                          r * math.sin(v),
                          (R + r * math.cos(v)) * math.sin(u)))
    edges = []
    for i in range(n_u):
        for j in range(n_v):
            edges.append((grid[(i, j)], grid[((i + 1) % n_u, j)]))
            edges.append((grid[(i, j)], grid[(i, (j + 1) % n_v)]))
    return np.array(verts, dtype=np.float32), np.array(edges, dtype=np.int32)


def apple(n_lon=22):
    """Surface of revolution - exactly how a 70s CAD package would build fruit.

    Profile walks bottom pole -> top pole as (radius, height). What makes it
    read as an apple rather than a sphere is the silhouette, not the mesh:
    widest *above* centre, tapered toward the base, and both ends recessed into
    wells (the profile curves back downward at the top) rather than closing to
    a point.
    """
    profile = [
        (0.000, 0.055), (0.090, 0.015), (0.200, 0.000), (0.340, 0.022),
        (0.460, 0.078), (0.550, 0.165), (0.610, 0.285), (0.638, 0.420),
        (0.632, 0.550), (0.592, 0.662), (0.522, 0.758), (0.424, 0.838),
        (0.303, 0.898), (0.192, 0.936), (0.104, 0.947), (0.048, 0.918),
        (0.016, 0.896), (0.000, 0.890),
    ]
    S = 2.35                                        # overall scale
    verts, edges, grid = [], [], {}
    for i, (r, h) in enumerate(profile):
        y = (h - 0.47) * S
        if r < 1e-6:                                # pole: one shared vertex
            grid[(i, 0)] = len(verts)
            verts.append((0.0, y, 0.0))
            for j in range(1, n_lon):
                grid[(i, j)] = grid[(i, 0)]
            continue
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            grid[(i, j)] = len(verts)
            verts.append((r * math.cos(phi) * S, y, r * math.sin(phi) * S))

    for i in range(len(profile)):
        for j in range(n_lon):
            a = grid[(i, j)]
            b = grid[(i, (j + 1) % n_lon)]
            if a != b:
                edges.append((a, b))                # parallel
            if i < len(profile) - 1:
                c = grid[(i + 1, j)]
                if a != c:
                    edges.append((a, c))            # meridian

    top_y = (0.890 - 0.47) * S

    # Stem, as a short tapered tube so it has actual presence in wireframe.
    stem_path, stem_r = [], []
    for k in range(7):
        t = k / 6.0
        stem_path.append((0.085 * t * t * S, top_y + 0.30 * t * S, -0.02 * t * S))
        stem_r.append((0.040 - 0.016 * t) * S)
    ring, prev = 4, None
    for (px, py, pz), rr in zip(stem_path, stem_r):
        idx = []
        for j in range(ring):
            phi = 2 * math.pi * j / ring
            idx.append(len(verts))
            verts.append((px + rr * math.cos(phi), py, pz + rr * math.sin(phi)))
        for j in range(ring):
            edges.append((idx[j], idx[(j + 1) % ring]))
        if prev:
            edges.extend((prev[j], idx[j]) for j in range(ring))
        prev = idx

    # A leaf, because a bare stem looks like an aerial. Built as a proper
    # closed lens: two arcs sharing a base and tip vertex, plus a midrib.
    base = np.array(stem_path[2], dtype=np.float32)
    d = np.array([0.88, 0.14, -0.45], dtype=np.float32)      # along the leaf
    d /= np.linalg.norm(d)
    s = np.array([0.25, 0.0, 0.80], dtype=np.float32)        # across it
    s -= d * float(np.dot(s, d))
    s /= np.linalg.norm(s)
    up = np.cross(d, s)

    L, W, n = 0.60 * S * 0.5, 0.19 * S * 0.5, 11
    rib, top, bot = [], [], []
    for k in range(n):
        u = k / (n - 1)
        # width tapers to zero at both ends so the outline closes on itself
        w = (math.sin(math.pi * u) ** 0.80) * (1.0 - 0.28 * u)
        c = base + d * (u * L) + up * (0.13 * math.sin(math.pi * u) * S * 0.5)
        rib.append(len(verts)); verts.append(tuple(c))
        if 0 < k < n - 1:
            top.append(len(verts)); verts.append(tuple(c + s * (w * W)))
            bot.append(len(verts)); verts.append(tuple(c - s * (w * W)))

    for k in range(len(rib) - 1):                            # midrib
        edges.append((rib[k], rib[k + 1]))
    for chain in (top, bot):                                 # both outlines,
        edges.append((rib[0], chain[0]))                     # anchored to the
        for k in range(len(chain) - 1):                      # base and tip
            edges.append((chain[k], chain[k + 1]))
        edges.append((chain[-1], rib[-1]))
    for k in range(1, len(top), 3):                          # a few veins
        edges.append((rib[k], top[k - 1]))
        edges.append((rib[k], bot[k - 1]))

    return np.array(verts, dtype=np.float32), np.array(edges, dtype=np.int32)


MODELS = {"cube": cube, "sphere": sphere, "torus": torus, "apple": apple}


# ---------------------------------------------------------------------------
# Transform + projection
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Storage-tube rasteriser
# ---------------------------------------------------------------------------
_gpu_filters = {}


def _blur(img, k):
    """Gaussian blur, CUDA when available. This is the compute-dense op the
    Maxwell GPU actually beats the CPU at."""
    try:
        if k not in _gpu_filters:
            _gpu_filters[k] = cv2.cuda.createGaussianFilter(
                cv2.CV_32FC1, cv2.CV_32FC1, (k, k), 0)
        g = cv2.cuda_GpuMat()
        g.upload(img)
        return _gpu_filters[k].apply(g).download()
    except Exception:
        return cv2.GaussianBlur(img, (k, k), 0)


def draw(segments, w, h, intensities=None, grain=0.012):
    """segments: list of ((x0,y0),(x1,y1)). Returns a BGR uint8 frame."""
    beam = np.zeros((h, w), dtype=np.float32)
    for idx, (p0, p1) in enumerate(segments):
        val = 1.0 if intensities is None else float(intensities[idx])
        cv2.line(beam,
                 (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))),
                 val, 1, cv2.LINE_AA)

    # Multi-scale bloom = the beam spreading in the phosphor.
    glow = (_blur(beam, 5) * 0.55 +
            _blur(beam, 15) * 0.40 +
            _blur(beam, 31) * 0.30)
    inten = beam * 1.15 + glow

    frame = inten[..., None] * PHOSPHOR_BGR[None, None, :]
    # Where the beam saturates, the core burns toward white.
    frame += np.clip(inten - 1.0, 0, None)[..., None] * 0.55
    frame += SCREEN_TINT[None, None, :]

    # Vignette - tube curvature falloff.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - .5) * 2) ** 2 + ((yy / h - .5) * 2) ** 2)
    frame *= np.clip(1.12 - 0.30 * r ** 2, 0, 1)[..., None]

    if grain:
        frame += np.random.normal(0, grain, frame.shape).astype(np.float32)

    # NOTE: deliberately no scanlines. A storage tube does not scan.
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def build_pts(verts, edges, w, h, rot=(0, 0, 0), dist=4.0):
    """Vectorised transform+project straight to the int32 (M,2,2) array that
    cv2.polylines wants. The old per-edge Python loop cost ~7 ms/frame for an
    814-edge model; this is a single fancy-index, ~0.5 ms.

    Constant intensity only - which is the authentic DVST behaviour anyway.
    """
    p2, _ = project(rotate(verts, *rot), w, h, dist)
    return np.rint(p2[edges]).astype(np.int32)


def build_segments(verts, edges, w, h, rot=(0, 0, 0), dist=4.0, depth_cue=False):
    p2, z = project(rotate(verts, *rot), w, h, dist)
    segs, ints = [], []
    for a, b in edges:
        segs.append((p2[a], p2[b]))
        if depth_cue:
            zm = (z[a] + z[b]) * 0.5
            ints.append(np.clip(1.45 - 0.42 * zm / dist, 0.30, 1.30))
        else:
            ints.append(1.0)
    return segs, ints


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="apple", choices=sorted(MODELS))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-W", "--width", type=int, default=1024)
    ap.add_argument("-H", "--height", type=int, default=768)
    ap.add_argument("--rx", type=float, default=-0.30)
    ap.add_argument("--ry", type=float, default=0.62)
    ap.add_argument("--rz", type=float, default=0.0)
    ap.add_argument("--dist", type=float, default=4.0)
    ap.add_argument("--depth-cue", action="store_true",
                    help="vary intensity with depth (refreshed-vector look, "
                         "not authentic DVST)")
    ap.add_argument("--animate", metavar="MP4",
                    help="render the beam plotting the model, then rotating")
    ap.add_argument("--frames", type=int, default=180)
    a = ap.parse_args()

    verts, edges = MODELS[a.model]()
    print(f"{a.model}: {len(verts)} vertices, {len(edges)} edges")

    if a.animate:
        vw = cv2.VideoWriter(a.animate, cv2.VideoWriter_fourcc(*"mp4v"),
                             24, (a.width, a.height))
        plot_frames = a.frames // 3
        for i in range(a.frames):
            if i < plot_frames:
                # Phase 1: watch the beam lay the drawing down, stroke by
                # stroke - the thing you actually sat and watched in 1974.
                rot = (a.rx, a.ry, a.rz)
                n = int(len(edges) * (i + 1) / plot_frames)
                segs, ints = build_segments(verts, edges[:n], a.width, a.height,
                                            rot, a.dist, a.depth_cue)
            else:
                t = (i - plot_frames) / max(1, a.frames - plot_frames)
                rot = (a.rx, a.ry + 2 * math.pi * t, a.rz)
                segs, ints = build_segments(verts, edges, a.width, a.height,
                                            rot, a.dist, a.depth_cue)
            vw.write(draw(segs, a.width, a.height, ints))
            if i % 20 == 0:
                print(f"  frame {i}/{a.frames}")
        vw.release()
        print("wrote", a.animate)
        return

    segs, ints = build_segments(verts, edges, a.width, a.height,
                                (a.rx, a.ry, a.rz), a.dist, a.depth_cue)
    out = a.out or f"{a.model}.png"
    cv2.imwrite(out, draw(segs, a.width, a.height, ints))
    print("wrote", out)


if __name__ == "__main__":
    main()
