"""Per-frame timing distribution and cache behaviour.

Mean fps hides stutter. What matters for "does it look continuous" is the tail:
p95/p99 frame time and how often we blow the budget.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo import app, geometry, phosphor, rig, speech

W, H = 1024, 600
N = 200

v, e, n, _ = app.load_geometry()
face = rig.Face()
face.static = (v, e, n)
face._edge_in = {k: rig.Face._inside_mask(face.static, r.box)
                 for k, r in face.regions.items()}
face.express("neutral", blend=0.01)
face.warm(verbose=True)
statics = phosphor.build_statics(W, H)

stage = {k: [] for k in ("speak", "update", "project", "render", "total")}
miss_at = []
prev_miss = sum(r.cache.misses for r in face.regions.values())

for i in range(N):
    t = i / 30.0
    f0 = time.perf_counter()

    a = time.perf_counter()
    face.speak(*speech.synthetic(t))
    b = time.perf_counter()
    vv, ee, nn = face.update(t, 1 / 30.0)
    c = time.perf_counter()
    pts = geometry.build_pts_culled(vv, ee, nn, W, H, (-0.045, 0.1, 0.0),
                                    16.0, -0.05, "and", 11.4)
    d = time.perf_counter()
    frame = phosphor.render_bgra(pts, W, H, statics)
    e_ = time.perf_counter()

    stage["speak"].append((b - a) * 1000)
    stage["update"].append((c - b) * 1000)
    stage["project"].append((d - c) * 1000)
    stage["render"].append((e_ - d) * 1000)
    stage["total"].append((e_ - f0) * 1000)

    m = sum(r.cache.misses for r in face.regions.values())
    miss_at.append(m - prev_miss)
    prev_miss = m

tot = np.array(stage["total"])
print("FRAME TIME over %d frames (ms)" % N)
print("  mean %.1f   median %.1f   p95 %.1f   p99 %.1f   max %.1f"
      % (tot.mean(), np.median(tot), np.percentile(tot, 95),
         np.percentile(tot, 99), tot.max()))
print("  implied fps: mean %.1f, worst %.1f" % (1000 / tot.mean(), 1000 / tot.max()))
print("  frames over 33ms: %d (%.0f%%)   over 66ms: %d"
      % ((tot > 33).sum(), 100.0 * (tot > 33).mean(), (tot > 66).sum()))

print("\nSTAGE BREAKDOWN (mean ms)")
for k in ("speak", "update", "project", "render"):
    arr = np.array(stage[k])
    print("  %-8s mean %6.1f   p95 %6.1f   max %6.1f"
          % (k, arr.mean(), np.percentile(arr, 95), arr.max()))

miss = np.array(miss_at)
print("\nREGION CACHE")
for nm, r in face.regions.items():
    tot_q = r.cache.hits + r.cache.misses
    print("  %-6s hits %4d  misses %4d  hit rate %.0f%%  entries %d"
          % (nm, r.cache.hits, r.cache.misses,
             100.0 * r.cache.hits / max(tot_q, 1), len(r.cache.d)))
print("  misses/frame: mean %.2f   frames with a miss: %.0f%%"
      % (miss.mean(), 100.0 * (miss > 0).mean()))

slow = tot[miss > 0]
fast = tot[miss == 0]
if len(slow) and len(fast):
    print("\n  frame time WITH a cache miss : %.1f ms" % slow.mean())
    print("  frame time WITHOUT           : %.1f ms" % fast.mean())
    print("  -> a miss costs ~%.0f ms" % (slow.mean() - fast.mean()))
