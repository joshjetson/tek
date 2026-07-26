import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import tekrig

face = tekrig.Face(verbose=True)
print("static edges after punch:", len(face.static[1]))
for name, r in face.regions.items():
    v, e, n = r.geometry(face.controls, 0)
    x0, x1, y0, y1 = r.box
    Z = r.base + r.fn(r.X, r.Y, face.controls)
    print("  %-6s box y %.2f..%.2f  grid %s  z %.3f..%.3f  -> %4d edges"
          % (name, y0, y1, Z.shape, Z[r.mask].min(), Z[r.mask].max(), len(e)))
    print("         mask covers %.0f%% of box" % (100.0 * r.mask.mean()))
