import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import tekfdl as F

for y in (-0.30, -0.60, -0.90, -1.20, -1.40):
    X = np.array([[0.0, 0.20, 0.30]])
    Y = np.full_like(X, y)
    print("y=%+.2f  neck_mask=%s  neck_surface=%s  zfield=%s"
          % (y, F.neck_mask(X, Y)[0].astype(int),
             np.round(F.neck_surface(X, Y)[0], 3),
             np.round(F.zfield(X, Y, lips=False)[0], 3)))

gx = np.linspace(-1.02, 1.02, 360)
gy = np.linspace(1.18, -1.50, 360)
X, Y = np.meshgrid(gx, gy)
Z = F.zfield(X, Y, lips=False)
M = F.head_mask(X, Y)
band = (Y < -0.90) & M
print("\nrows below y=-0.90 inside mask:", int(band.sum()))
if band.any():
    print("  z range there: %.3f .. %.3f" % (Z[band].min(), Z[band].max()))
print("head_mask total:", int(M.sum()))
nm = F.neck_mask(X, Y)
print("neck_mask total:", int(nm.sum()))
if nm.any():
    print("  z in neck_mask: %.3f .. %.3f" % (Z[nm].min(), Z[nm].max()))
