import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import cv2
import tekrun
import tekrig
import tekhead
import tekvector as tv

W, H = 1024, 600
POSE = (-0.045, 0.0, 0.0)


def render(v, e, n):
    pts = tekhead.build_pts_culled(v, e, n, W, H, POSE, 16.0, -0.05, "and", 11.4)
    return tv.draw([(p[0], p[1]) for p in pts], W, H)


def lit(i):
    return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) > 40


v, e, n, _ = tekrun.load_geometry()
face = tekrig.Face()
face.static = (v, e, n)
face._edge_in = {nm: tekrig.Face._inside_mask(face.static, r.box)
                 for nm, r in face.regions.items()}

imgs = []
for label, ctl in (("rest", {}), ("brow_raise=0.45", dict(brow_raise=0.45))):
    face.controls = dict(tekrig.DEFAULTS)
    face.controls.update(ctl)
    face._blend = 1.0
    face._target = dict(face.controls)
    act = [nm for nm, r in face.regions.items() if r.is_active(face.controls)]
    vv, ee, nn = face.update(0.0, 0.0)
    img = render(vv, ee, nn)
    print("%-16s active=%-10s total edges=%4d  lit=%d"
          % (label, ",".join(act) or "none", len(ee), lit(img).sum()))
    for nm in act:
        r = face.regions[nm]
        rv, re, rn = r.geometry(face.controls, 0)
        dropped = int(face._edge_in[nm].sum())
        print("     %-6s punched %3d static edges, supplied %3d"
              % (nm, dropped, len(re)))
    cv2.putText(img, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (90, 240, 130), 2, cv2.LINE_AA)
    imgs.append(img)

cv2.imwrite("/home/super/tek_out/browtest.png", np.hstack(imgs))
print("wrote browtest.png")
