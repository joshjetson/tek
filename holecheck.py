import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np
import tekrun
import tekrig

v, e, n, _ = tekrun.load_geometry()
face = tekrig.Face()
face.static = (v, e, n)
face._edge_in = {nm: tekrig.Face._inside_mask(face.static, r.box)
                 for nm, r in face.regions.items()}

print("%-11s %-8s %8s %9s %s" % ("expression", "region", "punched", "supplied", ""))
worst = 1e9
for name in sorted(tekrig.EXPRESSIONS):
    face.controls = dict(tekrig.DEFAULTS)
    face.controls.update(tekrig.EXPRESSIONS[name])
    face._blend = 1.0
    for nm, r in face.regions.items():
        if not r.is_active(face.controls):
            continue
        _, re, _ = r.geometry(face.controls, 0)
        p = int(face._edge_in[nm].sum())
        ratio = len(re) / max(p, 1)
        worst = min(worst, ratio)
        flag = "  <-- HOLE" if len(re) < p else ""
        print("%-11s %-8s %8d %9d   x%.2f%s" % (name, nm, p, len(re), ratio, flag))

# also sweep the blend path between neutral and each expression, since a
# half-blended state is a different control vector and could still hole
print("\nsweeping blend paths...")
bad = 0
for name in sorted(tekrig.EXPRESSIONS):
    tgt = dict(tekrig.DEFAULTS)
    tgt.update(tekrig.EXPRESSIONS[name])
    for s in (0.15, 0.35, 0.55, 0.75, 0.95):
        face.controls = {k: tekrig.DEFAULTS[k] + (tgt[k] - tekrig.DEFAULTS[k]) * s
                         for k in tekrig.CONTROLS}
        for nm, r in face.regions.items():
            if not r.is_active(face.controls):
                continue
            _, re, _ = r.geometry(face.controls, 0)
            p = int(face._edge_in[nm].sum())
            if len(re) < p:
                print("  HOLE: %s @%.2f  %s  %d < %d" % (name, s, nm, len(re), p))
                bad += 1
            worst = min(worst, len(re) / max(p, 1))
print("\nworst supplied/punched ratio anywhere: x%.2f   holes found: %d"
      % (worst, bad))
