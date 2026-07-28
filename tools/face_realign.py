#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-align an existing gallery in place, and prove it helped.

The gallery was enrolled before the recogniser aligned on the eyes, so its
samples sit wherever the Haar box happened to fall. Prediction now aligns, and
mixing aligned probes against unaligned training images is worse than either
consistently - so the stored samples have to be brought forward too.

Landmarks fit on the stored 120x120 crops (12 of 12 here), so this needs
nobody in front of the camera.

Measured before and after by leave-one-out, because "it looks aligned" is not
the claim - the claim is that a genuine match sits further below the threshold
than it did.

    tools/face_realign.py [--apply]
"""
import argparse
import glob
import os
import shutil
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tekdromo import camera, recog


def fit(fm, gray):
    h, w = gray.shape
    try:
        ok, shapes = fm.fit(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                            np.array([[0, 0, w, h]], dtype=np.int32))
    except Exception:
        return None
    if not ok or not len(shapes):
        return None
    return np.array(shapes[0]).reshape(-1, 2).astype(np.float32)


def loo(images, names):
    """Leave-one-out distance for a set of already-prepared images."""
    uniq = sorted(set(names))
    idx = {n: i for i, n in enumerate(uniq)}
    out = []
    for k in range(len(images)):
        tr = [images[j] for j in range(len(images)) if j != k]
        lb = [idx[names[j]] for j in range(len(images)) if j != k]
        if len(set(lb)) < 1 or not tr:
            continue
        m = cv2.face.LBPHFaceRecognizer_create()
        m.train(tr, np.array(lb, dtype=np.int32))
        lab, dist = m.predict(images[k])
        out.append(dist if uniq[lab] == names[k] else 999.0)
    return np.array(out)


def report(tag, d):
    print("   %-22s median %6.1f  p90 %6.1f  max %6.1f   over threshold: %d/%d"
          % (tag, np.median(d), np.percentile(d, 90), d.max(),
             int((d > recog.THRESHOLD).sum()), len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the re-aligned samples (a backup is kept)")
    a = ap.parse_args()

    fm = camera.Tracker._load_facemark()
    if fm is None:
        print("no landmark model")
        return 2

    paths, names, olds, news = [], [], [], []
    for name in recog.people():
        for path in recog.samples(name):
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            pts = fit(fm, g)
            al = recog.align(g, pts) if pts is not None else None
            paths.append(path)
            names.append(name)
            olds.append(recog._norm(g))
            news.append(recog._norm(al) if al is not None else recog._norm(g))
    if not paths:
        print("empty gallery")
        return 2

    aligned = sum(1 for p, n in zip(paths, news) if n is not None)
    print("gallery: %d samples across %s" % (len(paths), sorted(set(names))))
    print("landmarks fitted on %d of %d\n" % (aligned, len(paths)))

    print("-- leave-one-out, before and after --")
    before, after = loo(olds, names), loo(news, names)
    report("as stored", before)
    report("eye-aligned", after)
    better = np.median(after) < np.median(before)
    print("\n   %s" % ("aligned is better - safe to apply" if better else
                       "*** alignment did NOT help; not applying ***"))

    if not a.apply:
        print("\n(dry run - pass --apply to write)")
        return 0
    if not better:
        return 1

    backup = recog.GALLERY.rstrip("/") + ".before-align"
    if not os.path.isdir(backup):
        shutil.copytree(recog.GALLERY, backup)
        print("\nbackup: %s" % backup)
    for path, img in zip(paths, news):
        cv2.imwrite(path, img)
    print("rewrote %d samples" % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
