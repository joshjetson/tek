#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Would aligning on the eyes fix recognition? Measure it before building it.

face_diag.py showed the dominant error is not lighting - gamma and contrast
barely move the distance, because the pipeline already equalises - it is
GEOMETRY. An 8-pixel shift of the crop costs 78-84 LBPH distance against a
threshold of 62, and the crop is the raw Haar rectangle, which jitters from
frame to frame and sits differently on a different camera.

The head already fits 68 landmarks for the HUD face panel, so the eyes are
known for free. This checks the actual claim before any of the recogniser is
touched: does aligning on eye centres remove the penalty a shift imposes?

    tools/face_align_test.py
"""
import glob
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tekdromo import camera, recog

# iBUG 68: 36-41 left eye, 42-47 right eye.
LEFT, RIGHT = list(range(36, 42)), list(range(42, 48))


def fit(fm, gray):
    """Landmarks for the single face filling this crop, or None."""
    h, w = gray.shape
    ok, shapes = fm.fit(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                        np.array([[0, 0, w, h]], dtype=np.int32))
    if not ok or not len(shapes):
        return None
    return shapes[0][0]


def align(gray, pts, size=recog.SIZE, eye_y=0.36, eye_dx=0.34):
    """Rotate/scale so the eyes land on fixed points. The standard fix."""
    le = pts[LEFT].mean(axis=0)
    re = pts[RIGHT].mean(axis=0)
    dy, dx = (re[1] - le[1]), (re[0] - le[0])
    angle = np.degrees(np.arctan2(dy, dx))
    dist = np.hypot(dx, dy)
    want = (1.0 - 2.0 * (0.5 - eye_dx)) * size[0]
    scale = (want / dist) if dist > 1 else 1.0
    mid = ((le[0] + re[0]) * 0.5, (le[1] + re[1]) * 0.5)
    M = cv2.getRotationMatrix2D(mid, angle, scale)
    M[0, 2] += size[0] * 0.5 - mid[0]
    M[1, 2] += size[1] * eye_y - mid[1]
    return cv2.warpAffine(gray, M, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def shift(img, dx, dy, blur=0):
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if blur:
        out = cv2.GaussianBlur(out, (blur, blur), 0)
    return out


def main():
    fm = camera.Tracker._load_facemark()
    if fm is None:
        print("no landmark model at %s" % camera.LANDMARK_MODEL)
        return 2

    imgs = []
    for f in sorted(glob.glob(os.path.join(recog.GALLERY, "*", "*"))):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            imgs.append(im)
    if len(imgs) < 4:
        print("need a gallery to test against")
        return 2
    print("using %d gallery faces\n" % len(imgs))

    # Landmarks for every gallery face, once.
    pts = [fit(fm, im) for im in imgs]
    have = [i for i, p in enumerate(pts) if p is not None]
    print("landmarks fitted on %d of %d\n" % (len(have), len(imgs)))
    if len(have) < 4:
        print("cannot fit landmarks on the stored crops - alignment is not "
              "testable this way")
        return 1

    lbl = np.array([0] * len(have), dtype=np.int32)

    def dist_under(prep, perturb):
        """Median/p90 distance when the probe is perturbed but the gallery is
        not - which is exactly the situation a new camera creates."""
        gal = [prep(imgs[i], pts[i]) for i in have]
        m = cv2.face.LBPHFaceRecognizer_create()
        m.train(gal, lbl)
        ds = []
        for i in have:
            moved = perturb(imgs[i])
            p = fit(fm, moved)
            if p is None:
                ds.append(999.0)
                continue
            ds.append(m.predict(prep(moved, p))[1])
        a = np.array(ds)
        return float(np.median(a)), float(np.percentile(a, 90))

    def plain(img, _p):
        return recog._norm(img)

    def aligned(img, p):
        return cv2.equalizeHist(align(img, p))

    cases = [
        ("no change", lambda im: im),
        ("shifted 6px", lambda im: shift(im, 6, 4)),
        ("shifted 10px", lambda im: shift(im, 10, 8)),
        ("shifted + blurred", lambda im: shift(im, 8, 6, blur=5)),
        ("scaled 85%", lambda im: cv2.resize(
            cv2.resize(im, (102, 102), interpolation=cv2.INTER_AREA),
            recog.SIZE, interpolation=cv2.INTER_LINEAR)),
    ]

    print("%-20s %19s %19s" % ("", "CROP AS NOW", "EYE-ALIGNED"))
    print("%-20s %9s %9s %9s %9s" % ("perturbation", "median", "p90",
                                     "median", "p90"))
    print("-" * 62)
    for name, fn in cases:
        pm, pp = dist_under(plain, fn)
        am, ap = dist_under(aligned, fn)
        print("%-20s %9.1f %9.1f %9.1f %9.1f%s"
              % (name, pm, pp, am, ap,
                 "   <- alignment wins" if ap < pp - 5 else ""))
    print("\nthreshold is %.0f" % recog.THRESHOLD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
