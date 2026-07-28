#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why has recognition started saying UNKNOWN?

The gallery was enrolled on the old webcam. The new one is a different sensor
with a wider field of view, so at the same standing distance a face covers
fewer pixels, gets upscaled further to reach the 120x120 the recogniser wants,
and arrives blurrier and with different tone. LBPH is not magic about that.

Nobody has to be in front of the camera for this. Three things are measured:

  * LEAVE-ONE-OUT on the existing gallery - is the model sound at all, and
    what does a genuine match cost in LBPH distance?
  * SIMULATED DOMAIN SHIFT - the same faces put through the transformations a
    different camera applies (resolution loss, gamma, contrast, noise), to see
    which one actually pushes distance past the threshold.
  * WHAT THE CAMERA IS DOING NOW - frame size, detect scale, and therefore how
    many pixels a face at a normal distance would actually get.

    tools/face_diag.py
"""
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

import cv2
import numpy as np

from tekdromo import recog


def load_gallery():
    people = {}
    for d in sorted(glob.glob(os.path.join(recog.GALLERY, "*"))):
        if not os.path.isdir(d):
            continue
        imgs = []
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(im)
        if imgs:
            people[os.path.basename(d)] = imgs
    return people


def train(samples, labels):
    m = cv2.face.LBPHFaceRecognizer_create()
    m.train([recog._norm(s) for s in samples], np.array(labels, dtype=np.int32))
    return m


def leave_one_out(people):
    print("-- leave-one-out: train on the rest, test the one held back --")
    names = sorted(people)
    worst = 0.0
    for name in names:
        imgs = people[name]
        dists = []
        for i in range(len(imgs)):
            train_imgs, train_lbl = [], []
            for j, other in enumerate(names):
                for k, im in enumerate(people[other]):
                    if other == name and k == i:
                        continue
                    train_imgs.append(im)
                    train_lbl.append(j)
            m = train(train_imgs, train_lbl)
            lbl, dist = m.predict(recog._norm(imgs[i]))
            ok = names[lbl] == name
            dists.append(dist if ok else 999.0)
        d = np.array(dists)
        worst = max(worst, float(np.percentile(d, 90)))
        print("   %-10s n=%-3d median %.1f  p90 %.1f  max %.1f   (threshold %.0f)"
              % (name, len(imgs), np.median(d), np.percentile(d, 90), d.max(),
                 recog.THRESHOLD))
    return worst


def shift(img, kind):
    """Approximate what a different camera does to the same face."""
    h, w = img.shape
    if kind == "half resolution":
        small = cv2.resize(img, (max(8, w // 2), max(8, h // 2)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "third resolution":
        small = cv2.resize(img, (max(8, w // 3), max(8, h // 3)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "brighter (gamma 0.6)":
        return np.clip(((img / 255.0) ** 0.6) * 255.0, 0, 255).astype(np.uint8)
    if kind == "darker (gamma 1.7)":
        return np.clip(((img / 255.0) ** 1.7) * 255.0, 0, 255).astype(np.uint8)
    if kind == "low contrast":
        return np.clip(img * 0.55 + 60, 0, 255).astype(np.uint8)
    if kind == "sensor noise":
        n = np.random.RandomState(0).normal(0, 12, img.shape)
        return np.clip(img + n, 0, 255).astype(np.uint8)
    if kind == "slight blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if kind == "shifted 8px":
        M = np.float32([[1, 0, 8], [0, 1, 6]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def domain_shift(people):
    print("\n-- the same faces, through what a different camera does --")
    names = sorted(people)
    imgs, lbl = [], []
    for j, n in enumerate(names):
        for im in people[n]:
            imgs.append(im)
            lbl.append(j)
    m = train(imgs, lbl)
    print("   %-24s %8s %8s  %s" % ("transformation", "median", "p90", "verdict"))
    for kind in ("none", "half resolution", "third resolution", "slight blur",
                 "brighter (gamma 0.6)", "darker (gamma 1.7)", "low contrast",
                 "sensor noise", "shifted 8px"):
        ds = []
        for j, n in enumerate(names):
            for im in people[n]:
                p, d = m.predict(recog._norm(shift(im, kind)))
                ds.append(d if p == j else 999.0)
        ds = np.array(ds)
        med, p90 = float(np.median(ds)), float(np.percentile(ds, 90))
        verdict = ("fine" if p90 < recog.THRESHOLD * 0.75 else
                   "marginal" if p90 < recog.THRESHOLD else
                   "*** would read as UNKNOWN ***")
        print("   %-24s %8.1f %8.1f  %s" % (kind, med, p90, verdict))


def camera_now():
    print("\n-- what the camera is actually delivering --")
    from tekdromo import camera
    devs = camera.video_devices()
    print("   devices: %s" % devs)
    if not devs:
        return
    cap = cv2.VideoCapture(devs[0], cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = False, None
        for _ in range(20):
            ok, frame = cap.read()
            if ok:
                break
        if not ok:
            print("   (no frame - the display service is probably holding it)")
            return
        print("   frame: %s" % (frame.shape,))
        casc = cv2.CascadeClassifier(camera.CASCADE_FAST)
        if casc.empty():
            casc = cv2.CascadeClassifier(camera.CASCADE)
        small = cv2.resize(frame, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        g = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = casc.detectMultiScale(g, 1.2, 5, minSize=(40, 40))
        print("   detect frame: %s   faces found: %d" % (g.shape, len(faces)))
        for (x, y, w, h) in faces:
            print("      face %dx%d px  -> upscaled x%.1f to reach %dx%d"
                  % (w, h, recog.SIZE[0] / float(w), recog.SIZE[0],
                     recog.SIZE[1]))
        if not len(faces):
            print("      (nobody in view, which is expected while the house is "
                  "empty - the numbers above still stand)")
    finally:
        cap.release()


def main():
    people = load_gallery()
    if not people:
        print("no gallery at %s" % recog.GALLERY)
        return 2
    print("gallery: %s\n" % {k: len(v) for k, v in people.items()})
    leave_one_out(people)
    domain_shift(people)
    camera_now()
    return 0


if __name__ == "__main__":
    sys.exit(main())
