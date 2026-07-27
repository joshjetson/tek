"""
Who is that? Local face recognition, no network.

LBPH rather than an embedding model: it is in the OpenCV build already, trains
in milliseconds from a handful of photographs, predicts in about one, and needs
no 100 MB of weights. It is not state of the art and does not need to be - the
question here is "which of the three or four people who live here is this",
not "which of a million".

The gallery is plain PNG files under ~/.config/tekdromo/faces/<name>/, so
enrolling someone is copying in a few pictures and forgetting a person is
deleting a directory. Nothing is hidden in a binary blob.

Confidence in LBPH is a DISTANCE - lower is better, and it is unbounded. A
match is only reported below a threshold, because the recogniser will always
name its nearest neighbour however wrong that is, and a face panel confidently
labelling a stranger with your daughter's name is worse than one that says
UNKNOWN.
"""
import os

import numpy as np

GALLERY = os.path.expanduser("~/.config/tekdromo/faces")
# Kept OUTSIDE the gallery directory on purpose. Writing it inside would change
# the gallery's mtime every time someone is seen, and the tracker watches that
# mtime to notice enrolments - so every sighting would trigger a retrain.
REGISTRY = os.path.expanduser("~/.config/tekdromo/people.json")
SIZE = (120, 120)
# Below this LBPH distance we believe it. Tuned conservatively: the cost of a
# wrong name is much higher than the cost of "UNKNOWN".
THRESHOLD = 62.0
UNKNOWN = "UNKNOWN"


def _norm(gray):
    """One preprocessing path, used for BOTH training and prediction.

    If these ever differ the recogniser silently gets worse rather than
    failing, which is the hardest kind of bug to notice.
    """
    import cv2
    g = cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(g)


def registry():
    """Everyone known, with when they were enrolled and last seen.

    A record rather than just a pile of photographs: who is known, since when,
    how often they have been seen and when last. That is the difference between
    a novelty and something you could answer a question with.
    """
    try:
        import json
        with open(REGISTRY) as f:
            return json.load(f)
    except Exception:
        return {}


def write_registry(reg):
    import json
    d = os.path.dirname(REGISTRY)
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
    os.replace(tmp, REGISTRY)          # atomic: never a half-written record


def note_enrolled(name, samples):
    import time as _t
    reg = registry()
    e = reg.setdefault(name, {})
    e.setdefault("enrolled", _t.strftime("%Y-%m-%d %H:%M"))
    e["samples"] = samples
    write_registry(reg)
    return e


def note_seen(name, when=None):
    """Record a sighting. Callers must throttle - this writes a file."""
    import time as _t
    if not name or name == UNKNOWN:
        return
    reg = registry()
    e = reg.setdefault(name, {})
    e["last_seen"] = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(when))
    e["seen"] = int(e.get("seen", 0)) + 1
    write_registry(reg)


def forget(name):
    import shutil
    shutil.rmtree(os.path.join(GALLERY, name), ignore_errors=True)
    reg = registry()
    reg.pop(name, None)
    write_registry(reg)


def signature():
    """A value that changes whenever the gallery changes.

    The mtime of the GALLERY directory alone is not enough: adding photographs
    to faces/JOSH/ does not touch faces/, which only changes when a
    subdirectory appears. Watching just the parent meant an enrolment was
    noticed once, at whatever count existed the moment the person's directory
    was created, and every sample added afterwards was ignored.
    """
    try:
        best = os.path.getmtime(GALLERY)
    except OSError:
        return None
    for name in people():
        try:
            best = max(best, os.path.getmtime(os.path.join(GALLERY, name)))
        except OSError:
            pass
    return best


def people():
    try:
        return sorted(d for d in os.listdir(GALLERY)
                      if os.path.isdir(os.path.join(GALLERY, d)))
    except OSError:
        return []


def samples(name):
    d = os.path.join(GALLERY, name)
    try:
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith(".png"))
    except OSError:
        return []


def save_sample(name, gray):
    import cv2
    d = os.path.join(GALLERY, name)
    if not os.path.isdir(d):
        os.makedirs(d)
    n = len(samples(name))
    path = os.path.join(d, "%03d.png" % n)
    cv2.imwrite(path, _norm(gray))
    return path


class Recogniser(object):
    """Names a face crop, or returns UNKNOWN."""

    def __init__(self):
        self.model = None
        self.labels = {}
        self.trained = 0
        self.reload()

    def reload(self):
        """Train from whatever is in the gallery. Cheap enough to just redo."""
        import cv2
        imgs, ids = [], []
        self.labels = {}
        for i, name in enumerate(people()):
            got = 0
            for path in samples(name):
                g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                imgs.append(_norm(g))
                ids.append(i)
                got += 1
            if got:
                self.labels[i] = name
        self.trained = len(imgs)
        if len(self.labels) < 1 or not imgs:
            self.model = None
            return 0
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(imgs, np.array(ids))
        return self.trained

    def predict(self, gray_face):
        """(name, distance). UNKNOWN when there is no gallery or no good match."""
        if self.model is None or gray_face is None or not gray_face.size:
            return UNKNOWN, None
        try:
            label, dist = self.model.predict(_norm(gray_face))
        except Exception:
            return UNKNOWN, None
        if dist is None or dist > THRESHOLD:
            return UNKNOWN, dist
        return self.labels.get(label, UNKNOWN), dist
