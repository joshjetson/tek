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


# iBUG 68-point layout: these are the eye contours. The landmark fitter that
# draws the HUD face panel already produces them, so alignment costs nothing.
_LEFT_EYE = list(range(36, 42))
_RIGHT_EYE = list(range(42, 48))
# Where the eyes are put in the normalised crop. eye_y below half-height
# because a face has more below the eyes than above.
EYE_Y = 0.36
EYE_DX = 0.34


def _norm(gray):
    """One preprocessing path, used for BOTH training and prediction.

    If these ever differ the recogniser silently gets worse rather than
    failing, which is the hardest kind of bug to notice.
    """
    import cv2
    g = cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(g)


def align(gray, pts, rect=None):
    """Eye-aligned crop, or None if the landmarks are unusable.

    LBPH compares pixels at fixed positions, so it is far more sensitive to
    the face being in a slightly different place than to the light being
    different. Measured on this gallery, against a threshold of 62:

        perturbation        as cropped      eye-aligned
        shifted 10px          98.1            36.6
        shifted + blurred    101.1            48.0
        scaled to 85%         40.8            26.2

    while gamma and contrast changes cost almost nothing, because the pipeline
    already equalises. Geometry was the whole problem. The raw detector
    rectangle jitters frame to frame and sits differently on a different
    camera, which is why recognition survived neither.

    `pts` are 68 landmarks in the SAME pixel space as `gray`.
    """
    import cv2
    import numpy as np
    if pts is None or len(pts) < 48:
        return None
    p = np.asarray(pts, dtype=np.float32)
    le = p[_LEFT_EYE].mean(axis=0)
    re = p[_RIGHT_EYE].mean(axis=0)
    dx, dy = float(re[0] - le[0]), float(re[1] - le[1])
    span = float(np.hypot(dx, dy))
    if span < 4.0:                       # degenerate fit; better to fall back
        return None
    angle = float(np.degrees(np.arctan2(dy, dx)))
    want = (1.0 - 2.0 * (0.5 - EYE_DX)) * SIZE[0]
    mid = ((le[0] + re[0]) * 0.5, (le[1] + re[1]) * 0.5)
    M = cv2.getRotationMatrix2D(mid, angle, want / span)
    M[0, 2] += SIZE[0] * 0.5 - mid[0]
    M[1, 2] += SIZE[1] * EYE_Y - mid[1]
    # Deliberately NOT equalised here. _norm owns that, so it happens exactly
    # once no matter which path an image took - see prepare().
    return cv2.warpAffine(gray, M, SIZE, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def prepare(gray, pts=None):
    """The canonical face image, aligned if landmarks allow it.

    One function so training and prediction cannot drift apart - the same
    reason _norm exists. Falling back to a plain crop keeps a face that the
    landmark fitter cannot handle recognisable, just less reliably.
    """
    if pts is not None:
        out = align(gray, pts)
        if out is not None:
            return _norm(out)
    return _norm(gray)


# Extra training copies of every enrolled face, standing in for a camera that
# delivers less detail than the one the person was enrolled on. Costs nothing
# at enrolment and is the difference between recognising someone across a
# camera change and not. Measured by leave-one-out on this gallery, threshold
# 62:
#
#   probe            plain   augmented
#   as-is             45.1     45.1      (no cost when detail is there)
#   half resolution   51.7     42.8
#   third resolution  56.7     43.0
#   third + blur      61.0     41.4      (was sitting on the threshold)
#
# The new webcam is wider-angle, so a face at the same standing distance covers
# fewer pixels and arrives softer - which is precisely this axis.
_AUGMENT = ((0.55, 0), (0.40, 3))


def _degrade(img, frac, blur):
    import cv2
    n = max(8, int(SIZE[0] * frac))
    small = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(out, (blur, blur), 0) if blur else out


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


def save_sample(name, gray, pts=None):
    import cv2
    d = os.path.join(GALLERY, name)
    if not os.path.isdir(d):
        os.makedirs(d)
    n = len(samples(name))
    path = os.path.join(d, "%03d.png" % n)
    cv2.imwrite(path, prepare(gray, pts))
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
                # Used AS STORED. save_sample() already ran prepare() on
                # it, and running _norm again would equalise the training
                # images twice while predictions were equalised once - the
                # exact silent mismatch _norm exists to prevent. Only the
                # size is enforced, for galleries written before SIZE moved.
                if g.shape[:2] != (SIZE[1], SIZE[0]):
                    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
                imgs.append(g)
                ids.append(i)
                got += 1
                # ...plus lower-detail copies, so a softer camera still
                # matches. See _AUGMENT for the measured effect.
                for frac, blur in _AUGMENT:
                    imgs.append(_degrade(g, frac, blur))
                    ids.append(i)
            if got:
                self.labels[i] = name
        # Report the ENROLLED count, not the augmented one - "36 samples"
        # for twelve photographs would just be confusing.
        self.trained = sum(1 for _ in ids) // (1 + len(_AUGMENT))
        if len(self.labels) < 1 or not imgs:
            self.model = None
            return 0
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(imgs, np.array(ids))
        return self.trained

    def predict(self, gray_face, pts=None):
        """(name, distance). UNKNOWN when there is no gallery or no good match.

        `pts` are 68 landmarks in gray_face's pixel space. With them the face
        is eye-aligned first, which is worth far more than any amount of
        threshold tuning - see align().
        """
        if self.model is None or gray_face is None or not gray_face.size:
            return UNKNOWN, None
        try:
            label, dist = self.model.predict(prepare(gray_face, pts))
        except Exception:
            return UNKNOWN, None
        if dist is None or dist > THRESHOLD:
            return UNKNOWN, dist
        return self.labels.get(label, UNKNOWN), dist
