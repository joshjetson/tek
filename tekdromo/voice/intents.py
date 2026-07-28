# -*- coding: utf-8 -*-
"""
Spoken commands that TEK carries out itself, instead of asking a model.

"Register my face" is a thing to DO, not a thing to have an opinion about.
Sending it to the brain costs ~7.5 s and an API call to be told, at best, that
it cannot reach the camera - so intents are matched first and the model never
sees them.

Two rules keep this from swallowing ordinary conversation:

* **Match narrowly.** Every pattern requires both a verb and the object, so
  "how does face recognition work" is a question and reaches the brain, while
  "register my face" is a command and does not. When in doubt it is not an
  intent - a question answered wrongly is a small annoyance, a command that
  silently did nothing is a broken device.
* **Never match on the wake word alone.** The transcript arrives from a small
  recogniser in a room and is frequently wrong; the more words a pattern needs,
  the less often noise can satisfy it by accident.
"""
import os
import re
import time

# How long a follow-up answer is accepted for. The ear already re-arms for
# FOLLOWUP_S after speaking, so this only has to outlast the question being
# read aloud plus a person thinking about it.
PENDING_S = 30.0

# Samples to take when enrolling by voice. Fewer than the CLI's ten: the
# display writes a face crop every 1.5 s, so ten is a fifteen-second silence
# with somebody standing in front of a camera wondering if it has crashed.
# Eight is about twelve seconds, and recognition quality is dominated by
# alignment and augmentation rather than by sample count (README section 9).
VOICE_SAMPLES = 8

CROP = os.path.expanduser("~/.cache/tekdromo/crop.png")

# The verb has to be there, AND the object. "my face"/"me" is the object;
# recognise/register/remember/learn/enrol is the verb.
_ENROL = re.compile(
    r"\b(register|remember|enrol|enroll|learn|memoris|memoriz|save|record)\w*\b"
    r"[^.?!]{0,24}?\b(my face|me|my picture|who i am|what i look like)\b",
    re.I)

# "...as Josh", "...call me Josh", "...my name is Josh", "...I'm Josh".
_NAME_IN = re.compile(
    r"\b(?:as|call me|name is|i am|i'm|its|it's|this is)\s+"
    r"([a-z][a-z'\-]{1,15})\b", re.I)

# A bare name, for the follow-up turn. Tolerates the polite padding people
# actually say rather than requiring a single bare word.
_NAME_ONLY = re.compile(
    r"^(?:(?:my name is|call me|i am|i'm|its|it's|this is|just)\s+)?"
    r"([a-z][a-z'\-]{1,15})\s*$", re.I)

_FORGET = re.compile(
    r"\b(forget|delete|remove|erase)\w*\b[^.?!]{0,24}?\b(my face|me|"
    r"([a-z][a-z'\-]{1,15})'s face)\b", re.I)

# Words that are never a person's name, however confidently the recogniser
# offers them. Without this, "register my face please" enrols PLEASE.
_NOT_NAMES = set("""
face please thanks thank you yes no ok okay tek tech sure yeah yep nope now
here there that this it me my your the a an and but so then what who when
""".split())


def name_from(text, pattern=_NAME_IN):
    m = pattern.search(text or "")
    if not m:
        return None
    n = m.group(1).strip().upper()
    return None if n.lower() in _NOT_NAMES else n


# The poses, in order, each as (what to say, why it is in the list).
#
# Straight-on first, because if only one picture ever lands it should be the
# most useful one, and someone who has just been asked to hold still can do
# that immediately.
#
# They are deliberately GENTLE, which is the opposite of the first attempt. "Turn your head to your right", "tilt your head over", "chin up" -
# all of them produced samples that failed leave-one-out 10 times out of 16.
#
# The reason is that the capture pipeline is frontal end to end:
# haarcascade_frontalface_default finds the face and an LBF fitter places the
# eyes. The poses that would most enrich a gallery are precisely the ones that
# break both, so each awkward angle silently fell back to an unaligned crop -
# three had no face in them at all. Asking for variety the pipeline cannot
# capture does not produce varied samples, it produces broken ones.
#
# So the variation now comes from expression, distance and small movements
# that keep the face frontal. Real variety across pose has to come from the
# detector, not from the instructions.
POSES = [
    ("Look straight at me.",                         "the canonical view"),
    ("Stay there, and give me a small smile.",        "expression"),
    ("Now relax your face completely.",               "expression"),
    ("Lean in a little closer.",                      "scale, still frontal"),
    ("And sit back to where you were.",               "scale, still frontal"),
    ("Turn your head just slightly to one side.",     "a little yaw, safely"),
    ("Back to straight on, and raise your eyebrows.", "brow, still frontal"),
    ("Last one. Straight at me, relaxed.",            "a second canonical view"),
]


def _fresh_crop(after, timeout=6.0):
    """A face crop written strictly AFTER `after`. None if none arrives.

    Waiting for a NEW file rather than reading whatever is there is the whole
    difference between eight poses and eight copies of one: the display rewrites
    the crop every 1.5 s, so reading immediately after asking somebody to turn
    their head returns the picture from before they turned it.
    """
    import cv2
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            m = os.path.getmtime(CROP)
        except OSError:
            time.sleep(0.2)
            continue
        if m > after:
            g = cv2.imread(CROP, cv2.IMREAD_GRAYSCALE)
            if g is not None and g.size >= 100:
                return g
        time.sleep(0.2)
    return None


def enrol(service, name, samples=VOICE_SAMPLES):
    """Walk somebody through being photographed. Returns (ok, closing line).

    Spoken, posed and counted down, because the alternative - going quiet and
    silently grabbing frames - gets you eight pictures of a person standing
    still wondering whether the device has crashed, which is both a worse
    experience and a worse gallery.

    Runs on the voice service's consider-thread, never the display's frame
    loop. It blocks for the best part of a minute and the display must not
    notice.
    """
    from .. import recog

    poses = POSES[:samples]
    service._say("Right. I am going to take %d pictures, and I will tell you "
                 "how to hold your head for each one. Stay where you are."
                 % len(poses))

    got, missed, added = 0, 0, []
    for i, (instruction, _why) in enumerate(poses, 1):
        service._say("Picture %d. %s" % (i, instruction))
        # The countdown is not decoration. It is what makes somebody hold the
        # pose at a known moment instead of drifting through it, and it gives
        # the display time to write a crop of the NEW pose.
        service._say("Three. Two. One.")
        mark = time.time()
        g = _fresh_crop(mark)
        if g is None:
            missed += 1
            service._say("I did not catch that one.")
            continue
        added.append(recog.save_sample(name, g))
        got += 1
        # Not after every single one - eight "got it"s in a row is nagging.
        if i in (1, len(poses)) or i % 3 == 0:
            service._say("Got it.")

    if not got:
        return False, ("I could not see your face clearly for any of those. "
                       "Come a bit closer, face me, and ask me again.")

    # Prove it helped before keeping it. An enrolment that makes recognition
    # WORSE is not a hypothetical: the first version of this added 16 samples
    # that took leave-one-out from 0 failures in 12 to 10 in 16, and reported
    # itself a success the whole time. The gallery is the only thing that knows
    # whether new pictures belong in it, so ask it.
    #
    # Same principle as tools/face_realign.py, which measures before and after
    # and refuses to write if alignment did not help.
    kept = _verify(name, added)
    total = len(recog.samples(name))
    if kept == 0:
        return False, ("I took those pictures but they were not good enough to "
                       "keep, so I have thrown them away and left what I had. "
                       "Better light or a bit closer would help.")
    recog.note_enrolled(name, total)
    if kept < got or missed:
        return True, ("Done. I kept %d of them, and I have %d pictures of you "
                      "altogether. I will recognise you as %s."
                      % (kept, total, name.title()))
    return True, ("All done. That is %d new pictures, %d altogether, and I "
                  "will recognise you as %s from now on."
                  % (kept, total, name.title()))


def _loo_fail_rate(imgs):
    """Fraction of samples the rest of the gallery cannot recognise."""
    import cv2
    import numpy as np
    from .. import recog
    if len(imgs) < 4:
        return 0.0
    bad = 0
    for k in range(len(imgs)):
        train, ids = [], []
        for j, g in enumerate(imgs):
            if j == k:
                continue
            train.append(g)
            ids.append(0)
            for frac, blur in recog._AUGMENT:
                train.append(recog._degrade(g, frac, blur))
                ids.append(0)
        m = cv2.face.LBPHFaceRecognizer_create()
        m.train(train, np.array(ids))
        if m.predict(imgs[k])[1] > recog.THRESHOLD:
            bad += 1
    return float(bad) / len(imgs)


def _verify(name, added):
    """Keep the new samples only if they did not make the gallery worse.

    Returns how many were kept. Rejected files are moved out of the gallery
    entirely rather than deleted - they are the evidence for why an enrolment
    was refused, and a directory of them left INSIDE the gallery would be
    listed by recog.people() as a person.
    """
    import shutil
    import cv2
    from .. import recog

    if not added:
        return 0
    paths = recog.samples(name)
    old = [p for p in paths if p not in added]
    if len(old) < 4:
        return len(added)              # nothing to compare against; keep them

    def load(ps):
        out = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in ps]
        return [g for g in out if g is not None]

    before = _loo_fail_rate(load(old))
    after = _loo_fail_rate(load(paths))
    # A little slack: leave-one-out gets harder as a gallery grows, and a
    # single marginal sample is not a reason to throw away a whole enrolment.
    if after <= before + 0.10:
        return len(added)

    rej = os.path.expanduser("~/.cache/tekdromo/rejected")
    dest = os.path.join(rej, "%s-%d" % (name, int(time.time())))
    try:
        os.makedirs(dest)
    except OSError:
        pass
    for p in added:
        try:
            shutil.move(p, os.path.join(dest, os.path.basename(p)))
        except OSError:
            pass
    print("enrol: rejected %d samples for %s - leave-one-out failure rate "
          "%.0f%% -> %.0f%%; moved to %s"
          % (len(added), name, before * 100, after * 100, dest), flush=True)
    return 0


def handle(service, ev):
    """Take this utterance if it is a command. True if the brain should not see it.

    `service` is the VoiceService: this needs its speaker (to answer), and its
    pending-intent slot (to run a two-turn exchange).
    """
    heard = (ev.get("heard") or "").strip()
    if not heard:
        return False

    pending = getattr(service, "pending_intent", None)
    if pending and time.time() < pending.get("until", 0):
        if pending.get("want") == "enrol_name":
            name = name_from(heard, _NAME_ONLY) or name_from(heard)
            service.pending_intent = None
            if not name:
                service._say("I did not catch a name, so I have not saved "
                             "anything. Ask me again when you are ready.")
                return True
            service._say("Saving your face as %s. Look at me and move your "
                         "head a little." % name.title())
            ok, msg = enrol(service, name)
            service._say(msg)
            return True
    elif pending:
        service.pending_intent = None          # expired

    if _FORGET.search(heard):
        from .. import recog
        name = name_from(heard) or (
            _FORGET.search(heard).group(3) or "").upper() or None
        if not name:
            # "forget me" with no name: only unambiguous with one person
            # enrolled. Guessing which of several to delete is not a guess
            # worth making - the data is gone either way if it is wrong.
            people = recog.people()
            if len(people) == 1:
                name = people[0]
            else:
                service._say("Tell me which name to forget.")
                return True
        recog.forget(name)
        try:
            from .. import memory
            memory.forget(name)
        except Exception:
            pass
        service._say("Done. I have forgotten %s." % name.title())
        return True

    if _ENROL.search(heard):
        name = name_from(heard)
        if name:
            service._say("Saving your face as %s. Look at me and move your "
                         "head a little." % name.title())
            ok, msg = enrol(service, name)
            service._say(msg)
        else:
            service.pending_intent = {"want": "enrol_name",
                                      "until": time.time() + PENDING_S}
            service._say("Happy to. What name should I save you under?")
        return True

    return False
