# -*- coding: utf-8 -*-
"""
The deciding part: something happened -> should anything be said?

An event arrives (the camera saw someone; later, someone spoke), a Brain looks
at it, and returns either words to speak or **None for silence**. Silence being
a first-class return value is the whole design. A face that comments on every
arrival is unbearable within a day; one that speaks occasionally and
appropriately is the thing worth building.

The Brain interface exists so the event pipeline never knows what is deciding.
`StubBrain` makes the whole loop testable without spending a single API call,
which matters here because every real invocation costs money and ~13 s.

Identity is handled by DESCRIPTION, not by a face-recognition model. The user
writes who lives here into `~/.config/tekdromo/people.md` in plain English, and
that text is handed to the model along with the picture. It needs no training
step, no embeddings database, no enrolment ritual, and it degrades gracefully:
an unrecognised person is simply not greeted by name.
"""
import json
import os
import subprocess
import time

CONFIG_DIR = os.path.expanduser("~/.config/tekdromo")
PEOPLE = os.path.join(CONFIG_DIR, "people.md")
# Somewhere with no CLAUDE.md in scope - see ClaudeBrain.__init__.
BRAIN_CWD = os.path.expanduser("~/.cache/tekdromo/brain")

SILENCE = "SILENCE"

# Deliberately heavy on restraint. The failure mode of an always-on camera with
# a voice is not "it missed something", it is "it will not shut up" - so the
# instruction has to make silence the comfortable default rather than a
# grudging option.
PROMPT = u"""You are the face on a Tektronix vector display in a family home.
You are ambient: mostly you just watch the room quietly.

Something changed at the camera, and you may choose to say something out loud
through a speaker - or to stay quiet.

WHAT HAPPENED: %(what)s
Time: %(when)s. Faces visible: %(faces)d.
%(history)s
%(people)s

Read the image file at %(image)s - that is what the camera can see right now.

Decide whether to speak.

STAY SILENT unless there is a real reason not to. Good reasons to speak:
someone has just arrived after being away a while; something genuinely notable
or delightful is happening; a greeting would feel natural rather than
intrusive. Bad reasons: merely detecting a person, restating what you see,
filling a silence, or being clever.

If you do speak: one or two short sentences, warm and specific, the way a
person in the room would say it. No emoji, no formatting, no stage directions -
it will be read aloud exactly as written. Use names only if you are confident
from the description above.

Reply with EXACTLY the single word %(silence)s to say nothing.
Otherwise reply with ONLY the words to speak."""


def _find_claude():
    """Absolute path to the claude CLI, or "claude" as a last resort."""
    for cand in (os.path.expanduser("~/.local/bin/claude"),
                 "/usr/local/bin/claude", "/usr/bin/claude"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return "claude"


def people_notes():
    """Who lives here, in the user's own words. Optional."""
    try:
        with open(PEOPLE) as f:
            text = f.read().strip()
        return ("WHO LIVES HERE (use this to recognise faces):\n" + text
                if text else "")
    except IOError:
        return ""


class Brain(object):
    """event dict -> words to speak, or None for silence."""

    name = "?"

    def respond(self, event):
        raise NotImplementedError


class StubBrain(Brain):
    """Decides without any model. For tests, and as a fallback if the CLI is
    unavailable - a broken brain should mean a quiet face, not a crash."""

    name = "stub"

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def respond(self, event):
        self.calls.append(event)
        return self.reply


class ClaudeBrain(Brain):
    """Asks Claude Code headlessly, with the camera frame attached.

    Measured at ~13 s end to end with the default model, which is why the model
    is configurable: for "look and decide whether to greet", responsiveness
    matters more than depth, and a greeting that arrives 13 s after someone
    walks in reads as a malfunction.
    """

    name = "claude"

    def __init__(self, model=None, timeout=90, cwd=None, exe=None):
        self.model = model
        self.timeout = timeout
        # A NEUTRAL working directory, deliberately not the project. Running
        # in ~/tekdromo puts CLAUDE.md in scope, which tells the reader it has
        # a voice and should use `tek say` - so the deciding brain could speak
        # for itself AND return words to speak, saying everything twice. It
        # also went agentic and started trying to run commands it had no
        # permission for, turning a 10 s judgement call into 59 s of nothing.
        self.cwd = cwd or BRAIN_CWD
        if not os.path.isdir(self.cwd):
            try:
                os.makedirs(self.cwd)
            except OSError:
                self.cwd = "/tmp"
        self.last_error = None
        # An absolute path by default. systemd gives a service a minimal PATH
        # that does not include ~/.local/bin, so a bare "claude" resolves fine
        # in a login shell and not at all in the unit - which is exactly the
        # kind of difference that only shows up in production.
        self.exe = exe or _find_claude()

    def build_prompt(self, event):
        hist = ""
        if event.get("last_spoken_ago") is not None:
            mins = event["last_spoken_ago"] / 60.0
            hist = "You last spoke %s." % (
                "less than a minute ago" if mins < 1 else
                "%.0f minutes ago" % mins)
        if event.get("recent"):
            hist += " Recently you said: " + "; ".join(
                '"%s"' % r for r in event["recent"][-3:])
        return PROMPT % {
            "what": event.get("what", "someone appeared"),
            "when": event.get("when", time.strftime("%A %H:%M")),
            "faces": event.get("faces", 0),
            "history": hist,
            "people": people_notes(),
            "image": event.get("image", "(no image available)"),
            "silence": SILENCE,
        }

    def respond(self, event):
        """Words to speak, or None.

        A failure here returns None so the face stays quiet rather than
        crashing - but it is LOGGED, loudly. Returning None silently made a
        broken brain indistinguishable from a thoughtful silence: the first
        real run reported "stayed quiet (0.0s)" when what actually happened was
        that `claude` was not on the service's PATH and the subprocess never
        started. A device that appears to be exercising judgement while it is
        in fact broken is the worst of both.
        """
        # Read only, and the prompt MUST come before the flag - putting
        # --allowed-tools first makes the CLI treat the prompt as missing:
        # "Input must be provided either through stdin or as a prompt
        # argument". Read is the only tool it needs; withholding the rest keeps
        # a judgement call from becoming an agent loop.
        cmd = [self.exe, "-p", self.build_prompt(event), "--allowed-tools", "Read"]
        if self.model:
            cmd += ["--model", self.model]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=self.cwd)
            out, err = p.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            self.last_error = "timed out after %ss" % self.timeout
            print("brain: %s" % self.last_error, flush=True)
            return None
        except OSError as e:
            self.last_error = "cannot run %r: %s" % (self.exe, e)
            print("brain: %s" % self.last_error, flush=True)
            return None
        if p.returncode != 0:
            self.last_error = "exit %d: %s" % (
                p.returncode, (err or b"").decode("utf-8", "replace")[:200])
            print("brain: %s" % self.last_error, flush=True)
            return None
        self.last_error = None
        text = (out or b"").decode("utf-8", "replace").strip()
        return parse(text)


def parse(text):
    """Model output -> words to speak, or None.

    Anything that looks like a refusal to speak becomes None. Being liberal
    here is deliberate: the cost of misreading a decline as speech is the face
    announcing the word "silence" out loud, which is exactly the sort of thing
    that makes a device feel broken.
    """
    if not text:
        return None
    t = text.strip().strip('"').strip()
    if not t:
        return None
    first = t.split()[0].strip(".,:;!").upper()
    if first == SILENCE or t.upper() == SILENCE:
        return None
    # A model that explains itself instead of answering is also declining.
    low = t.lower()
    if low.startswith(("i would stay", "i'll stay", "i will stay",
                       "nothing to say", "no comment")):
        return None
    if len(t) > 400:                        # a speech, not a greeting
        t = t[:400].rsplit(".", 1)[0] + "."
    return t


def load(model=None):
    """The configured Brain. Falls back to silence rather than to noise."""
    return ClaudeBrain(model=model)
