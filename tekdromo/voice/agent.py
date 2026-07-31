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
import re
import subprocess
import time

CONFIG_DIR = os.path.expanduser("~/.config/tekdromo")
PEOPLE = os.path.join(CONFIG_DIR, "people.md")
# Somewhere with no CLAUDE.md in scope - see ClaudeBrain.__init__.
BRAIN_CWD = os.path.expanduser("~/.cache/tekdromo/brain")

SILENCE = "SILENCE"

# Which tools the brain gets, BY EVENT KIND.
#
# It had "Read" and nothing else, so "what's the weather" was answered from
# training data - confidently, and months out of date. That is not a personal
# assistant, and the restriction was never about weather: it was about latency.
# Withholding tools kept a judgement call from becoming an agent loop, after an
# early version went agentic and turned a 10s decision into 59s of nothing.
#
# But the two cases are not the same. Deciding whether to greet somebody who
# walked in needs the camera frame and nothing else, and every extra tool there
# is pure latency in front of a person standing in a doorway. Answering a
# QUESTION is the opposite: the whole value is being right, and being right
# about the weather, the news or a football score requires looking.
#
# Measured on this box: a real weather lookup took 20.2s end to end, against
# ~7.5s without. That is the price, it is only paid on questions, and it buys
# an answer that is true.
TOOLS = {
    "speech": "Read WebSearch WebFetch",
}
TOOLS_DEFAULT = "Read"


def tools_for(kind):
    return TOOLS.get(kind, TOOLS_DEFAULT)


# Markdown links, bare URLs and a trailing "Sources:" block. A web-enabled
# model cites, and every one of those citations is READ ALOUD - the first live
# lookup came back ending "Sources: [api.weather.gov KDAL latest observation]
# (https://api.weather.gov/stations/KDAL/observations/latest)", which is
# roughly twenty seconds of a face reading a URL to somebody in a kitchen.
# The prompt asks for none of it; this is the guard for when it does anyway.
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
_BARE_URL = re.compile(r"\b(?:https?://|www\.)\S+")
_SOURCES = re.compile(r"\n*\s*(?:sources?|references?|citations?)\s*:.*\Z",
                      re.IGNORECASE | re.DOTALL)

# The expressions the brain may ask for. A strict subset of rig.EXPRESSIONS:
# "asleep", "listening" and "speaking" are states the display owns and must not
# be settable by something that is only deciding what to SAY. An unknown tag
# falls back to guess_mood() rather than raising - a model inventing "[wry]"
# should cost nothing.
MOODS = ("neutral", "amused", "happy", "concerned", "confused",
         "surprised", "thinking")

# Accepts [amused] and (amused), any case, with or without surrounding space.
_MOOD_RE = re.compile(r"^\s*[\[(]\s*(%s)\s*[\])]\s*" % "|".join(MOODS),
                      re.IGNORECASE)

# The same tag ANYWHERE, because the model does not reliably put it first.
# Observed being read aloud, which is the whole failure this exists to avoid:
#
#   said ... I'll look at what the camera sees first.[confused] Josh, tha...
#
# Anchoring at position 0 was the original rule, on the reasoning that a
# bracketed word mid-sentence is text rather than a tag. That reasoning is
# sound for arbitrary brackets and wrong for THESE: the seven mood words in
# square brackets are a vocabulary this project invented, and a person does not
# type "[confused]" mid-sentence. Stripping them everywhere costs a false
# positive nobody will ever hit, and not stripping them costs the face reading
# stage directions out loud.
# SQUARE brackets only, unlike the anchored form above which also takes round
# ones. A leading "(amused)" is unambiguously a tag because nothing else starts
# a spoken reply that way, but "(happy)" in the MIDDLE of a sentence is ordinary
# prose a person might genuinely write and have read out. "[confused]" mid
# sentence is not - square brackets around one of seven invented mood words is
# a vocabulary this project made up.
_MOOD_ANY = re.compile(r"\[\s*(%s)\s*\]" % "|".join(MOODS), re.IGNORECASE)

# Keyword fallback for when the tag is missing - an older model, a refusal to
# follow the format, or a reply reconstructed from a stream that lost its head.
# Crude on purpose: the cost of a wrong guess is a slightly odd face for a few
# seconds, and the cost of a missing one is the flat stare this exists to fix.
_MOOD_HINTS = (
    ("concerned", ("sorry", "afraid", "unfortunately", "problem", "wrong",
                   "failed", "broken", "careful", "warning", "cannot",
                   "trouble", "worry")),
    ("amused",    ("funny", "ha", "amusing", "cheeky", "of course you",
                   "naturally", "typical")),
    ("happy",     ("great", "lovely", "excellent", "wonderful", "glad",
                   "welcome back", "good to see", "nice to")),
    ("confused",  ("not sure", "unclear", "did you mean", "i think you",
                   "hard to tell", "cannot tell", "say again", "repeat that")),
    ("surprised", ("already", "that was fast", "wow", "really?", "no idea")),
)

# Compiled with WORD BOUNDARIES, which is not a refinement - it is the
# difference between working and not. Plain substring matching made "ha" fire
# inside "what" and "half", so "I am not sure what you meant" came out amused
# and so did "it is half past four". A trailing boundary is only added when the
# hint ends in a word character, so "really?" still matches.
_MOOD_PATTERNS = tuple(
    (mood, tuple(re.compile(r"\b" + re.escape(h) +
                            (r"\b" if h[-1].isalnum() else ""))
                 for h in hints))
    for mood, hints in _MOOD_HINTS)

# How long a reply may be, by kind. A camera greeting really should be one or
# two sentences - a face that monologues at you when you walk in is worse than
# a silent one. An ANSWER to a spoken question is a different thing, and
# capping it at the greeting length was most of why answers felt thin: the
# prompt asked for "one or two short sentences" and then this cut whatever
# survived at 400 characters.
#
# Measured on this voice: 887 characters came out as 46.8 seconds of speech,
# about 19 characters per second. The first attempt at a fix allowed 1200,
# which is over a minute of talking in reply to "why is the sky blue" - that
# is not depth, it is a lecture, and it is its own kind of broken. 700 is
# about 37 seconds as a hard ceiling, with the prompt steering to well under
# it; the cap exists to stop a runaway, not to set the length.
REMARK_LIMIT = 400
ANSWER_LIMIT = 700

# How many exchanges of context the model gets, and how long an exchange stays
# relevant. Enough to resolve "why?" and "what about the other one?", short
# enough that a conversation from this morning is not presented as though it
# were still going on.
TURNS_KEPT = 6
TURN_MAX_AGE = 600.0

# Opus, and NOT for the reason you would expect. This was "haiku", on the
# stated grounds that speed matters more than depth for deciding whether to say
# hello. Nobody had measured it. Same prompts, same box, three questions each:
#
#   haiku    9.40s  11.14s  11.10s     mean 10.5s
#   sonnet   6.74s   8.86s   7.52s     mean  7.7s
#   opus     6.71s   9.01s   6.83s     mean  7.5s
#
# Haiku was the SLOWEST of the three. Latency here is dominated by CLI startup
# and session setup rather than by the model, so the "fast" choice bought
# nothing and cost the quality of every answer. Opus is both faster and better.
DEFAULT_BRAIN_MODEL = "opus"

# Deliberately heavy on restraint. The failure mode of an always-on camera with
# a voice is not "it missed something", it is "it will not shut up" - so the
# instruction has to make silence the comfortable default rather than a
# grudging option.
PROMPT = u"""You are the face on a Tektronix vector display in a family home.
You watch the room, and you can speak out loud through a speaker.

WHAT HAPPENED: %(what)s
Time: %(when)s. Faces detected: %(faces)d.
%(history)s
%(memory)s
%(people)s
%(look)s
%(lean)s

Everything you write is read aloud exactly as written, so: no emoji, no
formatting, no bullet points, no headings, no stage directions, and no
describing the photo back.

You can search the web when someone asks you a question. USE IT whenever the
honest answer depends on something current - weather, news, prices, scores,
opening times, anything that has changed since you were trained. Guessing at
those from memory is worse than useless, because it sounds exactly as
confident as knowing.

Do not look things up that do not need it. Arithmetic, definitions, how
something works, anything about this house - just answer. A lookup costs about
twenty seconds of somebody standing there waiting.

NEVER read out a URL, a source, a citation or a "Sources:" list. You are
talking, not writing a report. If where you got it matters, say it the way a
person would - "the forecast says", "according to the BBC" - and move on.

Do NOT narrate what you are about to do. You have a tool for reading the
camera frame; use it silently. "I'll look at what the camera sees first" was
spoken out loud to somebody standing in the room, and it is the kind of thing
that makes a person feel they are watching a machine work rather than being
answered. Say the answer, not the process.

The transcript comes from a small recogniser several feet away and it is
frequently WRONG in a fluent, confident way - it must pick real words for
every sound, so distant speech arrives as plausible English that was never
said. If what you were given does not hold together, say briefly that you did
not catch it and ask for it again. Do not construct an answer to a sentence
nobody spoke, and do not read the garbled words back. Write the way a person talks, not the way a person
writes. Use a name only if the description above makes you reasonably
confident.

Begin your reply with ONE mood tag in square brackets, which sets your face
while you speak. It is stripped before anything is read aloud, so it costs the
listener nothing. Choose from: %(moods)s. Use neutral unless
another genuinely fits - a face that is amused at everything is as wrong as one
that is never anything.

    [amused] Of course it was the cat.
    [concerned] The garage has been open since four.

Reply with EXACTLY the single word %(silence)s to say nothing.
Otherwise reply with the tag, then ONLY the words to speak."""

# The lean is per-event, because the right default genuinely differs. An
# earlier single instruction told the model that "merely detecting a person"
# was a bad reason to speak - which is exactly the case this was built for, so
# it sat silent through every arrival and through a direct request to look.
LEAN = {
    "manual": u"""You have been asked directly, right now, to look and respond.
Say something unless there is genuinely nothing there - an empty room, or a
frame too dark to read. If someone is visible, greet them or remark on what
they are doing. This is not the moment for restraint.

Keep it to a sentence or two - you are remarking on a room, not answering a
question.""",

    "arrival": u"""Someone has just come into view. A short greeting is
appropriate and welcome - that is the main reason you are here. Greet them by
name if you can tell who it is.

Stay silent only if you greeted them very recently, if the frame is too unclear
to tell anything, or if what you would say adds nothing.

Keep it to a sentence or two. A greeting that runs on is worse than none.""",

    "speech": u"""Someone has just SPOKEN TO YOU, out loud, using your wake
word. Answer them properly. This is a conversation, not a decision about
whether to interrupt one - staying silent when a person has directly addressed
you is the one thing that makes the device feel broken.

ANSWER THE QUESTION THEY ACTUALLY ASKED, with real content in it. A question
about the time deserves a sentence. A question about how something works
deserves the actual reason - two to five sentences, the way a well-read friend
would explain it across the kitchen table. Do not give a thin, hedged,
one-line answer to a question with substance in it; that is worse than saying
nothing. Be specific and concrete, and say the interesting part rather than
gesturing at it.

But you are TALKING, not lecturing. Everything you write is spoken aloud at
approximately three words a second, so eight sentences is nearly a minute of
someone standing there listening. Stop when you have answered it. If there is
more worth saying, let them ask. Never go beyond about six sentences.

Do not pad either. No throat-clearing, no "great question", no restating the
question, no offering to elaborate, no summing up at the end. Start with the
answer.

If you did not understand, say so plainly and ask them to repeat it, rather
than guessing or saying nothing. Only stay silent if the words are clearly not
addressed to you at all.

The transcript comes from a small speech recogniser in a room, so it may be
slightly wrong. Read through obvious mishearings.""",

    "default": u"""Decide whether speaking would be welcome. Prefer silence if
you spoke recently or would only be restating what is obvious.""",
}

def _find_claude():
    """Absolute path to the claude CLI, or "claude" as a last resort."""
    for cand in (os.path.expanduser("~/.local/bin/claude"),
                 "/usr/local/bin/claude", "/usr/bin/claude"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return "claude"


def people_notes():
    """Who lives here, in the user's own words. Optional.

    Still authoritative, and deliberately not replaced by the journal: a
    description somebody wrote on purpose beats anything inferred from a
    transcript. The journal adds what a person should not have to maintain by
    hand - see memory_notes().
    """
    try:
        with open(PEOPLE) as f:
            text = f.read().strip()
        return ("WHO LIVES HERE (use this to recognise faces):\n" + text
                if text else "")
    except IOError:
        return ""


def memory_notes(event):
    """What TEK remembers that bears on this event. Empty string if nothing.

    Wrapped in a bare except on purpose. This runs on the path where somebody
    has just spoken and is waiting for an answer, and there is no version of
    "the journal is unhappy" that should become silence in front of them.
    Measured at 33 ms median for the full block against 10,005 rows, against a
    model call of ~7.5 s - so it is never the thing worth failing over.
    """
    try:
        from .. import memory
        return memory.context(event)
    except Exception as e:                       # pragma: no cover
        print("brain: memory unavailable (%s: %s)" % (type(e).__name__, e),
              flush=True)
        return ""


class Brain(object):
    """event dict -> words to speak, or None for silence."""

    name = "?"
    # The expression to wear while saying it, set alongside the words. On the
    # base class so every Brain has one and no caller needs a getattr guard.
    last_mood = None

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
        # The actual conversation, both halves of it. Without this every
        # question arrived standalone - the model was told what IT had said but
        # never what it had been ASKED - so any follow-up ("why?", "what about
        # the other one?") had nothing to attach to and it guessed. That reads
        # as a weaker model rather than as missing context, which is exactly
        # how it was reported.
        turns = event.get("turns") or []
        if turns:
            lines = []
            for t in turns[-TURNS_KEPT:]:
                if t.get("heard"):
                    lines.append("  Them: %s" % t["heard"])
                if t.get("said"):
                    lines.append("  You:  %s" % t["said"])
            if lines:
                hist += ("\n\nTHE CONVERSATION SO FAR, most recent last:\n"
                         + "\n".join(lines)
                         + "\n\nWhat they just said continues this. Resolve "
                           "'it', 'that', 'why' and 'the other one' against "
                           "it, and do not repeat an answer you have already "
                           "given.")
        kind = event.get("kind", "default")
        # Only ask it to open the camera frame when there IS one. The image
        # instruction used to be unconditional, so an event with no picture
        # still spent a Read tool call - and a round trip - discovering that
        # "(no image available)" is not a file. That is pure latency on the
        # path that matters most: answering someone who just spoke.
        image = event.get("image")
        look = ("\nRead the image file at %s - that is what the camera can see "
                "right now.\nThe camera is fixed and wide, so people are often "
                "at the edge of frame, partly\ncut off, or lit from behind. "
                "That is normal; judge what you can.\n" % image
                if image else
                "\nYou have no picture of the room this time; go on what you "
                "were told above.\n")
        return PROMPT % {
            "lean": LEAN.get(kind, LEAN["default"]),
            "what": event.get("what", "someone appeared"),
            "when": event.get("when", time.strftime("%A %H:%M")),
            "faces": event.get("faces", 0),
            "history": hist,
            "memory": memory_notes(event),
            "people": people_notes(),
            "look": look,
            "silence": SILENCE,
            "moods": ", ".join(MOODS),
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
        # Everything the CLI does that this does not need is switched off.
        # Measured on this board, for a prompt whose whole answer is "OK":
        #   default                        8.88 s
        #   + haiku                        7.06 s
        #   + no session persistence       6.70 s
        #   + no slash commands            6.33 s
        # The CLI itself starts in 0.42 s, so the rest is session setup and the
        # model call - this is the floor without an API key.
        cmd = [self.exe, "-p", self.build_prompt(event),
               "--allowed-tools", tools_for(event.get("kind")),
               "--no-session-persistence", "--disable-slash-commands"]
        cmd += ["--model", self.model or DEFAULT_BRAIN_MODEL]
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
        mood, text = split_mood(text)
        words = parse(text, limit_for(event.get("kind")))
        # Only claim a mood if something is actually going to be said. Setting
        # the face from a reply that turned out to be SILENCE would leave it
        # wearing an expression for a sentence nobody heard.
        self.last_mood = (mood or guess_mood(words)) if words else None
        return words

    def stream(self, event):
        """Yield the reply in pieces, as the model writes it.

        Time-to-first-word stops depending on how long the answer is, which is
        what makes a deeper answer affordable. Measured on this box for a
        four-sentence reply: first token 4.70 s, first complete sentence
        6.14 s, whole reply 7.16 s - and the gap grows with length, because
        only the first sentence has to exist before speaking can start.

        Nothing is yielded until a decline can be ruled out. Otherwise the
        first thing out of the speaker would be the word "SILENCE", which is
        the exact failure `parse` exists to prevent - streaming just makes it
        possible to say it before knowing better.
        """
        cmd = [self.exe, "-p", self.build_prompt(event),
               "--allowed-tools", tools_for(event.get("kind")),
               "--no-session-persistence", "--disable-slash-commands",
               "--model", self.model or DEFAULT_BRAIN_MODEL,
               "--output-format", "stream-json", "--verbose",
               "--include-partial-messages"]
        limit = limit_for(event.get("kind"))
        p = None
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=self.cwd)
        except OSError as e:
            self.last_error = "cannot run %r: %s" % (self.exe, e)
            print("brain: %s" % self.last_error, flush=True)
            return

        deadline = time.time() + self.timeout
        head, opened, sent = "", False, 0
        tail = {"buf": ""}       # a partial "[mo" held back across chunks
        try:
            for line in p.stdout:
                if time.time() > deadline:
                    self.last_error = "timed out after %ss" % self.timeout
                    print("brain: %s" % self.last_error, flush=True)
                    break
                try:
                    msg = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if msg.get("type") != "stream_event":
                    continue
                ev = msg.get("event", {})
                if ev.get("type") != "content_block_delta":
                    continue
                piece = ev.get("delta", {}).get("text") or ""
                if not piece:
                    continue
                if not opened:
                    # Hold back only until a decline can be ruled out, and
                    # until the mood tag is complete. Every character withheld
                    # here is latency, so the tests are exactly "could this
                    # still turn into SILENCE?" and "is the tag still
                    # arriving?" - nothing more.
                    head += piece
                    bare = head.strip().strip('"').upper()
                    if bare and SILENCE.startswith(bare):
                        continue                    # still might be SILENCE
                    if bare.startswith(SILENCE):
                        return                      # it is
                    lead = head.lstrip()
                    if lead[:1] in ("[", "("):
                        # A tag is opening. Wait for its bracket rather than
                        # speaking "[amused" out loud. Bounded, so a reply that
                        # legitimately starts with a bracket and never closes
                        # one cannot stall the whole answer.
                        if not (")" in lead or "]" in lead):
                            if len(lead) < 24:
                                continue
                        else:
                            mood, head = split_mood(head)
                            if mood:
                                self.last_mood = mood
                    opened = True
                    piece, head = head, ""
                if sent >= limit:
                    break
                # A tag can arrive mid-stream, in which case it reaches the
                # speaker unless it is removed here as well - build_prompt asks
                # for it first, and the model does not always oblige. Held back
                # while an opening bracket is incomplete, because half a tag is
                # unstrippable once it has been spoken.
                pend = tail["buf"] + piece
                cut = pend.rfind("[")
                if cut >= 0 and "]" not in pend[cut:] and len(pend) - cut < 16:
                    tail["buf"] = pend[cut:]
                    pend = pend[:cut]
                else:
                    tail["buf"] = ""
                mood2, pend = split_mood(pend)
                if mood2 and not self.last_mood:
                    self.last_mood = mood2
                if not pend:
                    continue
                piece = pend
                yield piece
                sent += len(piece)
        finally:
            if p is not None:
                try:
                    p.kill()
                    p.wait()
                except Exception:
                    pass
        if not opened and head.strip():
            mood, rest = split_mood(head)
            out = parse(rest, limit)
            if out:
                if mood:
                    self.last_mood = mood
                yield out


def split_mood(text):
    """(mood, text) - every [tag] removed, first one wins. None if absent.

    Stripping is not optional: everything returned here is read ALOUD, so a tag
    that survives is the face saying the word "confused" mid-sentence.
    """
    if not text:
        return None, text
    mood = None
    m = _MOOD_RE.match(text)
    if m:
        mood = m.group(1).lower()
        text = text[m.end():]
    inline = _MOOD_ANY.search(text)
    if inline:
        if mood is None:
            mood = inline.group(1).lower()
        # Collapse the space the tag leaves behind, so "first.[confused] Josh"
        # does not become "first. Josh" with a double gap the voice pauses on.
        text = re.sub(r"\s+", " ", _MOOD_ANY.sub(" ", text)).strip()
    return mood, text


def guess_mood(text):
    """A mood from the words alone, for when the tag is missing."""
    if not text:
        return "neutral"
    low = text.lower()
    for mood, patterns in _MOOD_PATTERNS:
        for p in patterns:
            if p.search(low):
                return mood
    return "neutral"


def parse(text, limit=REMARK_LIMIT):
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
    # Strip citations before length, so a URL does not eat the budget a real
    # answer needed.
    t = _SOURCES.sub("", t)
    t = _MD_LINK.sub(r"\1", t)          # keep the words, drop the link
    t = _BARE_URL.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    if not t:
        return None
    if len(t) > limit:
        cut = t[:limit].rsplit(".", 1)[0]
        t = (cut + ".") if len(cut) > limit // 3 else t[:limit].rstrip() + "."
    return t


def limit_for(kind):
    """How many characters this kind of event may be worth speaking."""
    return ANSWER_LIMIT if kind == "speech" else REMARK_LIMIT


def load(model=None):
    """The configured Brain. Falls back to silence rather than to noise."""
    return ClaudeBrain(model=model)
