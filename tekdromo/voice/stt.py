"""
Speech recognition. Local only - nothing leaves the house.

One engine, two grammars. Vosk's recogniser can be constrained to a small
vocabulary at construction time, so the wake word is the SAME model with
`["hey tek", "[unk]"]` instead of a second model, a second dependency and a
second thing to tune. Constrained decoding is also much cheaper than free
decoding, which matters when it is the thing running whenever someone speaks.

    wake = Recogniser(grammar=WAKE)     # cheap, always on
    full = Recogniser()                 # only after the wake word fires

Both share one loaded Model - it is ~68 MB on disk and the process has ~800 MB
to work with, so loading it twice is not an option.

The vosk wheel's libvosk.so is built against GCC 11+'s libstdc++ and will not
load against Ubuntu 18.04's GCC 7.5 runtime. tek-voice.service sets
LD_LIBRARY_PATH to a newer libstdc++ kept in tekdromo/lib for this reason; the
system runtime is deliberately untouched. See docs.
"""
import json
import os

import numpy as np

from . import pcm

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")

# The wake word. "[unk]" is required: without it the decoder must map whatever
# it hears onto the listed phrases, so ordinary conversation is forced into a
# false match. With it, everything else has somewhere to go.
# Every word here must exist in the model's vocabulary. "tekdromo" does not,
# and Vosk only warns ("Ignoring word missing in vocabulary") before carrying
# on - so it looked configured and could never once have fired. Multiple
# spellings are listed because the recogniser genuinely cannot tell "tek" from
# "tech", and insisting on one of them just loses wake-ups.
# Every one of these was checked against the model's vocabulary with
# tools/wake_probe.py - a grammar entry containing a word the model cannot
# pronounce is SILENTLY DEAD (Vosk logs a warning and carries on), which has
# already happened once here with "tekdromo". "hey tekk" is out of vocabulary
# and is deliberately absent.
#
# The extra spellings are not different wake words, they are the same one
# heard badly. Deliberately excluded: "hey take", "hey check" and "hi tech",
# which the probe shows would also fire but which people actually say.
WAKE_WORDS = ["hey tek", "hey tech", "hey tec", "hey tex", "hey deck",
              "ok tek", "ok tech", "okay tek", "okay tech"]
WAKE_GRAMMAR = json.dumps(WAKE_WORDS + ["[unk]"])

# A second constrained grammar, for the handful of CONTROL phrases that have to
# work when nothing else does.
#
# Free decoding cannot do this job in a real room. The segmenter caps an
# utterance at 15 s, and a continuously noisy room never gives it a silence to
# close on, so every "utterance" is a 15-second block of everything at once.
# Free-decoding that turned "hey tek ears on" into "years arm hate tech ears
# are" - and no amount of fuzzy matching rescues a transcript like that.
#
# A grammar can only ever emit its own phrases or [unk], which is exactly why
# the wake word survives the same 15-second blocks that destroy free decoding.
# The control phrases deserve the same protection: they are the commands you
# need most when the room is worst.
CONTROL_WORDS = ["ears on", "ears off", "wake up", "go to sleep",
                 "stop listening", "start listening", "be quiet"]
CONTROL_GRAMMAR = json.dumps(WAKE_WORDS + CONTROL_WORDS + ["[unk]"])

_MODEL = None


def model(path=DEFAULT_MODEL):
    """The shared Model. Loaded once; ~1.5s and a large chunk of RAM."""
    global _MODEL
    if _MODEL is None:
        import vosk
        vosk.SetLogLevel(-1)              # its default chatter is per-frame
        if not os.path.isdir(path):
            raise RuntimeError(
                "no speech model at %s - fetch one from "
                "alphacephei.com/vosk/models" % path)
        _MODEL = vosk.Model(path)
    return _MODEL


class Recogniser(object):
    """Offline recognition over the pipeline's PCM contract.

    grammar=None gives free decoding; a JSON list restricts the vocabulary.
    """

    def __init__(self, grammar=None, path=DEFAULT_MODEL):
        import vosk
        self.grammar = grammar
        # Set by transcribe(); None until it has run.
        self.last_conf = None
        self.last_min_conf = None
        self.last_words = 0
        self._vosk = vosk
        self._path = path
        self.rec = self._new()

    def _new(self):
        # Written out rather than as a conditional lambda: `lambda: A if g else
        # lambda: B` puts the conditional INSIDE the lambda body, so calling it
        # returns a function instead of a recogniser.
        m = model(self._path)
        if self.grammar:
            return self._vosk.KaldiRecognizer(m, pcm.RATE, self.grammar)
        return self._vosk.KaldiRecognizer(m, pcm.RATE)

    def reset(self):
        """Start a fresh utterance.

        Vosk keeps state between calls, so a recogniser reused across
        utterances leaks the previous one's context into the next result - but
        the fix is Reset(), NOT a new recogniser. Measured on this board:

            construct free recogniser   332.0 ms
            construct grammar recogniser  1.1 ms
            Reset()                       0.0 ms

        Rebuilding it per utterance made short commands run at 2.7-3.0x
        real-time while a long sentence managed 1.12x - the giveaway that the
        cost was fixed overhead rather than decoding.
        """
        try:
            self.rec.Reset()
        except AttributeError:            # very old vosk: no Reset()
            self.rec = self._new()

    def transcribe(self, samples):
        """A complete utterance -> text. Empty string if nothing was heard.

        Also records per-word confidence on `last_conf`, because the text alone
        cannot tell a real sentence from a confident hallucination. A small
        model with a 200k vocabulary MUST choose something for every sound it
        segments, so distant speech does not come back empty or garbled - it
        comes back as fluent, plausible, entirely wrong English:

            heard: "more or killed either way she didn't know what it was
                    like before remain heated know what's this"

        The brain then answers that, which is what "it says odd things" looks
        like from the room. Reading the confidence is the only way to tell that
        case from a sentence somebody actually said.
        """
        self.reset()
        try:
            self.rec.SetWords(True)
        except Exception:
            pass
        data = np.asarray(samples, dtype=pcm.DTYPE).tobytes()
        self.rec.AcceptWaveform(data)
        res = json.loads(self.rec.FinalResult())
        words = res.get("result") or []
        confs = [w.get("conf", 0.0) for w in words]
        self.last_conf = (sum(confs) / len(confs)) if confs else None
        self.last_min_conf = min(confs) if confs else None
        self.last_words = len(confs)
        return res.get("text", "").strip()

    def stream(self, source):
        """Yield (final, text) as audio arrives. final=False is a partial.

        Partials are what make a display feel responsive - the words appear as
        they are spoken rather than in one lump at the end.
        """
        self.reset()
        for frame in source:
            data = np.asarray(frame, dtype=pcm.DTYPE).tobytes()
            if self.rec.AcceptWaveform(data):
                text = json.loads(self.rec.Result()).get("text", "").strip()
                if text:
                    yield True, text
            else:
                text = json.loads(self.rec.PartialResult()).get("partial", "").strip()
                if text:
                    yield False, text
        text = json.loads(self.rec.FinalResult()).get("text", "").strip()
        if text:
            yield True, text


def heard_wake(text):
    """Did a wake-grammar result actually contain the wake word?

    The grammar decoder still emits "[unk]" and fragments, so the caller cannot
    simply treat any non-empty result as a trigger.
    """
    if not text:
        return False
    t = text.lower()
    return any(w in t for w in WAKE_WORDS)


# ---------------------------------------------------------------------------
# Radio protocol.
#
# Every turn-taking heuristic in this project exists to GUESS where a turn
# ended: VAD segmentation guessing you stopped talking, a follow-up window
# guessing the next sentence was still meant for it, a level gate guessing you
# were close enough to be addressing it, an acoustic detector guessing somebody
# was talking over a reply. Each one is a probability dressed up as a decision,
# and each one is wrong often enough to be noticed.
#
# "Over" removes the guess. It is not a heuristic, it is a statement, and radio
# operators settled on it a century ago for exactly the reason it is needed
# here: a half-duplex channel where both ends must agree whose turn it is and
# neither can hear the other while transmitting.
#
#   "hey tek"        open the channel
#   "...over"        my turn is finished, yours starts
#   "copy that"      acknowledged, I am working on it
#   "...over"        my turn is finished, yours starts
#   "over and out"   channel closed
#
# The phrases are checked at the END of an utterance, because that is where a
# sign-off goes, and "over" mid-sentence ("come over here", "think it over") is
# ordinary English. OVER_OUT is tested FIRST since it contains OVER.
OVER_OUT = ("over and out", "over out", "overandout")
OVER = ("over", "go ahead", "your turn")

# Tolerated mishearings. The small model is reliable on short words but not
# perfect, and every one of these was chosen because it is not a word somebody
# ends a sentence with by accident.
OVER_FUZZ = ("ova", "overr", "oever")


def _tail(text, n=4):
    return " ".join((text or "").lower().replace(",", " ").split()[-n:])


def ends_over_out(text):
    """Did that utterance sign off for good?"""
    t = _tail(text)
    return any(t == p or t.endswith(" " + p) or t.endswith(p)
               for p in OVER_OUT)


def ends_over(text):
    """Did that utterance hand the turn back? OVER_OUT is not OVER."""
    if ends_over_out(text):
        return False
    words = (text or "").lower().replace(",", " ").replace(".", " ").split()
    if not words:
        return False
    if words[-1] in OVER + OVER_FUZZ:
        return True
    return _tail(text, 2) in OVER


def strip_over(text):
    """The message without its sign-off - what actually gets answered."""
    words = (text or "").replace(",", " ").split()
    low = [w.lower().strip(".!?") for w in words]
    for phrase in sorted(OVER_OUT + OVER, key=lambda p: -len(p.split())):
        n = len(phrase.split())
        if len(low) >= n and " ".join(low[-n:]) == phrase:
            return " ".join(words[:-n]).strip()
    if low and low[-1] in OVER_FUZZ:
        return " ".join(words[:-1]).strip()
    return " ".join(words).strip()


# How close the first two words must look to a wake phrase to be treated as a
# mangled one. The free decoder has 200k words to choose from, so it mishears
# far more often than the grammar, which can only pick between four phrases and
# therefore matched perfectly. Observed in this room: "hey tech" came back as
# "hate tech", and "hey tek" as "we tank" - both were then handed on as part of
# the command.
#
# A list of known variants was the first attempt and it does not converge; each
# new mishearing needs a new entry. Measured similarity separates the two cases
# on its own:
#
#   mishearings   hey tec 0.93  hey tak 0.86  okay tek 0.86  hate tech 0.82
#                 hi tech 0.80  a tech 0.77   we tank 0.57
#   real openings what time 0.50  is the 0.50  how are 0.46  turn the 0.43
#                 play some 0.38  we should 0.35  tell me 0.31  what day 0.27
#
# 0.55 splits them. The margin at the bottom is thin (0.57 against 0.50), and
# both ways of being wrong are mild: a bad strip drops two words the brain can
# usually infer, a missed strip leaves noise it is already told to read
# through.
_WAKE_FUZZ = 0.55


def wake_only(text):
    """Was that JUST the wake word, with no command attached?

    Needed because the free decoder mangles the wake word, and a mangled wake
    word does not strip. Observed: someone said "hey tek" and nothing else,
    the decoder wrote "hate tech", strip_wake left it alone because there was
    no command after it to strip from - and "hate tech" was then dispatched as
    the question. It answered as though the speaker had announced they dislike
    technology, which is a memorable way to be wrong.
    """
    import difflib
    t = (text or "").strip().strip('"').lower()
    if not t:
        return True
    if any(t == w or t.rstrip(".,!?") == w for w in WAKE_WORDS):
        return True
    return max(difflib.SequenceMatcher(None, t, w).ratio()
               for w in WAKE_WORDS) >= _WAKE_FUZZ


def strip_wake(text):
    """Remove the wake word from the front of a command.

    "hey tek what time is it" -> "what time is it". Without this the wake word
    ends up in whatever the request is passed to, which reads oddly and wastes
    context.

    The exact match is tried first. It is not enough on its own: this is only
    ever called when the WAKE GRAMMAR has already fired, so the wake word was
    definitely said - but the free decoder that produced `text` may have
    written it down as something else. So a near miss is stripped too, on the
    evidence of the token that follows.
    """
    t = (text or "").strip()
    low = t.lower()
    for w in sorted(WAKE_WORDS, key=len, reverse=True):
        if low.startswith(w):
            return t[len(w):].strip(" ,.")

    import difflib
    words = t.split()
    if len(words) < 3:
        # Two words or fewer that did not match exactly: there is no command
        # in there to rescue, and stripping would leave nothing.
        return t
    head = " ".join(w.lower().strip(",.!?") for w in words[:2])
    if max(difflib.SequenceMatcher(None, head, w).ratio()
           for w in WAKE_WORDS) >= _WAKE_FUZZ:
        return " ".join(words[2:]).strip(" ,.")
    return t
