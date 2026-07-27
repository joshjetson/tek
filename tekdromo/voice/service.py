"""
The voice service. Owns the voice, the speaker, and the mouth stream.

Why this is a service rather than something `tek say` does itself: loading the
Piper model costs ~4.6 s and synthesising costs ~0.7x the audio duration. A
per-invocation load would dominate every reply and make short answers feel
broken. The session is built once here and reused, so `tek say` is just a
socket write.

It is also the only thing that owns a Voice, so there is exactly one place
where "how do we speak" is decided - the CLI has no fallback logic of its own.

While speaking it publishes mouth frames derived from the SAME PCM that is
being played, so the face cannot drift out of step with the audio.
"""
import argparse
import os
import re
import threading
import time
import traceback

import numpy as np

from . import agent, bus, io as vio, pcm, tts

CONFIG = os.path.expanduser("~/.config/tekdromo/voice.json")


def load_choice():
    """The voice chosen by ear, if there is one.

    Kept outside the repo because it is a per-machine preference, not code -
    the same checkout on another box with different speakers may want a
    different voice.
    """
    try:
        import json
        with open(CONFIG) as f:
            return json.load(f).get("voice")
    except Exception:
        return None


def load_settings():
    """Per-machine tuning that is not code. Same file as the voice choice."""
    try:
        import json
        with open(CONFIG) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(update):
    import json
    cur = load_settings()
    cur.update(update)
    d = os.path.dirname(CONFIG)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(CONFIG, "w") as f:
        json.dump(cur, f, indent=2, sort_keys=True)


def save_choice(name):
    # Merge, do not overwrite: writing {"voice": ...} on its own silently threw
    # away every keepalive setting the moment the voice was changed.
    save_settings({"voice": name})

# Chunking for streaming synthesis.
#
# Chunk sizes RAMP. The first is short because it alone sets how long the
# listener waits for any sound; later ones grow because bigger units give
# Piper better prosody, and by then the producer is far ahead.
#
# The ramp is not cosmetic. Synthesis runs at ~0.71x real-time, so each second
# of playback buys only 0.4 s of lead - which means the danger of starving is
# entirely at the START, before any lead has accumulated. Going straight from a
# 90-character chunk to a 260-character one made chunk 2 take ~8 s to
# synthesise while chunk 1 bought only ~4.5 s of audio, and playback caught up:
# three silences of 1.3-1.7 s in a 72 s reply.
_SENT = re.compile(r'(?<=[.!?])\s+')
# Each chunk at most ~1.3x the previous. That ratio is not arbitrary:
# synthesis runs at ~0.75x real-time, so while chunk k+1 is being made, chunk k
# is being played, and the producer stays ahead as long as
# 0.75 * a(k+1) <= a(k) - i.e. a(k+1) <= 1.33 * a(k).
# The FIRST entry sets how long you wait for the first word, so it is kept
# small; later ones grow because bigger units give Piper better prosody and
# by then the producer is comfortably ahead.
CHUNK_RAMP = [55, 75, 100, 135, 180, 240, 300]
# Max growth from one chunk to the next, from the 0.75x synthesis rate:
# 0.75 * a(k+1) <= a(k) would allow 1.33, but the rate is NOT constant: it was
# measured between 0.73x and 0.98x depending on what else the board is doing,
# and at 0.98x the safe growth is barely 1.0. 1.15 holds up under contention;
# 1.30 put two gaps in a long sentence.
GROWTH = 1.15
# Largest atom. Small enough that the packer can hit any limit closely.
ATOM = 45
# Never start speaking on less than this much buffered audio. With the ramp
# above this is what actually guarantees the producer stays ahead for the rest
# of the reply, at the cost of ~1 s more before the first word.
# Only enough to cover the first chunk boundary. It was 5 seconds, which for a
# reply shorter than about six seconds meant waiting for essentially the whole
# thing: a three-sentence answer took 6.7s to say its first word while total
# synthesis was 6.3s.
#
# A big lead is not what prevents starvation - the RATIO is. Synthesis at 0.75x
# real-time produces faster than playback consumes, so once started, playback
# cannot catch up; the only risk is the first boundary, before any lead has
# accumulated, and the chunk ramp above handles that.
MIN_LEAD_S = 1.5

# Keeping the Bluetooth speaker awake.
#
# Unloading module-suspend-on-idle stops PULSEAUDIO suspending the sink, but
# the SPEAKER has its own idle timer and powers itself off. Something has to be
# played, and it has to be something the speaker's detector actually counts.
#
# The first attempt used 40 Hz, chosen precisely BECAUSE a portable driver
# cannot reproduce it - which was self-defeating. A speaker's auto-off detector
# works on the same post-filter signal path as its amplifier, so a tone it
# cannot reproduce is a tone it cannot detect either. "Inaudible because
# unreproducible" and "invisible to the silence detector" are the same
# property. It sent 34 tones over three hours and the speaker still switched
# off.
#
# So: a frequency the speaker genuinely reproduces, kept brief and quiet
# instead. Every parameter is tunable at runtime and persisted, because the
# right values depend on the specific speaker and only listening can settle
# them.
KEEPALIVE_HZ = 200.0          # well inside any speaker's range
KEEPALIVE_AMP = 0.02          # -34 dBFS: quiet, but unmistakably signal
KEEPALIVE_SECS = 0.25         # brief enough not to read as a "sound"
KEEPALIVE_EVERY = 120.0       # only if nothing else has been played


_CLAUSE_SPLIT = re.compile(r'(?<=[,;:])\s+')
# A sentence end, with any closing quote or bracket, followed by a space. Used
# to find the last point in a partial reply that is safe to start speaking.
_SENT_END = re.compile(r'[.!?][\"\')\]]?\s')


# Shortest prefix worth speaking on its own. Deliberately small: "It's Monday."
# is a complete answer and waiting for 45 more characters that may never come
# would throw away the whole point of streaming. It is not smaller because a
# two-word fragment on its own sounds clipped. Consecutive short sentences do
# not each become a chunk - _speakable takes the LAST sentence end in the
# buffer, so "Oh. Hello there." goes out as one piece.
MIN_SPEAK = 12


def _speakable(buf, limit):
    """How much of a partly-arrived reply can be spoken now. 0 if none.

    A completed sentence is the natural cut - Piper's prosody depends on
    getting whole sentences, and cutting mid-clause is audible. Only if the
    buffer has grown past the chunk limit without one is a word boundary used
    instead, because at that point waiting costs more than the seam does.
    """
    last = None
    for m in _SENT_END.finditer(buf):
        if m.end() > limit:
            break            # a whole sentence, but too big to cut here yet
        last = m
    if last is not None and last.end() >= MIN_SPEAK:
        return last.end()
    if len(buf) >= limit:
        cut = buf.rfind(" ", 0, limit)
        return cut if cut >= MIN_SPEAK else limit
    return 0


def _stream_chunks(pieces, said):
    """Text arriving from a model -> speakable chunks, as early as possible.

    The non-streaming `_chunks` needs the whole reply before it can produce
    anything, so time-to-first-word grows with the length of the answer. That
    is exactly the wrong trade once answers are allowed to be substantial: a
    good long answer would be punished with a long silence in front of it.

    The chunk ramp is carried ACROSS pieces rather than restarted per
    sentence, so the same "grow slowly enough never to starve playback"
    guarantee that `_chunks` provides still holds - see CHUNK_RAMP.
    """
    buf = ""
    n, prev = 0, 0
    for piece in pieces:
        if not piece:
            continue
        buf += piece
        said["text"] += piece
        while True:
            # Same two constraints as _chunks: the ramp is the absolute size,
            # GROWTH is what actually prevents gaps. Cutting at whatever
            # sentence end happened to arrive ignored the ratio and produced
            # 32 characters followed by 83 - playback of the first runs out
            # long before the second is synthesised.
            ramp = CHUNK_RAMP[min(n, len(CHUNK_RAMP) - 1)]
            limit = ramp if not n else max(ATOM, min(ramp, int(prev * GROWTH)))
            cut = _speakable(buf, limit)
            if cut <= 0:
                break
            out, buf = buf[:cut].strip(), buf[cut:]
            if out:
                n, prev = n + 1, len(out)
                yield out
    for tail in _chunks(buf):          # whatever is left when the model stops
        n += 1
        yield tail


def _atoms(text):
    """Break text into the smallest pieces worth speaking together.

    Sentences first, then clauses, then words if a clause is still huge. Small
    atoms are what let the packer below honour a size limit exactly; handing it
    an 81-character piece when the limit is 45 means the limit is ignored, and
    that is how a 35-character chunk ended up followed by an 81-character one.
    """
    out = []
    for sent in _SENT.split((text or "").strip()):
        sent = sent.strip()
        if not sent:
            continue
        for clause in _CLAUSE_SPLIT.split(sent):
            clause = clause.strip()
            if not clause:
                continue
            while len(clause) > ATOM:
                cut = clause.rfind(" ", 0, ATOM)
                if cut <= 0:
                    break
                out.append(clause[:cut])
                clause = clause[cut + 1:]
            if clause:
                out.append(clause)
    return out


def _chunks(text):
    """Pack atoms into chunks that grow slowly enough to never starve playback.

    Two constraints, tighter wins. CHUNK_RAMP is the absolute size. GROWTH is
    what actually prevents gaps: synthesis at ~0.75x real-time means chunk k+1
    may be at most ~1.33x chunk k, or playback of chunk k ends before chunk k+1
    is ready. Ignoring the ratio put three one-second holes inside a single
    spoken sentence.
    """
    out, buf = [], ""

    def limit():
        ramp = CHUNK_RAMP[min(len(out), len(CHUNK_RAMP) - 1)]
        if not out:
            return ramp
        return max(ATOM, min(ramp, int(len(out[-1]) * GROWTH)))

    for a in _atoms(text):
        if buf and len(buf) + len(a) + 1 > limit():
            out.append(buf)
            buf = a
        else:
            buf = (buf + " " + a).strip()
    if buf:
        out.append(buf)
    return out


class VoiceService(object):

    def __init__(self, voice=None, device=None, path=bus.DEFAULT_PATH,
                 latency_trim=0.0, brain_model=None, cooldown=180.0,
                 keepalive=KEEPALIVE_EVERY):
        t0 = time.time()
        # An explicit --voice wins; otherwise use whatever was chosen by ear.
        self.voice = tts.load(voice or load_choice())
        # Alternates are loaded on demand and kept, so auditioning does not
        # pay the model load twice. Only Piper is heavy; the others are
        # subprocess wrappers costing almost nothing.
        self.voices = {self.voice.name: self.voice}
        self.device = device
        self.load_time = time.time() - t0
        self.spoken = 0
        self.speaking = False
        self._lock = threading.Lock()     # one utterance at a time
        # How far the mouth must lag the audio. Measured from PulseAudio, plus
        # a manual trim for the part it cannot see (a Bluetooth speaker's own
        # internal buffer is invisible from this side).
        self.latency = vio.sink_latency(device) + float(latency_trim)
        # Camera/mic events. The cooldown is the thing standing between an
        # ambient face and a device that talks over your evening - and between
        # you and an API bill, since every event that gets through costs a
        # model call. Deliberately generous by default.
        self.brain = agent.load(brain_model)
        self.watching = True
        # The ear, if there is a microphone. Separate switch from `watching`:
        # "stop watching me" and "stop listening to me" are different requests.
        self.listening = True
        self.ears = None
        self.mic = None               # PulseAudio source; None = the default
        self.cooldown = float(cooldown)
        self.last_event = 0.0
        self.last_spoke = 0.0
        self.recent = []
        self.events_seen = 0
        self.events_acted = 0
        cfg = load_settings()
        self.keepalive_every = float(cfg.get("keepalive_every", keepalive))
        self.keepalive_hz = float(cfg.get("keepalive_hz", KEEPALIVE_HZ))
        self.keepalive_amp = float(cfg.get("keepalive_amp", KEEPALIVE_AMP))
        self.keepalive_secs = float(cfg.get("keepalive_secs", KEEPALIVE_SECS))
        self.last_audio = time.time()
        self.keepalives = 0
        self.server = bus.Server(path, self.handle)
        print("voice=%s %dHz (loaded in %.2fs)  mouth lag %.0fms  socket=%s"
              % (self.voice.name, self.voice.rate, self.load_time,
                 self.latency * 1000, path), flush=True)

    # -- protocol ----------------------------------------------------------
    def handle(self, msg, conn):
        cmd = msg.get("cmd")
        if cmd == "say":
            text = (msg.get("text") or "").strip()
            if not text:
                return {"ok": False, "error": "empty text"}
            try:
                return self.say(text, wait=msg.get("wait", True),
                                voice=msg.get("voice"))
            except Exception as e:
                traceback.print_exc()
                return {"ok": False, "error": str(e)}
        if cmd == "set_voice":
            name = msg.get("voice")
            try:
                v = self._voice(name)
            except Exception as e:
                return {"ok": False, "error": str(e)}
            self.voice = v
            save_choice(v.name)
            print("voice set to %s (saved to %s)" % (v.name, CONFIG), flush=True)
            return {"ok": True, "voice": v.name, "rate": v.rate}
        if cmd == "keepalive":
            changed = False
            for key, attr, lo in (("every", "keepalive_every", 0.0),
                                  ("hz", "keepalive_hz", 1.0),
                                  ("amp", "keepalive_amp", 0.0),
                                  ("secs", "keepalive_secs", 0.02)):
                if key in msg and msg[key] is not None:
                    setattr(self, attr, max(lo, float(msg[key])))
                    changed = True
            if changed:
                save_settings({
                    "keepalive_every": self.keepalive_every,
                    "keepalive_hz": self.keepalive_hz,
                    "keepalive_amp": self.keepalive_amp,
                    "keepalive_secs": self.keepalive_secs})
            if msg.get("now"):
                self._keepalive()
            return {"ok": True, "every": self.keepalive_every,
                    "sent": self.keepalives,
                    "idle": round(time.time() - self.last_audio, 1),
                    "hz": self.keepalive_hz, "amp": self.keepalive_amp,
                    "secs": self.keepalive_secs}
        if cmd == "event":
            return self.on_event(msg.get("event") or {})
        if cmd == "watch":
            if "on" in msg:
                self.watching = bool(msg["on"])
            if "cooldown" in msg:
                self.cooldown = max(0.0, float(msg["cooldown"]))
            return {"ok": True, "watching": self.watching,
                    "cooldown": self.cooldown, "brain": self.brain.name,
                    "seen": self.events_seen, "acted": self.events_acted,
                    "next_in": max(0.0, round(
                        self.cooldown - (time.time() - self.last_event), 1))}
        if cmd == "latency":
            # Lets the lag be trimmed by ear without a restart.
            if "seconds" in msg:
                self.latency = max(0.0, float(msg["seconds"]))
            return {"ok": True, "latency": round(self.latency, 3)}
        # "ears", not "listen": `tek listen` already means "print the mouth
        # frames as they are published", and two different meanings for one
        # word in the same tool is a trap for whoever reads it next.
        if cmd == "ears":
            if "on" in msg:
                self.listening = bool(msg["on"])
                if self.listening and (self.ears is None
                                       or not self.ears.alive()):
                    self.start_ears()
                elif not self.listening and self.ears is not None:
                    self.ears.stop()
                    self.ears = None
            st = {"ok": True, "listening": self.listening,
                  "wake_words": None}
            try:
                from . import stt
                st["wake_words"] = stt.WAKE_WORDS
            except Exception:
                pass
            if self.ears is not None:
                st.update(self.ears.state())
            return st
        if cmd == "status":
            st = {"ok": True, "voice": self.voice.name,
                  "latency": round(self.latency, 3),
                  "rate": self.voice.rate, "speaking": self.speaking,
                  "spoken": self.spoken, "load_time": round(self.load_time, 2),
                  "watching": self.watching, "listening": self.listening}
            if self.ears is not None:
                st["ears"] = self.ears.state()
            return st
        if cmd == "ping":
            return {"ok": True}
        return {"ok": False, "error": "unknown command %r" % cmd}

    # -- keeping the speaker awake -----------------------------------------
    def _keepalive(self):
        """Play a brief inaudible tone so the speaker does not power down.

        Skipped entirely while speaking: the lock would serialise it behind the
        utterance anyway, and there is no point nudging a speaker that is
        already being driven.
        """
        if self.speaking:
            return False
        if not self._lock.acquire(False):
            return False
        try:
            rate = 44100          # the sink's native rate; no conversion
            sig = pcm.tone(self.keepalive_hz, self.keepalive_secs,
                           self.keepalive_amp, rate)
            sink = vio.SpeakerSink(device=self.device, rate=rate)
            try:
                for f in pcm.frames(sig, int(rate * pcm.FRAME_MS / 1000)):
                    sink.write(f)
            finally:
                sink.close()
            self.keepalives += 1
            self.last_audio = time.time()
            # Logged so a disconnect can be correlated against the last nudge.
            print("keepalive #%d (%.0fHz %.0fms at %.0f%%)"
                  % (self.keepalives, self.keepalive_hz,
                     self.keepalive_secs * 1000, self.keepalive_amp * 100),
                  flush=True)
            return True
        except Exception:
            traceback.print_exc()
            return False
        finally:
            self._lock.release()

    def _keepalive_loop(self):
        while True:
            time.sleep(5.0)
            if self.keepalive_every <= 0:
                continue
            if time.time() - self.last_audio >= self.keepalive_every:
                self._keepalive()

    # -- events ------------------------------------------------------------
    def on_event(self, ev):
        """A camera (later, a microphone) noticed something.

        Returns immediately. Deciding costs ~13 s of model call and speaking
        costs however long the sentence is, so neither may happen on the
        caller's connection - the display sends these, and the display must
        never wait for anything.
        """
        self.events_seen += 1
        now = time.time()
        kind = ev.get("kind")

        if kind == "speech":
            # Somebody used the wake word and asked something. This deliberately
            # skips the cooldown: that exists to stop the CAMERA remarking on an
            # ordinary evening, and applying it here would mean ignoring a
            # person who spoke directly to us - which reads as broken, not as
            # restraint. Its own switch, because "stop watching me" and "stop
            # listening to me" are different requests.
            if not self.listening:
                return {"ok": True, "acted": False, "reason": "listening is off"}
        else:
            if not self.watching:
                return {"ok": True, "acted": False, "reason": "watching is off"}
            if now - self.last_event < self.cooldown:
                return {"ok": True, "acted": False, "reason": "cooldown",
                        "next_in": round(self.cooldown - (now - self.last_event), 1)}
            # Departures are recorded but never spoken about: announcing that
            # someone has left, to an empty room, is talking to nobody.
            if kind == "departure":
                return {"ok": True, "acted": False, "reason": "departure"}
        self.last_event = now
        t = threading.Thread(target=self._consider, args=(ev,))
        t.daemon = True
        t.start()
        return {"ok": True, "acted": True, "considering": True}

    def _consider(self, ev):
        """Ask the Brain, and speak only if it decided to."""
        ev.setdefault("when", time.strftime("%A %H:%M"))
        ev["last_spoken_ago"] = (time.time() - self.last_spoke
                                 if self.last_spoke else None)
        ev["recent"] = list(self.recent)
        t0 = time.time()

        # A spoken question is answered AS IT IS WRITTEN. Anything else waits
        # for the whole reply first: a camera remark is one or two sentences,
        # so streaming would save nothing and only add ways to go wrong.
        if ev.get("kind") == "speech" and hasattr(self.brain, "stream"):
            try:
                if self._stream_reply(ev, t0):
                    return
            except Exception:
                traceback.print_exc()
            return

        try:
            words = self.brain.respond(ev)
        except Exception:
            traceback.print_exc()
            words = None
        took = time.time() - t0
        if not words:
            print("event %s: stayed quiet (%.1fs)" % (ev.get("kind"), took),
                  flush=True)
            return
        self.events_acted += 1
        self.recent.append(words)
        del self.recent[:-5]
        self.last_spoke = time.time()
        print("event %s: saying %r (decided in %.1fs)"
              % (ev.get("kind"), words[:60], took), flush=True)
        try:
            self._say(words)
        except Exception:
            traceback.print_exc()

    def _stream_reply(self, ev, t0):
        """Speak a reply as the model writes it. True if anything was said.

        The first chunk is what the whole exercise is for, so it is timed and
        logged: it is the number a person actually experiences, and it is now
        roughly independent of how long the answer turns out to be.
        """
        first = {"at": None}

        def timed():
            for piece in self.brain.stream(ev):
                if first["at"] is None:
                    first["at"] = time.time() - t0
                yield piece

        r = self._say(timed()) or {}
        words = (r.get("text") or "").strip()
        if not words:
            print("event speech: stayed quiet (%.1fs)" % (time.time() - t0),
                  flush=True)
            return False
        self.events_acted += 1
        self.recent.append(words)
        del self.recent[:-5]
        self.last_spoke = time.time()
        print("event speech: first word %.1fs, whole answer %.1fs, %d chars: %s"
              % (first["at"] or -1, time.time() - t0, len(words), words[:60]),
              flush=True)
        return bool(r and r.get("ok", True))

    def _voice(self, name):
        """A loaded Voice by name, cached."""
        if not name or name == self.voice.name:
            return self.voice
        if name not in self.voices:
            self.voices[name] = tts.load(name)
        return self.voices[name]

    # -- speaking ----------------------------------------------------------
    def say(self, text, wait=True, voice=None):
        if not wait:
            t = threading.Thread(target=self._say, args=(text, voice))
            t.daemon = True
            t.start()
            return {"ok": True, "queued": True}
        return self._say(text, voice)

    def _say(self, text, voice=None):
        """Speak, synthesising AHEAD of playback so there are no gaps.

        `text` is either a string, or an ITERABLE of pieces arriving from a
        model as it writes. The streaming form is what makes a deep answer
        affordable: chunking only what has arrived keeps time-to-first-word
        flat instead of growing with the length of the reply.

        Saying a long reply as one synth call means waiting for all of it
        before a sound comes out - ~85 s for a two-minute answer. Saying it as
        a series of separate calls is worse: each one opens a sink, plays, and
        closes it, so every sentence boundary becomes a silence the length of
        the NEXT sentence's synthesis. That is audible and it is what a
        listener describes as "too many breaks - it is not fluid".

        Piper runs at ~0.72x real-time, comfortably faster than speech. So one
        sink is opened for the whole reply, a producer thread stays ahead of
        it, and playback is continuous from the first sentence to the last.
        """
        with self._lock:
            v = self._voice(voice)
            said = {"text": ""}
            if isinstance(text, str):
                parts = _chunks(text)
                if not parts:
                    return {"ok": False, "error": "nothing to say"}
                said["text"] = text
            else:
                parts = _stream_chunks(text, said)

            rate = v.rate
            n = int(rate * pcm.FRAME_MS / 1000)
            frames = []                 # grows as synthesis proceeds
            envs = []
            done = threading.Event()
            state = {"synth": 0.0, "audio": 0.0, "rounding": 0.0, "error": None,
                     "parts": 0}

            def produce():
                try:
                    for part in parts:
                        state["parts"] += 1
                        t0 = time.time()
                        samples, _ = v.synth(part)
                        state["synth"] += time.time() - t0
                        if not len(samples):
                            continue
                        state["rounding"] = float(
                            getattr(v, "last_rounding", state["rounding"]))
                        for f in pcm.frames(np.asarray(samples), n):
                            frames.append(f)
                            envs.append(pcm.envelope(f))
                        state["audio"] = len(frames) * pcm.FRAME_MS / 1000.0
                except Exception as e:
                    state["error"] = str(e)
                    traceback.print_exc()
                finally:
                    done.set()

            pt = threading.Thread(target=produce)
            pt.daemon = True
            pt.start()

            # Build a head start before opening the speaker. Waiting only for
            # the first chunk is not enough: the lead has to be big enough that
            # the producer, running at 0.71x, never gets overtaken later.
            need = int(MIN_LEAD_S * 1000 / pcm.FRAME_MS)
            while len(frames) < need and not done.is_set():
                time.sleep(0.02)
            if not frames:
                return {"ok": False,
                        "error": state["error"] or "voice produced no audio"}

            self.speaking = True
            self.server.publish({"speaking": True, "text": said["text"]})
            sink = vio.SpeakerSink(device=self.device, rate=rate)

            # ONE sink for the entire reply. Opening a new one per sentence is
            # what put a gap at every boundary: pacat startup plus the A2DP
            # stream being torn down and rebuilt.
            def writer():
                i = 0
                try:
                    while True:
                        if i < len(frames):
                            sink.write(frames[i])
                            i += 1
                        elif done.is_set():
                            break
                        else:
                            time.sleep(0.01)   # producer is behind; wait
                finally:
                    sink.close()

            wt = threading.Thread(target=writer)
            wt.daemon = True
            wt.start()

            # The mouth runs on the wall clock - pacat's stdin has no
            # backpressure and cannot be used as an audio clock - offset by
            # however long the sound takes to emerge. A2DP is most of that.
            try:
                t_start = time.time() + self.latency
                step = pcm.FRAME_MS / 1000.0
                i = 0
                while True:
                    if i >= len(envs):
                        if done.is_set():
                            break
                        time.sleep(0.01)
                        continue
                    slack = (t_start + i * step) - time.time()
                    if slack > 0:
                        time.sleep(slack)
                    self.server.publish({"mouth": [round(envs[i], 4),
                                                   round(state["rounding"], 3)]})
                    i += 1
            finally:
                wt.join(timeout=state["audio"] + 30)
                self.speaking = False
                # Close the mouth BEFORE announcing speech ended: a consumer
                # that stops reading at speaking=False would otherwise never
                # receive the zero and would hold the last audio frame open.
                self.server.publish({"mouth": [0.0, 0.0]})
                self.server.publish({"speaking": False})

            self.spoken += 1
            self.last_spoke = self.last_audio = time.time()
            dur = state["audio"]
            print("said %.1fs in %.1fs (%.2fx, %d chunks) [%s] %s"
                  % (dur, state["synth"], state["synth"] / max(dur, 1e-6),
                     state["parts"], v.name,
                     said["text"][:44].replace("\n", " ")),
                  flush=True)
            return {"ok": True, "duration": round(dur, 2),
                    "synth": round(state["synth"], 2),
                    "chunks": state["parts"], "voice": v.name,
                    # The caller may have handed us a generator and so cannot
                    # know what was actually said until now.
                    "text": said["text"]}

    def start_ears(self, device=None):
        """Begin listening, if there is anything to listen with.

        Failure here is deliberately not fatal. A box with no microphone, or
        without the Vosk model downloaded, must still speak and still answer
        the camera - the ear is an addition, not a prerequisite.
        """
        try:
            from . import ears as _ears
            self.ears = _ears.Ears(self, device=device or self.mic).start()
            return True
        except Exception:
            traceback.print_exc()
            print("voice: no ear (continuing without one)", flush=True)
            return False

    def run(self):
        self.server.start()
        k = threading.Thread(target=self._keepalive_loop)
        k.daemon = True
        k.start()
        if self.listening:
            self.start_ears()
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tek-voice")
    ap.add_argument("--voice", default=None,
                    help="piper|pico|flite|espeak (default: best available)")
    ap.add_argument("--device", default=None, help="PulseAudio sink name")
    ap.add_argument("--socket", default=bus.DEFAULT_PATH)
    ap.add_argument("--brain-model", default=None,
                    help="model for camera/mic decisions. Faster is better "
                         "here: a greeting that lands 13s after someone walks "
                         "in reads as a malfunction.")
    ap.add_argument("--keepalive", type=float, default=KEEPALIVE_EVERY,
                    help="seconds of silence before nudging the Bluetooth "
                         "speaker with an inaudible tone. 0 disables it.")
    ap.add_argument("--cooldown", type=float, default=180.0,
                    help="minimum seconds between acting on camera events")
    ap.add_argument("--latency-trim", type=float, default=0.0,
                    help="extra seconds to delay the MOUTH behind the audio. "
                         "PulseAudio cannot see a Bluetooth speaker's own "
                         "buffer, so if the face still leads the sound, add it "
                         "here (e.g. 0.15).")
    ap.add_argument("--no-ears", action="store_true",
                    help="do not listen. The microphone is never opened at "
                         "all, which is a stronger statement than a flag that "
                         "merely ignores what it hears.")
    ap.add_argument("--mic", default=None,
                    help="PulseAudio SOURCE name. Default is whatever pactl "
                         "calls the default source.")
    ap.add_argument("--say", default=None,
                    help="speak once and exit, without starting the service")
    a = ap.parse_args(argv)

    if a.say:
        v = tts.load(a.voice)
        samples, rate = v.synth(a.say)
        # Deliberately NOT via ArraySource: that normalises to pcm.RATE, and
        # playing 16 kHz samples out of a sink opened at 22.05 kHz is a
        # chipmunk. The speaker is a hardware edge, so the voice's own rate
        # goes straight to it.
        sink = vio.SpeakerSink(device=a.device, rate=rate)
        for f in pcm.frames(np.asarray(samples), int(rate * pcm.FRAME_MS / 1000)):
            sink.write(f)
        sink.close()
        return 0
    svc = VoiceService(a.voice, a.device, a.socket, a.latency_trim,
                       a.brain_model, a.cooldown, a.keepalive)
    svc.listening = not a.no_ears
    svc.mic = a.mic
    svc.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
