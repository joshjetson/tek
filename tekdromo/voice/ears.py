# -*- coding: utf-8 -*-
"""
Listening: a microphone, continuously, gated by a wake word.

    mic -> Gate -> VAD -> wake grammar -> free decode -> event

The design decisions that matter here are all about what this must NOT do.

**It must not hear itself.** Measured on this box: the mic picks up the
Bluetooth speaker at ~11x the ambient level, and Vosk transcribes Piper's
output *perfectly* - the acoustic loopback test reads back all twelve keywords.
So a face that listens while it talks will answer its own replies, forever.
`Gate` feeds silence to the segmenter while the service is speaking, plus a
tail for Bluetooth latency and room reverb.

**It must not transcribe the household.** The wake grammar runs on everything
because it has to, but it is a four-phrase grammar that can only emit those
phrases or "[unk]" - it cannot produce a transcript. Full decoding happens only
after the wake word matches. That is the privacy posture the project chose:
local only, wake-word gated, nothing leaves the house.

**It must survive the microphone vanishing.** The mic is built into the camera,
so every camera replug takes the mic with it - and a replug is now a routine
event rather than a fault (see camera.py). parec dies, the source ends, and
this reopens it. A listener that goes permanently deaf after someone bumps a
USB plug is the same defect the camera just had.

**One model, two recognisers.** Vosk supports grammar-constrained decoding, so
wake spotting costs about a ninth of real time while a free decode costs most
of it. Running the cheap one always and the expensive one rarely is what makes
continuous listening affordable on four A57 cores that are already drawing a
face at 30 fps.
"""
import collections
import os
import threading
import time
import traceback

import numpy as np

from . import bargein, io as vio, pcm, stt

# How long after the wake word to wait for a command, if the wake word arrived
# on its own. "Hey tek" ... "what time is it" is a normal way to speak; so is
# saying both in one breath, which is handled without waiting at all.
WAKE_WINDOW = 8.0

# After it finishes speaking, keep listening this long WITHOUT the wake word.
# People do not say "hey tek" before every sentence of a conversation, and
# making them is most of why this felt like operating a machine rather than
# talking to something. Short enough that the room's ordinary chatter a minute
# later is not taken as addressed to it.
FOLLOWUP_S = 12.0

# How many follow-ups in a row before the wake word is required again.
#
# Without a cap this is a self-sustaining loop, and it was observed being one:
# a single wake at 21:48:37 produced nine "commands" over seven minutes, none
# of them addressed to the device. Every reply refreshes gate.spoke_at, which
# re-opens the window, which accepts the next thing said, which produces
# another reply. Anybody talking in the room keeps it alive indefinitely, and
# from the sofa that reads as "the mic is too sensitive" - the microphone is
# fine, the exit condition is missing.
#
# Three is a real conversation - a question, a follow-up, and a clarification -
# and short enough that a runaway costs three exchanges rather than an evening.
FOLLOWUP_MAX_TURNS = 3

# A follow-up has to be SAID TO IT, not merely audible. Measured with
# tools/wake_tune.py: deliberate speech at a normal distance peaks 0.49-0.75,
# and the junk that got through this window peaked far below that. This is the
# same argument as the barge-in proximity gate - in a house there is nearly
# always a voice somewhere, and "audible" is not the question.
#
# Only applied to FOLLOW-UPS. A command after an explicit wake word has already
# proved it was addressed here, and holding it to a level bar as well would
# make the device ignore somebody who had just said its name.
FOLLOWUP_MIN_PEAK = 0.20

# Junk transcripts to refuse outright. "the", "i went what" and "right away
# from me again" were all dispatched as questions. A follow-up worth answering
# is at least a few words, because the free decoder emits fragments from room
# noise and each one costs a model call and an unprompted reply.
FOLLOWUP_MIN_WORDS = 3

# Silence fed to the segmenter after the last sample is written, covering the
# speaker's own buffer and the room's reverb tail. A2DP alone is 150-250 ms.
SPEAK_TAIL = 1.2

# The display writes this from live camera frames. If it is fresh, the brain
# gets to see who is talking to it.
SNAPSHOT = os.path.expanduser("~/.cache/tekdromo/seen.jpg")
SNAPSHOT_FRESH = 20.0

RETRY_S = 2.0

# How often to check that the ear is listening to something real, and how long
# without any audio counts as deaf. Both are cheap: the check reuses a cached
# probe, and a live source delivers frames continuously even in a silent room -
# so "no frames at all" is a genuine fault signal, not just quiet.
CHECK_S = 20.0
DEAF_S = 15.0
# How long a "which input actually works" answer is reused. Long, because the
# cache is invalidated the moment the set of capture devices changes, which is
# the only thing that can alter the answer - so the periodic check costs one
# `pactl list`, not a recorder opened on every candidate.
CACHE_S = 900.0

# Capture gain asserted whenever the microphone is opened. 100% is 0 dB on
# this card - full hardware gain, no software amplification - and measured
# ambient at that setting peaks around 0.017 with no clipped frames at all,
# so there is headroom to spare. The device was found at 63%.
MIC_GAIN_PCT = 100


# What a wake word that WORKS looks like at the microphone, measured with
# tools/wake_tune.py rather than assumed. Six consecutive successful attempts
# at a normal speaking distance peaked between 0.49 and 0.75, while a
# real-world failure the same evening peaked at 0.205 - about three times
# quieter, and the only difference between them.
#
# The hint used to fire below 0.05, which is essentially "the microphone is
# dead" and useless as advice: at 0.205 it said nothing at all, so the one
# person who could have fixed it by stepping closer had no idea that was the
# problem. Recognition does not fail at a cliff, so neither does the hint.
WAKE_PEAK_GOOD = 0.45
WAKE_PEAK_WEAK = 0.20


def _level_hint(peak):
    """Advice a person can act on, attached to the miss it explains."""
    if peak < WAKE_PEAK_WEAK:
        return "  <- far too quiet; much closer, or speak up"
    if peak < WAKE_PEAK_GOOD:
        return "  <- quiet; attempts that work here peak above %.2f" % WAKE_PEAK_GOOD
    return ""


class Gate(vio.Source):
    """Passes the microphone through, except while the face is speaking.

    Frames are still READ and thrown away rather than simply not read: parec
    keeps producing whether or not anyone is listening, and a reader that
    stalls just builds a backlog which arrives later as a burst of stale audio.

    Muted frames are also where BARGE-IN is decided. The frames are already
    being read and thrown away, so handing each one to the detector on its way
    to the bin costs nothing extra and is the only place in the system that
    holds mic audio captured while the speaker is playing.
    """

    # How long the gate stays open after a barge-in, regardless of `speaking`
    # or the tail. Without it the interrupted reply's last moments re-arm the
    # tail and the person's next 1.2 s - the words they interrupted WITH - are
    # fed to the segmenter as silence, so they have to say it twice.
    BARGE_OPEN_S = 6.0

    def __init__(self, source, service, tail=SPEAK_TAIL):
        self.source = source
        self.service = service
        self.tail = tail
        self.until = 0.0
        self.spoke_at = 0.0
        self.muted = 0
        self.frames = 0
        self.barges = 0
        self.open_until = 0.0
        self._barged_det = None       # the Detector we interrupted, if any
        self.ambient = None           # rms of the room with nothing playing
        self.while_speaking = None    # rms of the room while the speaker plays
        self._seen = None             # the detector we last calibrated
        self.last_at = time.monotonic()

    def read(self):
        frame = self.source.read()
        if frame is None:
            return None
        self.frames += 1
        now = time.monotonic()
        self.last_at = now

        # An interruption already happened: stay open and let the words the
        # person is still saying reach the segmenter.
        #
        # But ONLY for the utterance that was interrupted. If the service has
        # started speaking again, the window is over regardless of the clock -
        # otherwise the gate is held open across a NEW reply and the ear
        # transcribes the speaker, which is the one thing it must never do.
        # Observed exactly once, and it is unmistakable in the log:
        #
        #   interrupted (barge-in) ...
        #   ears: heard 'as say bitter and nothing else number one say it now'
        #
        # That is TEK reading its own prompt back to itself. Detected by
        # identity, not by time: service.barge is a fresh Detector per
        # utterance, so a different object means a different reply.
        if now < self.open_until:
            if (getattr(self.service, "speaking", False)
                    and getattr(self.service, "barge", None)
                    is not self._barged_det):
                self.open_until = 0.0
                self._barged_det = None
            else:
                return frame

        if getattr(self.service, "speaking", False):
            self.until = now + self.tail
            self.spoke_at = now
            # The other half of the acoustic health check - see heard_ratio().
            r = float(np.sqrt(np.mean(pcm.to_float(np.asarray(frame)) ** 2)))
            self.while_speaking = r if self.while_speaking is None else (
                0.98 * self.while_speaking + 0.02 * r)
            det = getattr(self.service, "barge", None)
            if det is not None:
                # Tell it what this room sounds like. The Gate is the only
                # thing that sees mic frames while NOTHING is being said, so
                # it is the only place a true ambient level can be measured -
                # and a detector calibrated on a constant instead of on the
                # room is a detector that never fires. It did not, for exactly
                # this reason: a hardcoded 0.012 against a mic reading 0.0044.
                if det is not self._seen and self.ambient:
                    det.min_level = max(bargein.MIN_LEVEL_ABS,
                                        self.ambient * bargein.AMBIENT_MULT)
                    self._seen = det
                try:
                    if det.feed(frame, now):
                        self.service.interrupt("barge-in")
                        self.barges += 1
                        self.until = 0.0
                        self.open_until = now + self.BARGE_OPEN_S
                        self._barged_det = det
                        return frame
                except Exception:
                    # A detector fault must not deafen the ear. Falling back to
                    # the old behaviour - mute while speaking - is exactly what
                    # this did before barge-in existed, so the failure mode is
                    # a previous version rather than a broken one.
                    pass

        if now < self.until:
            self.muted += 1
            return pcm.silence()

        # Not speaking, not muted: this frame IS the room. A slow EMA, because
        # ambient is a property of the house rather than of the moment, and one
        # door slamming must not recalibrate the detector.
        r = float(np.sqrt(np.mean(pcm.to_float(np.asarray(frame)) ** 2)))
        self.ambient = r if self.ambient is None else (
            0.995 * self.ambient + 0.005 * r)
        return frame

    def heard_ratio(self):
        """How much louder the room is while speaking. None until both are known.

        This exists because the acoustic path failing is INVISIBLE otherwise.
        The speaker stayed paired, A2DP stayed up, PulseAudio kept accepting
        audio and the monitor kept showing signal - every software layer
        reported healthy - while nothing was actually coming out of the
        speaker. It took a hand-built experiment to find, and it silently
        disables barge-in and every acoustic test in tools/.

        This box measured ~11x when it was working. A ratio near 1.0 means the
        microphone cannot hear the speaker: it is powered off, turned down, or
        has been moved.
        """
        if not self.ambient or not self.while_speaking:
            return None
        return self.while_speaking / max(self.ambient, 1e-9)

    def close(self):
        self.source.close()


class Draining(vio.Source):
    """Reads the microphone in its own thread so nothing downstream can stall it.

    This is the difference between a conversation and a device that ignores
    you. Recognition is NOT faster than real time on this board - measured on
    the small model:

        "hey tek"                            0.7s audio -> 1.47s to decode (2.10x)
        "what is the capital of france"      1.8s audio -> 2.38s (1.34x)
        "explain how a rainbow works ..."    2.5s audio -> 2.51s (0.99x)

    and parec's pipe holds 65536 bytes, which is 2.05 seconds. So decoding
    inline meant that for the whole time it was thinking about what you just
    said, it was deaf - and the buffer holding your next sentence overflowed
    while it worked. That produced exactly the reported symptoms: nothing heard
    right after the wake word, replies missed at random, and then an answer to
    something said minutes earlier arriving out of nowhere, because what
    finally got decoded was stale audio.

    A bounded deque drops the OLDEST frames when the consumer falls hopelessly
    behind. Losing old audio is right: stale speech answered late is worse than
    silence, which is the failure this replaces.
    """

    def __init__(self, source, seconds=12.0):
        self.source = source
        self.max = max(8, int(seconds * 1000 / pcm.FRAME_MS))
        self.q = collections.deque(maxlen=self.max)
        self.dropped = 0
        self.deepest = 0
        self._run = True
        self._ended = False
        self._t = threading.Thread(target=self._pump)
        self._t.daemon = True
        self._t.start()

    def _pump(self):
        try:
            while self._run:
                f = self.source.read()
                if f is None:
                    break
                if len(self.q) == self.max:
                    self.dropped += 1
                self.q.append(f)
                self.deepest = max(self.deepest, len(self.q))
        finally:
            self._ended = True

    def read(self):
        while self._run:
            try:
                return self.q.popleft()
            except IndexError:
                if self._ended:
                    return None
                time.sleep(0.005)
        return None

    def backlog(self):
        """Seconds of audio waiting to be processed."""
        return len(self.q) * pcm.FRAME_MS / 1000.0

    def close(self):
        self._run = False
        try:
            self.source.close()
        except Exception:
            pass


class Ears(object):
    """Continuous listening for a VoiceService.

        ears = Ears(service).start()

    Calls `service.on_event({"kind": "speech", "heard": ...})` when someone
    addresses it. Deciding and replying are the service's business; this only
    decides whether something was said TO us.
    """

    def __init__(self, service, device=None, window=WAKE_WINDOW):
        self.service = service
        self.device = device
        self.window = window
        self._run = False
        self._t = None
        # Two recognisers over ONE model (stt.model is cached), so the second
        # one costs memory for its decoder and nothing for the acoustic model.
        self.wake = None
        self.free = None
        self.seg = None
        self.armed_until = 0.0
        # Consecutive commands taken WITHOUT a wake word. Reset by a real one.
        self.followups = 0
        self.utterances = 0
        self.wakes = 0
        self.commands = 0
        self.opens = 0
        self.listening_since = 0.0
        self.last_heard = None
        self.device_in_use = None
        self.misses = []
        self._src = None
        self._gate = None
        self._drain = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop)
        self._t.daemon = True
        self._t.start()
        w = threading.Thread(target=self._watch)
        w.daemon = True
        w.start()
        return self

    def stop(self):
        self._run = False

    def alive(self):
        return self._t is not None and self._t.is_alive()

    def _build(self):
        """Recognisers and segmenter, built once, lazily.

        Lazily because loading Vosk costs a second or two and the service must
        answer `tek say` immediately at startup - the ear can take its time.
        """
        from . import vad
        if self.wake is None:
            self.wake = stt.Recogniser(grammar=stt.WAKE_GRAMMAR)
            self.free = stt.Recogniser()
            self.seg = vad.Segmenter()
            print("ears: wake words %s" % (stt.WAKE_WORDS,), flush=True)

    # -- worker ------------------------------------------------------------
    def _loop(self):
        """Listen forever, reopening the microphone whenever it disappears.

        Nothing but stop() may end this thread. The camera's worker had a bare
        `except Exception: pass` around exactly this shape of loop and it made
        face tracking die silently for good; there is no reason to repeat that
        here, especially when the microphone is physically part of that same
        camera.
        """
        try:
            self._build()
        except Exception:
            traceback.print_exc()
            print("ears: no recogniser - listening disabled", flush=True)
            return
        while self._run:
            src = None
            try:
                # NOT simply the PulseAudio default. The microphone is built
                # into the webcam, so a camera replug takes the source down and
                # PulseAudio moves the default to the Tegra onboard input,
                # which has nothing plugged into it - and it does not move back
                # when the camera returns. That was observed: the ear reported
                # itself healthy, with a recorder open, hearing nothing at all,
                # because it was listening to a dead device. An ear that cannot
                # tell "silent room" from "wrong device" is the worst kind of
                # broken, so the device is probed rather than assumed.
                device = self.device or vio.working_source()
                if not device:
                    print("ears: no working microphone; retrying", flush=True)
                    time.sleep(RETRY_S)
                    continue
                # Assert the gain before opening. The C922 was sitting at
                # 63% / -12 dB, which throws away a factor of four in
                # amplitude before the recogniser sees anything - and wake
                # spotting is measurably level-sensitive at the bottom end.
                vio.set_source_volume(device, MIC_GAIN_PCT)
                src = vio.MicSource(device)
                self.device_in_use = device
                self._src = src
                self.opens += 1
                self.listening_since = time.time()
                print("ears: listening to %s (mic open #%d)"
                      % (device, self.opens), flush=True)
                # Gate FIRST, then drain: the mute decision has to be made
                # when the audio is captured, not when it is finally read, or
                # everything buffered during a reply comes out unmuted.
                self._gate = Gate(src, self.service)
                self._drain = Draining(self._gate)
                for utt in self.seg.segments(self._drain):
                    if not self._run:
                        break
                    self._utterance(utt)
                # segments() ended: the source returned None, which for a live
                # mic means parec died - usually because the device went away.
                if self._run:
                    print("ears: microphone stream ended, reopening", flush=True)
            except Exception:
                traceback.print_exc()
            finally:
                if self._drain is not None:
                    try:
                        self._drain.close()
                    except Exception:
                        pass
                self._src, self._gate, self._drain = None, None, None
                if src is not None:
                    try:
                        src.close()
                    except Exception:
                        pass
            if self._run:
                time.sleep(RETRY_S)

    def _watch(self):
        """Force a reconnect when the ear is listening to the wrong thing.

        Needed because neither fault ends the stream on its own, so the reader
        sits happily blocked forever:

        * **Wrong device.** The default capture source moves to the dead
          onboard input whenever the webcam is replugged, and stays there. The
          only way to notice is to go and check which source actually works.
        * **No frames.** A source that stops delivering without closing looks
          exactly like a silent room from inside the read.

        Closing the source is what breaks the reader loose: parec dies, the
        pipe closes, read() returns None, and the reader re-probes.
        """
        while self._run:
            time.sleep(CHECK_S)
            src, gate = self._src, self._gate
            if src is None:
                continue
            why = None
            if gate is not None and time.monotonic() - gate.last_at > DEAF_S:
                why = "no audio for %.0fs" % DEAF_S
            elif self.device is None:
                try:
                    best = vio.working_source(ttl=CACHE_S)
                    if best and best != self.device_in_use:
                        why = "%s works and %s does not" % (best,
                                                            self.device_in_use)
                except Exception:
                    pass
            if why:
                print("ears: %s - reopening the microphone" % why, flush=True)
                try:
                    src.close()
                except Exception:
                    pass

    def _utterance(self, samples):
        """One segment of speech. Decide whether it was addressed to us."""
        self.utterances += 1
        secs = len(samples) / float(pcm.RATE)

        # Just answered someone? Then the next thing said is almost certainly
        # a reply, and demanding the wake word again before every single
        # sentence is the main reason this did not feel like a conversation.
        gate = self._gate
        peak = float(np.abs(samples.astype(np.int32)).max()) / 32768.0
        if (gate is not None and gate.spoke_at
                and time.monotonic() - gate.spoke_at < FOLLOWUP_S
                and time.monotonic() >= self.armed_until
                and self.followups < FOLLOWUP_MAX_TURNS):
            self.armed_until = time.monotonic() + self.window
            self._auto_armed = True

        if time.monotonic() < self.armed_until:
            # Already woken: this is the command, whatever it is.
            text = self.free.transcribe(samples)
            # Saying the wake word twice is a normal thing to do when the
            # first one seemed to go unheard. Dispatching the second one as
            # the question gets an answer to "hey tek", which is nonsense.
            auto = getattr(self, "_auto_armed", False)
            words = len((text or "").split())
            if text and not stt.wake_only(text) and auto and (
                    peak < FOLLOWUP_MIN_PEAK or words < FOLLOWUP_MIN_WORDS):
                # Audible, but not addressed here. Refused rather than
                # answered, and logged with the numbers so the thresholds can
                # be tuned from what a house actually does.
                print("ears: not addressed %r (peak %.3f, %d words) - ignoring"
                      % (text, peak, words), flush=True)
                self.armed_until = 0.0
                self._auto_armed = False
                return
            if text and not stt.wake_only(text):
                self.armed_until = 0.0
                if auto:
                    self.followups += 1
                    if self.followups >= FOLLOWUP_MAX_TURNS:
                        print("ears: %d follow-ups without a wake word - "
                              "say it again to carry on" % self.followups,
                              flush=True)
                self._auto_armed = False
                self._command(text, secs)
            else:
                # Do NOT disarm. The window belongs to the person, not to the
                # first scrap of audio that happens to land in it - and after a
                # reply the first thing segmented is often the tail of the
                # room settling, which decodes to nothing. Spending the arm on
                # that meant the actual follow-up arrived to a closed door,
                # which is precisely "sometimes it hears us, sometimes it
                # doesn't".
                print("ears: armed, ignoring %r - still listening for %.0fs"
                      % (text, self.armed_until - time.monotonic()), flush=True)
            return

        # The cheap pass. This grammar can only return the wake phrases or
        # "[unk]", so ordinary conversation cannot be transcribed by it.
        spotted = self.wake.transcribe(samples)
        if not stt.heard_wake(spotted):
            # Log the near miss. "Sometimes it says nothing" is impossible to
            # diagnose without knowing WHICH stage dropped it - the grammar is
            # permissive enough to wake on "hey deck" and "hey tex" when the
            # audio is clean, so a miss is nearly always level or segmentation,
            # not vocabulary. Recording the peak and duration alongside the
            # result is what tells those apart.
            #
            # Safe to log: this grammar is physically incapable of emitting
            # anything but its own phrases and "[unk]".
            peak = float(np.abs(samples.astype(np.int32)).max()) / 32768.0
            self.misses.append({"secs": round(secs, 2), "peak": round(peak, 4),
                                "got": spotted})
            del self.misses[:-12]
            # Only a NEAR miss is worth a log line. A bare "[unk]" is somebody
            # talking in the room and there are dozens an hour; printing those
            # buries the one line that matters. "hey [unk]" is different - the
            # head was heard and the tail was not, which is a wake attempt that
            # got away.
            if spotted and spotted != "[unk]":
                print("ears: NEAR MISS %r (%.1fs, peak %.3f)%s"
                      % (spotted, secs, peak, _level_hint(peak)), flush=True)
            return
        self.wakes += 1
        # An explicit wake word is the person saying "yes, I mean you". It
        # clears the follow-up budget so a real conversation is never cut off
        # by a cap that exists to stop an unattended one.
        self.followups = 0
        self._auto_armed = False

        # Was there anything AFTER the wake word? The grammar already knows:
        # it emits "hey tek" for the wake word alone and "hey tek [unk]" when
        # more was said. Asking it costs nothing and saves a free decode -
        # measured at 1.47s for a 0.7s utterance, paid at the exact moment
        # somebody has just said "hey tek" and is about to start talking. That
        # was most of "after the wake word it cannot hear us instantly".
        if "[unk]" not in spotted:
            self.armed_until = time.monotonic() + self.window
            print("ears: woken (%.1fs) - listening for a command" % secs,
                  flush=True)
            return

        # Something followed it in the same breath, which is how people
        # actually talk. Now the full decode is worth paying for.
        text = self.free.transcribe(samples)
        rest = stt.strip_wake(text)
        # If strip_wake removed something, what is left IS the command - do not
        # second-guess it. wake_only is only consulted when nothing was
        # stripped, which means the decoder produced a couple of words with no
        # recognisable wake phrase in front. That is nearly always a mangled
        # bare wake word ("hate tech" for "hey tek"), and dispatching it as a
        # question got an answer about disliking technology.
        #
        # Narrowing it to that case matters: on the whole string the fuzzy test
        # cannot separate "we tank" (0.57, a wake word) from "hey there" (0.75,
        # a greeting), so it must not be asked about strings where the answer
        # is already known.
        if rest and (rest != text.strip() or not stt.wake_only(rest)):
            self._command(rest, secs)
        else:
            self.armed_until = time.monotonic() + self.window
            print("ears: woken (%.1fs) - waiting %.0fs for a command"
                  % (secs, self.window), flush=True)

    def _command(self, text, secs=0.0):
        """Hand a heard command to the service's event pipeline."""
        self.commands += 1
        self.last_heard = text
        print("ears: heard %r (%.1fs)" % (text, secs), flush=True)
        ev = {"kind": "speech", "heard": text,
              "what": 'Someone spoke to you and said: "%s"' % text}
        # If the display has a recent frame, let the brain see who is talking.
        try:
            if time.time() - os.path.getmtime(SNAPSHOT) < SNAPSHOT_FRESH:
                ev["image"] = SNAPSHOT
        except OSError:
            pass
        try:
            self.service.on_event(ev)
        except Exception:
            traceback.print_exc()

    def state(self):
        return {"listening": self.alive(), "utterances": self.utterances,
                "wakes": self.wakes, "commands": self.commands,
                "opens": self.opens, "last_heard": self.last_heard,
                "device": self.device_in_use,
                "backlog": (round(self._drain.backlog(), 1)
                            if self._drain is not None else None),
                "dropped": (self._drain.dropped
                            if self._drain is not None else 0),
                "misses": list(self.misses[-6:]),
                "ambient": (round(self._gate.ambient, 5)
                            if self._gate is not None and self._gate.ambient
                            else None),
                "heard_ratio": (round(self._gate.heard_ratio(), 2)
                                if self._gate is not None
                                and self._gate.heard_ratio() else None),
                "barges": self._gate.barges if self._gate is not None else 0,
                "followups": self.followups,
                "armed": time.monotonic() < self.armed_until}
