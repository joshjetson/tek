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

# --- radio protocol --------------------------------------------------------
# "hey tek" opens the channel, "over" hands the turn back, "over and out"
# closes it. See the note in stt.py for why an explicit protocol beats every
# heuristic it replaces.
#
# While the channel is OPEN there is no level gate, no word-count gate and no
# follow-up budget: the person has said they are talking to it, so it listens
# until they say they have stopped. All of that machinery exists to guess at
# what this states outright.
#
# How long an open channel survives with nothing said. Long, because an open
# channel is a deliberate act and closing it behind somebody's back is exactly
# the "sometimes it hears us, sometimes it doesn't" this is meant to end - but
# not unbounded, because a missed "out" must not leave the mic live all night.
CHANNEL_IDLE_S = 180.0

# Utterances collected into one turn before it is sent regardless. A missed
# "over" should cost a slightly early answer, not a turn that never ends.
CHANNEL_MAX_PARTS = 8

# An implicit "over". Saying it is faster and certain, but REQUIRING it would
# make the device worse for anyone who has not learned the protocol, and a
# feature that only works if you know the magic word is one most of a household
# does not have. So silence this long with a turn held ends it anyway.
TURN_SILENCE_S = 3.5

# Vosk's own confidence, used to throw noise away before it becomes a question.
#
# Measured downstairs in a room with real background noise, which is where the
# numbers finally separated:
#
#   "there's a lot of background noise so just don't say anything for a while"
#                                     mean 0.99  min 0.86   <- a person
#   "cool"                            mean 0.84  min 0.84   <- a person
#   "why"                             mean 0.72  min 0.72   <- a person
#   "i thought level of or"           mean 0.51  min 0.26   <- noise
#   "well i yeah"                     mean 0.62  min 0.22   <- noise
#
# The MIN matters more than the mean and is the reason this works at all. A
# small model must emit real words for every sound it segments, so noise comes
# back as fluent, plausible English - but it comes back with one or two words
# it is quietly unsure of, and a genuine sentence does not. 0.86 against 0.26
# is not a close call.
#
# Deliberately not thresholded until there was data. Synthetic degradation
# never reproduced this: quiet alone still transcribes perfectly and
# quiet-plus-noise returns nothing, so only a real noisy room shows the middle
# case this exists for.
CONF_MEAN_MIN = 0.65
CONF_WORD_MIN = 0.40

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

# Capture gain asserted whenever the microphone is opened, overridable and
# persisted as "mic_gain" in ~/.config/tekdromo/voice.json.
#
# It was 100%, set on the evidence that AMBIENT peaked around 0.017 with no
# clipped frames - "headroom to spare". Ambient was the wrong thing to measure.
# Speech is what has to fit, and tools/wake_tune.py measured real wake words at
# this distance peaking 0.49-0.75, with the loudest at 0.747: just 2.5 dB below
# full scale. Lean in, or say something with a plosive in it, and it clips -
# and clipping damages recognition far more than being quiet does, because a
# flattened waveform is not merely fainter, it is a different sound.
#
# PulseAudio's percentage is CUBIC, not linear, and that changes the whole
# decision. Measured on this source:
#
#   100%  ->   0.00 dB   (1.00x amplitude)
#    75%  ->  -7.50 dB   (0.42x)
#    50%  -> -18.06 dB   (0.125x)
#
# So "halve it to 50%" would have been an EIGHT-fold amplitude cut, not a
# halving - and every dB taken off the speaker also comes off the distant
# speech that is already marginal enough to transcribe as mush. 75% is the
# setting that actually does what halving was meant to do: it takes real
# speech from a 0.747 peak to about 0.31, which is 10 dB of headroom instead
# of 2.5, without gutting the far end of the room.
#
# What this does NOT fix, stated plainly because it is the intuitive
# expectation: gain scales the voice and the room by the SAME factor, so
# signal-to-noise is unchanged and distant speech will still transcribe as
# mush. That is distance and reverb, not sensitivity. Nor does it quiet
# barge-in, whose thresholds are all multiples of measured ambient and
# therefore re-derive themselves at any gain.
MIC_GAIN_PCT = 75


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
        # Radio protocol: is the channel open, and what has been said into it
        # since the last "over".
        self.channel_open = False
        self.channel_at = 0.0
        self.parts = []
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
        tw = threading.Thread(target=self._turn_watch)
        tw.daemon = True
        tw.start()
        return self

    def stop(self):
        self._run = False

    def alive(self):
        return self._t is not None and self._t.is_alive()

    def _gain_pct(self):
        """Capture gain, from settings if somebody has tuned it."""
        try:
            from .service import load_settings
            return int(load_settings().get("mic_gain", MIC_GAIN_PCT))
        except Exception:
            return MIC_GAIN_PCT

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
                # Assert the gain before opening. The C922 was found at
                # 63% / -12 dB, which throws away amplitude before the
                # recogniser sees anything; it was then set to 100%, which
                # left real speech 2.5 dB from clipping. See MIC_GAIN_PCT.
                gain = self._gain_pct()
                vio.set_source_volume(device, gain)
                src = vio.MicSource(device)
                self.device_in_use = device
                self._src = src
                self.opens += 1
                self.listening_since = time.time()
                print("ears: gain %d%%" % gain, flush=True)
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

    def _turn_watch(self):
        """Flush a held turn when the person simply stops talking.

        Separate from _watch, which ticks every 20s - a turn boundary has to be
        noticed in seconds or the implicit "over" is useless. Cheap enough to
        run at 2Hz: it compares two floats.
        """
        while self._run:
            time.sleep(0.5)
            try:
                if not self.channel_open:
                    continue
                idle = time.monotonic() - self.channel_at
                if self.parts and idle > TURN_SILENCE_S:
                    print("ears: silence - taking that as OVER", flush=True)
                    self._send_turn()
                    continue
                # Closing an idle channel belongs HERE, not in _utterance.
                # It was in _utterance, which only runs when a sound arrives,
                # so a quiet room never reached it: one channel opened at
                # 22:57 and closed at 01:53, nearly three hours against a
                # 180-second setting, and only then because some noise
                # happened to land.
                #
                # That is a privacy bug rather than an untidiness. An open
                # channel free-decodes every utterance, so the wake-word gate
                # that is supposed to stop the household being transcribed was
                # off for the whole of those three hours. A timeout that only
                # fires when somebody speaks is not a timeout.
                if not self.parts and idle > CHANNEL_IDLE_S:
                    print("ears: channel idle %.0fs - closing" % idle,
                          flush=True)
                    self._close_channel()
            except Exception:
                pass

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

        # -- radio protocol: the channel is open -----------------------
        # No level gate, no word count, no follow-up budget. The person opened
        # the channel; everything until "over" is theirs. Those gates exist to
        # guess whether speech was addressed here, and an open channel is the
        # answer to that question, stated rather than inferred.
        if self.channel_open:
            # The idle timeout lives on _turn_watch's timer, not here - see
            # the note there. A timeout that only fires when somebody speaks
            # is not a timeout, and this is the branch that free-decodes.
            text = self.free.transcribe(samples)
            if not text:
                return
            # An open channel drops the level and word-count gates, because the
            # person said they were talking to it. It must NOT drop this one:
            # confidence is not asking "were you addressing me", it is asking
            # "were those even words". In a noisy room every fragment of
            # background became a turn, then a model call, then a reply, and
            # the replies queued up minutes deep.
            cf = getattr(self.free, "last_conf", None)
            cw = getattr(self.free, "last_min_conf", None)
            if cf is not None and (cf < CONF_MEAN_MIN or
                                   (cw is not None and cw < CONF_WORD_MIN)):
                print("ears: noise %r (conf %.2f/%.2f) - ignoring"
                      % (text, cf, cw if cw is not None else -1), flush=True)
                return

            # Refresh the idle timer ONLY for speech that passed the gate.
            #
            # It used to be refreshed on every utterance, before this check, so
            # rejected noise still held the channel open - and in a room with
            # background noise something arrives every few seconds forever. The
            # channel never went idle, never closed, and the wake word was
            # never needed again: reported as "Tek keeps talking even without
            # me saying hey Tek". The confidence gate was working the whole
            # time and was being undone by one line above it.
            self.channel_at = time.monotonic()
            if stt.ends_over_out(text):
                msg = stt.strip_over(text)
                if msg:
                    self.parts.append(msg)
                self._send_turn(secs, closing=True)
                return
            if stt.ends_over(text):
                msg = stt.strip_over(text)
                if msg:
                    self.parts.append(msg)
                self._send_turn(secs)
                return
            # Mid-turn: hold it and wait for the sign-off.
            self.parts.append(text)
            print("ears: holding %r (%d part%s, waiting for OVER)"
                  % (text, len(self.parts),
                     "" if len(self.parts) == 1 else "s"), flush=True)
            if len(self.parts) >= CHANNEL_MAX_PARTS:
                print("ears: %d parts without an OVER - answering anyway"
                      % len(self.parts), flush=True)
                self._send_turn(secs)
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
        if not self.channel_open:
            self._open_channel()

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
            # "hey tek what is the weather over" is ONE turn, not a wake word
            # followed by a separate command. It goes into the same buffer as
            # everything else so the sign-off decides when it is answered.
            if stt.ends_over_out(rest):
                self.parts.append(stt.strip_over(rest))
                self._send_turn(secs, closing=True)
            elif stt.ends_over(rest):
                self.parts.append(stt.strip_over(rest))
                self._send_turn(secs)
            else:
                self.parts.append(rest)
                self.channel_at = time.monotonic()
                print("ears: holding %r (waiting for OVER)" % rest, flush=True)
        else:
            self.armed_until = time.monotonic() + self.window
            print("ears: woken (%.1fs) - waiting %.0fs for a command"
                  % (secs, self.window), flush=True)

    def _open_channel(self):
        self.channel_open = True
        self.channel_at = time.monotonic()
        self.parts = []
        self.armed_until = 0.0
        print("ears: CHANNEL OPEN - say 'over' when you have finished",
              flush=True)
        try:
            self.service.channel_open = True
        except Exception:
            pass

    def _close_channel(self):
        self.channel_open = False
        self.parts = []
        try:
            self.service.channel_open = False
        except Exception:
            pass

    def _send_turn(self, secs=0.0, closing=False):
        """Hand the accumulated turn over, and hold the channel unless signed off."""
        text = " ".join(p for p in self.parts if p).strip()
        self.parts = []
        self.channel_at = time.monotonic()
        if closing:
            self._close_channel()
            print("ears: OVER AND OUT - channel closed", flush=True)
        if not text:
            if closing:
                self.service.say("Out.", wait=False)
            return
        self._command(text, secs,
                      getattr(self.free, "last_conf", None),
                      getattr(self.free, "last_min_conf", None),
                      closing=closing)

    def _command(self, text, secs=0.0, conf=None, min_conf=None, closing=False):
        """Hand a heard command to the service's event pipeline."""
        self.commands += 1
        self.last_heard = text
        print("ears: heard %r (%.1fs, conf %s/%s)"
              % (text, secs,
                 "-" if conf is None else "%.2f" % conf,
                 "-" if min_conf is None else "%.2f" % min_conf), flush=True)
        ev = {"kind": "speech", "heard": text,
              "conf": conf, "min_conf": min_conf,
              # Protocol turns get an immediate "copy that" and a closing
              # "over", so the channel is audibly half-duplex.
              "protocol": True, "closing": closing,
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
                "channel": "open" if self.channel_open else "closed",
                "parts_held": len(self.parts),
                "armed": time.monotonic() < self.armed_until}
