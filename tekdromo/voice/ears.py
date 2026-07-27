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
import os
import threading
import time
import traceback

from . import io as vio, pcm, stt

# How long after the wake word to wait for a command, if the wake word arrived
# on its own. "Hey tek" ... "what time is it" is a normal way to speak; so is
# saying both in one breath, which is handled without waiting at all.
WAKE_WINDOW = 8.0

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


class Gate(vio.Source):
    """Passes the microphone through, except while the face is speaking.

    Frames are still READ and thrown away rather than simply not read: parec
    keeps producing whether or not anyone is listening, and a reader that
    stalls just builds a backlog which arrives later as a burst of stale audio.
    """

    def __init__(self, source, service, tail=SPEAK_TAIL):
        self.source = source
        self.service = service
        self.tail = tail
        self.until = 0.0
        self.muted = 0
        self.frames = 0
        self.last_at = time.monotonic()

    def read(self):
        frame = self.source.read()
        if frame is None:
            return None
        self.frames += 1
        self.last_at = time.monotonic()
        if getattr(self.service, "speaking", False):
            self.until = time.monotonic() + self.tail
        if time.monotonic() < self.until:
            self.muted += 1
            return pcm.silence()
        return frame

    def close(self):
        self.source.close()


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
        self.utterances = 0
        self.wakes = 0
        self.commands = 0
        self.opens = 0
        self.listening_since = 0.0
        self.last_heard = None
        self.device_in_use = None
        self._src = None
        self._gate = None

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
                src = vio.MicSource(device)
                self.device_in_use = device
                self._src = src
                self.opens += 1
                self.listening_since = time.time()
                print("ears: listening to %s (mic open #%d)"
                      % (device, self.opens), flush=True)
                self._gate = Gate(src, self.service)
                for utt in self.seg.segments(self._gate):
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
                self._src, self._gate = None, None
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

        if time.monotonic() < self.armed_until:
            # Already woken: this is the command, whatever it is.
            text = self.free.transcribe(samples)
            self.armed_until = 0.0
            if text:
                self._command(text, secs)
            else:
                print("ears: armed, but heard nothing usable", flush=True)
            return

        # The cheap pass. This grammar can only return the wake phrases or
        # "[unk]", so ordinary conversation cannot be transcribed by it.
        spotted = self.wake.transcribe(samples)
        if not stt.heard_wake(spotted):
            return
        self.wakes += 1

        # Woken - now a full decode of the SAME audio, in case the command came
        # in the same breath ("hey tek what time is it"), which is how people
        # actually talk.
        text = self.free.transcribe(samples)
        rest = stt.strip_wake(text)
        if rest:
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
                "armed": time.monotonic() < self.armed_until}
