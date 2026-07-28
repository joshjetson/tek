# -*- coding: utf-8 -*-
"""
Barge-in: noticing that somebody is talking over the reply, and stopping.

`Gate` (ears.py) feeds silence to the segmenter while the face is speaking, plus
a 1.2 s tail. That is correct for self-hearing - the mic picks the speaker up at
~11x ambient and Vosk transcribes Piper perfectly, so without it TEK answers its
own replies forever. But it also means a 35-second answer cannot be interrupted,
which is the failure a listener notices on the first evening.

**This is a detector, not an echo canceller.** The distinction is the whole
design. Cancelling means reconstructing clean near-end speech, which over A2DP
means tracking a delay that wanders and a codec that is not linear - that is
what `module-echo-cancel` and WebRTC AEC try to do, and why the tail had to be
1.2 s in the first place. Answering "is a second voice present" needs none of
that. It needs one bit, and one bit survives a crude subtraction that leaves
plenty of residual echo behind.

    lag   cross-correlate mic against the outgoing signal, ONCE per utterance.
          Bluetooth delay is stable within an utterance and unstable between
          them, so this is the right granularity - per-frame tracking would be
          chasing noise, and a fixed constant would be wrong after every
          reconnect.
    sub   residual = mic - alpha * reference_aligned, alpha by least squares.
    hold  residual energy above threshold for >300ms -> somebody is talking.

The hold is what makes it usable. A door, a cough, a plosive and a codec
artefact are all short; a person talking is not. 300 ms of *sustained* residual
is the cheapest discriminator available and it costs one counter.
"""
import time

import numpy as np

from . import pcm

# How much audio to correlate when estimating the lag. Long enough to have
# structure in it, short enough that the estimate arrives before the person has
# finished their first word.
#
# 0.65 rather than the obvious 0.5, and that is not a rounding choice - it was
# swept, because at 0.5 the window at a long acoustic lag is mostly pre-arrival
# silence and the resulting lock is too loose to keep the residual floor down:
#
#   CORR_S   false stops (135)   missed (81)   median notice
#   0.50     0                   16            340 ms
#   0.65     0                    0            300 ms
#   0.80     0                    0            320 ms
#   1.00     0                   19            360 ms
#
# The far end is not a plateau either: at 1.0 s the window spans enough of the
# reply's own variation that a single least-squares gain no longer fits it, so
# the lock degrades again. 0.65 sits in the middle of the good range.
CORR_S = 0.65

# The A2DP latency the service already measured is a strong prior - the mouth
# loop runs on it and lip-sync measures correct. So the search is a window
# around it rather than a blind hunt over a second and a half of lag, which
# would be both slower and easier to fool with a spurious peak.
SEARCH_S = 0.35

# Below this correlation the lag estimate is not trustworthy - typically the
# reply has not started coming out of the speaker yet, or the room is loud
# enough to bury it. Detection stays OFF rather than running on a bad
# alignment, because a bad alignment makes the echo look like a second voice
# and stops the reply for no reason.
MIN_CORR = 0.15

# How far the residual must rise ABOVE ITS OWN FLOOR to count as a voice.
#
# Measuring the residual against the microphone was the first attempt and it is
# wrong in both directions at once. Under a loud reply the mic is dominated by
# echo, so a quiet interruption never clears the bar (19 of 81 missed); under a
# poor alignment the leftover echo IS the residual, so it clears it with nobody
# in the room (8 of 135 false stops). The two failures pull the threshold in
# opposite directions and no single value fixes both.
#
# The residual floor is the honest baseline. Crude cancellation leaves a
# consistent amount behind, whatever that amount happens to be; a second voice
# makes it jump. Measuring the rise instead of the level means the detector
# adapts to how well cancellation is going rather than assuming it goes well.
RESIDUAL_RISE = 2.6

# How long to spend measuring that floor after the lag locks, before any
# detection is allowed. Long enough to average over a few syllables of echo.
FLOOR_S = 0.35

# A lock is only accepted if the aligned reference actually explains the
# microphone - if subtracting it removes at least this much of the energy.
# This is the direct test of the thing that matters, and it catches what a
# correlation score alone does not: a confident peak in the wrong place. The
# -68.9ms estimate for a true 320ms lag scored corr=0.60 and cancelled nothing.
LOCK_CANCELS = 0.35

# Sustained for this long. The single most important parameter in the file:
# it is what separates a person from a cough, a door, or a codec artefact.
HOLD_S = 0.30

# Never trigger in the first moments of a reply. The lag estimate does not
# exist yet, and the speaker is still ramping - the two together make early
# frames the least trustworthy audio in the whole utterance.
WARMUP_S = 0.45

# The absolute floor, as a MULTIPLE OF MEASURED AMBIENT rather than a constant.
#
# It was a constant, 0.012, taken from the synthetic bench where the echo sits
# at 0.05-0.15 rms. On the real box the microphone reads 0.0044 rms, so that
# constant rejected 100% of real frames and the detector never once locked -
# it looked like it was working because every offline test passed. An absolute
# level is a statement about somebody else's room, somebody else's microphone
# and somebody else's speaker volume; on a project meant to run on other
# people's hardware it cannot be hardcoded.
#
# The Gate measures ambient continuously (it is the only thing that sees mic
# frames while nothing is being said) and hands it over. These multiples are
# then relative to whatever that room actually is.
AMBIENT_MULT = 1.8            # a frame must beat ambient by this to be signal
MIN_LEVEL_ABS = 1e-4          # a guard against an ambient estimate of zero
MIN_LEVEL = 0.012             # only the fallback when no ambient is supplied


class Reference(object):
    """The outgoing audio, at pcm.RATE, indexed by when it will be audible.

    A Sink in everything but name, and fed as one - the signal driving the
    speaker and the signal used for cancellation are one signal with two
    consumers, the same property that makes lip-sync structural here rather
    than approximate.

    The clock is the important part. `pacat`'s stdin has NO backpressure - it
    accepted 3.0 s of audio in 0.01 s when that was measured - so the time
    `write()` is called says nothing about when the sound emerges. Sample k is
    audible at `t0 + latency + k / RATE`, which is exactly the schedule the
    mouth loop runs on, and that loop is verified against /dev/fb0 at 5.92 s of
    mouth for 5.92 s of audio.
    """

    def __init__(self, rate, latency, seconds=8.0):
        self.src_rate = int(rate)
        self.latency = float(latency)
        self.t0 = None                       # set when the first frame is written
        self.buf = np.zeros(int(pcm.RATE * seconds), dtype=np.float32)
        self.n = 0                           # total samples ever written
        self.cap = len(self.buf)

    def write(self, frame):
        """Append one outgoing frame. Called from the writer thread."""
        if self.t0 is None:
            self.t0 = time.monotonic()
        x = pcm.to_float(np.asarray(frame))
        if self.src_rate != pcm.RATE:
            x = pcm.resample(x, self.src_rate, pcm.RATE)
        k = len(x)
        if k >= self.cap:                    # pathological; keep the tail
            x = x[-self.cap:]
            k = len(x)
        i = self.n % self.cap
        end = i + k
        if end <= self.cap:
            self.buf[i:end] = x
        else:
            split = self.cap - i
            self.buf[i:] = x[:split]
            self.buf[:end - self.cap] = x[split:]
        self.n += k

    def at(self, index, count):
        """`count` samples starting at absolute sample `index`, or None.

        None when the request falls outside what the ring still holds, which is
        the honest answer - a caller must not be handed zeros and told they are
        reference audio.
        """
        if index < 0 or count <= 0:
            return None
        if index + count > self.n or index < self.n - self.cap:
            return None
        i = index % self.cap
        end = i + count
        if end <= self.cap:
            return self.buf[i:end]
        return np.concatenate((self.buf[i:], self.buf[:end - self.cap]))

    def close(self):
        """A no-op, so this can sit in a TeeSink beside the real speaker.

        Being fed through the Tee rather than by a second `ref.write()` call
        next to `sink.write()` is deliberate: it keeps "the reference is the
        signal going to the speaker" true by construction instead of true by
        two lines staying in step.
        """
        pass

    def index_for(self, when, extra_lag=0.0):
        """Which sample should be emerging from the speaker at wall time `when`."""
        if self.t0 is None:
            return None
        off = (when - self.t0) - self.latency - extra_lag
        return int(off * pcm.RATE)


class Detector(object):
    """Is somebody talking over the reply? One bit, held for HOLD_S."""

    def __init__(self, reference, rise=RESIDUAL_RISE, hold_s=HOLD_S,
                 ambient=None):
        self.ref = reference
        self.rise = rise
        self.hold_s = hold_s
        # What counts as "not silence" in THIS room. See AMBIENT_MULT.
        self.min_level = (max(MIN_LEVEL_ABS, ambient * AMBIENT_MULT)
                          if ambient else MIN_LEVEL)
        self.floor = None             # residual rms with nobody talking
        self._floor_acc = []
        self.lag = 0.0                # refinement on top of ref.latency
        self.corr = 0.0               # how well the lag fits, 0..1
        self.cancels = 0.0            # fraction of energy the lock removes
        self.locked = False
        self.locked_at = None
        self.started = time.monotonic()
        self.over = 0.0               # seconds of sustained residual so far
        self.fired = False
        # Kept for the bench and for `tek ears`, so a false stop can be
        # diagnosed from a number rather than from a description of a noise.
        self.peak_residual = 0.0
        self._mic = []                # correlation window, mic side
        self._at = []                 # capture times for those frames

    # -- lag ---------------------------------------------------------------
    def _estimate_lag(self, when):
        """Cross-correlate mic against reference, once, near the known latency.

        FFT-based rather than np.correlate: at 16 kHz a 0.5 s window against a
        +/-0.35 s search is ~11k x 8k multiply-adds the direct way, which is
        not something to run on a board that is also holding 29 fps.
        """
        if not self._mic:
            return False
        mic = np.concatenate(self._mic)
        if len(mic) < int(CORR_S * pcm.RATE):
            return False
        span = int(SEARCH_S * pcm.RATE)
        start = self.ref.index_for(self._at[0])
        if start is None:
            return False
        ref = self.ref.at(start - span, len(mic) + 2 * span)
        if ref is None:
            return False

        # Normalised cross-correlation via FFT. Both sides are mean-removed so
        # a DC offset cannot manufacture a peak.
        m = mic - mic.mean()
        r = ref - ref.mean()
        if not m.any() or not r.any():
            return False
        n = 1
        while n < len(r) + len(m):
            n <<= 1
        cc = np.fft.irfft(np.fft.rfft(r, n) * np.conj(np.fft.rfft(m, n)), n)
        cc = cc[:2 * span + 1]
        if not len(cc):
            return False
        k = int(np.argmax(cc))
        denom = np.sqrt(float(np.dot(m, m)) * float(np.dot(r, r)))
        self.corr = float(cc[k] / denom) if denom > 0 else 0.0
        if self.corr < MIN_CORR:
            return False

        # Sign convention, worked through once so nobody has to do it again.
        #
        #   ref[i]  = ref_abs[start - span + i]        (the window we pulled)
        #   mic[i] ~= ref_abs[start + i - L]           (sound emitted L ago is
        #                                               what is arriving now)
        #   irfft(rfft(r) * conj(rfft(m)))[k] = sum_i ref[i+k] * mic[i]
        #
        # which peaks where i + k - span == i - L, i.e. at k = span - L. So the
        # lag is span - k, NOT k - span. Getting this backwards is not a subtle
        # bug: the magnitude comes out right, so the estimate looks correct,
        # but the subtraction is then misaligned by twice the true lag and the
        # entire echo lands in the residual - which is indistinguishable from
        # somebody talking, so every reply stops itself. Caught by
        # tools/bargein_bench.py, where echo alone fired at 1.26s.
        lag = (span - k) / float(pcm.RATE)

        # Does this alignment actually cancel? A correlation peak can be
        # confident and still be in the wrong place - a periodic signal has
        # many peaks, and a window that begins before the sound has arrived
        # correlates mostly silence against mostly silence. So the lock is
        # accepted only if subtracting the aligned reference removes a real
        # share of the energy, which is a direct test of the property the rest
        # of the detector depends on.
        aligned = ref[span - int(round(lag * pcm.RATE)):][:len(mic)]
        if len(aligned) < len(mic):
            return False
        rr = float(np.dot(aligned, aligned))
        if rr <= 0:
            return False
        alpha = float(np.dot(mic, aligned)) / rr
        resid = mic - alpha * aligned
        before = float(np.dot(mic, mic))
        if before <= 0:
            return False
        removed = 1.0 - float(np.dot(resid, resid)) / before
        if removed < LOCK_CANCELS:
            self.corr = 0.0
            return False

        self.lag = lag
        self.cancels = removed
        self.locked = True
        # `when`, not time.monotonic(). Every clock in this class has to be the
        # capture time of the frame being fed, or the offline bench - which
        # pushes an hour of audio through in a second of wall time - has a
        # floor timer that never elapses and a detector that never fires.
        self.locked_at = when
        return True

    # -- the per-frame decision --------------------------------------------
    def feed(self, frame, when=None):
        """One 20 ms mic frame in. True the moment barge-in is confirmed."""
        if self.fired:
            return True
        when = when or time.monotonic()
        x = pcm.to_float(np.asarray(frame))
        age = when - self.started

        if not self.locked:
            # Gather a correlation window, then lock. Detection does not run
            # until it does: an unaligned subtraction leaves the whole echo in
            # the residual, which looks exactly like a second voice.
            #
            # Do not START on silence. The reply has not reached the microphone
            # for the whole of the acoustic lag, so a window opened at the
            # first frame is mostly nothing: at a 320 ms lag, 64% of a 500 ms
            # window is pre-arrival silence. That still locks - correlating
            # silence against silence is easy - but on so little real signal
            # that the alignment is loose and the residual floor lands high,
            # which then desensitises the rise test for the rest of the
            # utterance. Every remaining miss in the sweep was this, and only
            # this. Frames already gathered are kept, since by then the sound
            # has demonstrably arrived.
            if not self._mic and float(np.sqrt(np.mean(x * x))) < self.min_level:
                return False
            self._mic.append(x)
            self._at.append(when)
            if sum(len(f) for f in self._mic) >= int(CORR_S * pcm.RATE):
                if not self._estimate_lag(when):
                    # Drop the oldest half and try again rather than giving up
                    # for the rest of the utterance - the speaker may simply
                    # not have started yet.
                    half = len(self._mic) // 2
                    self._mic = self._mic[half:]
                    self._at = self._at[half:]
            return False

        if age < WARMUP_S:
            return False

        idx = self.ref.index_for(when, self.lag)
        ref = self.ref.at(idx, len(x)) if idx is not None else None
        if ref is None or not ref.any():
            return False

        # Least-squares gain: the one scale factor that removes as much of the
        # reference from the mic as a single number can. Solving for it per
        # frame rather than fixing it tracks the speaker's volume and the
        # room's coupling for free.
        rr = float(np.dot(ref, ref))
        if rr <= 0:
            return False
        alpha = float(np.dot(x, ref)) / rr
        residual = x - alpha * ref

        r_rms = float(np.sqrt(np.mean(residual * residual)))
        self.peak_residual = max(self.peak_residual, r_rms)
        step = pcm.FRAME_MS / 1000.0

        # Learn the residual floor before judging anything against it. The
        # median, not the mean: a cough or a door during these few hundred
        # milliseconds would drag a mean up and desensitise the detector for
        # the rest of the utterance.
        if self.floor is None:
            self._floor_acc.append(r_rms)
            if (when - (self.locked_at or self.started)) < FLOOR_S:
                return False
            self.floor = float(np.median(self._floor_acc))
            self._floor_acc = []
            return False

        loud = r_rms > self.min_level and r_rms > self.rise * max(
            self.floor, self.min_level / 2)
        if loud:
            self.over += step
            if self.over >= self.hold_s:
                self.fired = True
                return True
        else:
            # Decay rather than reset. A person's speech has gaps in it at
            # 20 ms resolution - every stop consonant is one - and resetting
            # on the first quiet frame means a slow talker never trips it.
            self.over = max(0.0, self.over - step * 0.5)
        return False

    def state(self):
        return {"locked": self.locked, "corr": round(self.corr, 3),
                "min_level": round(self.min_level, 5),
                "cancels": round(self.cancels, 3),
                "lag_ms": round(self.lag * 1000, 1),
                "floor": None if self.floor is None else round(self.floor, 4),
                "held_ms": round(self.over * 1000, 1),
                "peak_residual": round(self.peak_residual, 4),
                "fired": self.fired}
