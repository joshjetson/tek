"""
Voice activity detection - the thing that makes always-on listening affordable.

Recognition is expensive; silence is most of the day. Gating on speech means
the recogniser is idle ~95% of the time, which is the single largest efficiency
decision in the whole pipeline - far larger than the choice of model.

WebRTC's VAD is used rather than a neural one (Silero needs torch, which will
not fit here) or a plain energy threshold. It costs essentially nothing, and
unlike an energy gate it does not trip on a fridge compressor or a fan.

Two details that decide whether this feels good or awful:

  * PRE-ROLL. Speech is detected a beat after it starts, so without a rolling
    buffer of what came just before, the first phoneme is always clipped -
    "hey tek" arrives as "ey tek" and the wake word never matches. A ring
    buffer holds the last few hundred ms at all times.
  * HANG-OVER. People pause mid-sentence. Ending an utterance at the first
    quiet frame chops "what time... is it" into two fragments, so silence has
    to persist before the utterance is closed.
"""
import collections

import numpy as np

from . import pcm


class Segmenter(object):
    """Turns a continuous Source into discrete utterances.

        for utterance in Segmenter().segments(source):
            text = recogniser.transcribe(utterance)

    Yields int16 arrays at pcm.RATE.
    """

    def __init__(self, aggressiveness=2, pre_roll=0.30, hang=0.60,
                 min_speech=0.20, max_utterance=15.0):
        import webrtcvad
        # 0..3. 3 rejects the most non-speech but also drops quiet talkers;
        # 2 is the usual choice for a room mic at conversational distance.
        self.vad = webrtcvad.Vad(aggressiveness)
        self.pre_n = max(1, int(pre_roll * 1000 / pcm.FRAME_MS))
        self.hang_n = max(1, int(hang * 1000 / pcm.FRAME_MS))
        self.min_n = max(1, int(min_speech * 1000 / pcm.FRAME_MS))
        self.max_n = max(1, int(max_utterance * 1000 / pcm.FRAME_MS))

    def is_speech(self, frame):
        """WebRTC accepts only 10/20/30 ms frames at 8/16/32/48 kHz - which is
        exactly why pcm.FRAME_MS is 20 and pcm.RATE is 16000. Anything else
        would need a second framing just for this."""
        if len(frame) != pcm.FRAME:
            return False
        return self.vad.is_speech(frame.tobytes(), pcm.RATE)

    def segments(self, source):
        pre = collections.deque(maxlen=self.pre_n)
        voiced = []
        quiet = 0
        active = False

        for frame in source:
            speech = self.is_speech(frame)
            if not active:
                pre.append(frame)
                if speech:
                    # Start with everything already in the ring buffer, so the
                    # onset of the first word is included rather than clipped.
                    voiced = list(pre)
                    pre.clear()
                    active = True
                    quiet = 0
                continue

            voiced.append(frame)
            quiet = 0 if speech else quiet + 1

            if quiet >= self.hang_n or len(voiced) >= self.max_n:
                # Trim the trailing silence the hang-over let through, but keep
                # a little: recognisers do better with a bit of room tone after
                # the last word than with an abrupt cut.
                keep = len(voiced) - max(0, quiet - self.pre_n)
                out = voiced[:max(1, keep)]
                if len(out) >= self.min_n:
                    yield np.concatenate(out)
                voiced, active, quiet = [], False, 0

        if active and len(voiced) >= self.min_n:
            yield np.concatenate(voiced)          # flush at end of stream


def speech_ratio(source, aggressiveness=2):
    """Fraction of frames containing speech. Useful for sanity-checking a
    recording, and for reporting how much of the day was actually processed."""
    seg = Segmenter(aggressiveness)
    n = spoken = 0
    for f in source:
        n += 1
        if seg.is_speech(f):
            spoken += 1
    return (float(spoken) / n) if n else 0.0
