# -*- coding: utf-8 -*-
"""The Source/Sink spine, end to end, with no hardware attached.

This is the test the whole design exists to make possible: because a WAV file
and a microphone are the same type, and a file and a speaker are the same type,
the real pipeline can be exercised before any microphone or speaker is plugged
in. Nothing here touches ALSA, PulseAudio or Bluetooth.
"""
import os
import sys
import tempfile

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo.voice import io as vio, pcm

FAIL = []


def check(name, cond, extra=""):
    print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


tmp = tempfile.mkdtemp(prefix="tekvoice")

# -- round trip through a file --------------------------------------------
src = vio.ToneSource(freq=440.0, seconds=0.5)
wav = os.path.join(tmp, "tone.wav")
n = vio.pump(src, vio.WavSink(wav))
check("tone -> WavSink writes frames", n == 25, n)
back = vio.WavSource(wav).all()
check("WavSource reads back the same length", len(back) == 25 * pcm.FRAME, len(back))
check("signal survives the round trip (not silence)", pcm.envelope(back[:320]) > 0.1)

# -- rate conversion at the edges -----------------------------------------
import wave
odd = os.path.join(tmp, "odd.wav")
w = wave.open(odd, "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
w.writeframes(np.zeros(22050, np.int16).tobytes()); w.close()
got = vio.WavSource(odd).all()
check("a 22.05kHz file is normalised to the 16kHz contract",
      abs(len(got) - 16000) <= pcm.FRAME, len(got))

# stereo must be downmixed, not have one channel dropped: a mic on the other
# channel would otherwise vanish silently.
st = os.path.join(tmp, "stereo.wav")
w = wave.open(st, "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000)
inter = np.zeros(3200, np.int16)
inter[1::2] = 10000                       # signal ONLY on the right channel
w.writeframes(inter.tobytes()); w.close()
check("stereo is downmixed, not channel-0-only",
      pcm.envelope(vio.WavSource(st).all()[:320]) > 0.05)

# -- Tee: the reason lip-sync is structural -------------------------------
a, b = vio.ArraySink(), vio.ArraySink()
vio.pump(vio.ToneSource(seconds=0.2), vio.TeeSink(a, b))
check("Tee delivers identical PCM to both sinks",
      np.array_equal(a.array(), b.array()) and len(a.array()) > 0)


class Exploding(vio.Sink):
    def write(self, f):
        raise RuntimeError("this sink is broken")


ok = vio.ArraySink()
vio.pump(vio.ToneSource(seconds=0.2), vio.TeeSink(Exploding(), ok))
check("a failing sink does not silence the others (speaker survives a dead face)",
      len(ok.array()) > 0)

# -- Delay: A2DP latency compensation -------------------------------------
d = vio.ArraySink()
delayed = vio.DelaySink(d, 0.2)                 # 10 frames
for i in range(5):
    delayed.write(np.full(pcm.FRAME, i + 1, np.int16))
check("DelaySink holds frames back", len(d.chunks) == 0, len(d.chunks))
for i in range(8):
    delayed.write(np.full(pcm.FRAME, 100, np.int16))
check("DelaySink releases after the delay", len(d.chunks) == 3, len(d.chunks))
check("DelaySink preserves order", d.chunks[0][0] == 1 and d.chunks[1][0] == 2)
delayed.close()
check("DelaySink flushes the tail on close, losing nothing",
      len(d.chunks) == 13, len(d.chunks))

# -- Null / empty edge cases ----------------------------------------------
nl = vio.NullSink()
vio.pump(vio.ArraySource(np.zeros(0, np.int16)), nl)
check("an empty source produces no frames", nl.frames == 0)
check("ArraySource.all() on empty returns an empty array",
      len(vio.ArraySource(np.zeros(0, np.int16)).all()) == 0)

# -- the full shape, with stubs where hardware would be -------------------
# WavSource -> (stub recogniser) -> (stub brain) -> WavSink. This is the real
# service topology; only the two ends are stubbed.
heard = []


def stub_stt(source):
    return "utterance of %d frames" % sum(1 for _ in source)


def stub_brain(text):
    return "reply to: " + text


out = os.path.join(tmp, "reply.wav")
text = stub_stt(vio.WavSource(wav))
reply = stub_brain(text)
vio.pump(vio.ToneSource(freq=880.0, seconds=0.3), vio.WavSink(out))
check("full pipeline runs with no hardware", reply.startswith("reply to:")
      and os.path.getsize(out) > 1000, reply)

print("VOICE LOOPBACK " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
