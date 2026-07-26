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
import threading
import time
import traceback

import numpy as np

from . import bus, io as vio, pcm, tts


class VoiceService(object):

    def __init__(self, voice=None, device=None, path=bus.DEFAULT_PATH,
                 latency_trim=0.0):
        t0 = time.time()
        self.voice = tts.load(voice)
        self.device = device
        self.load_time = time.time() - t0
        self.spoken = 0
        self.speaking = False
        self._lock = threading.Lock()     # one utterance at a time
        # How far the mouth must lag the audio. Measured from PulseAudio, plus
        # a manual trim for the part it cannot see (a Bluetooth speaker's own
        # internal buffer is invisible from this side).
        self.latency = vio.sink_latency(device) + float(latency_trim)
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
                return self.say(text, wait=msg.get("wait", True))
            except Exception as e:
                traceback.print_exc()
                return {"ok": False, "error": str(e)}
        if cmd == "latency":
            # Lets the lag be trimmed by ear without a restart.
            if "seconds" in msg:
                self.latency = max(0.0, float(msg["seconds"]))
            return {"ok": True, "latency": round(self.latency, 3)}
        if cmd == "status":
            return {"ok": True, "voice": self.voice.name,
                    "latency": round(self.latency, 3),
                    "rate": self.voice.rate, "speaking": self.speaking,
                    "spoken": self.spoken, "load_time": round(self.load_time, 2)}
        if cmd == "ping":
            return {"ok": True}
        return {"ok": False, "error": "unknown command %r" % cmd}

    # -- speaking ----------------------------------------------------------
    def say(self, text, wait=True):
        if not wait:
            t = threading.Thread(target=self._say, args=(text,))
            t.daemon = True
            t.start()
            return {"ok": True, "queued": True}
        return self._say(text)

    def _say(self, text):
        with self._lock:
            t0 = time.time()
            samples, rate = self.voice.synth(text)
            synth_s = time.time() - t0
            dur = len(samples) / float(rate)
            if not len(samples):
                return {"ok": False, "error": "voice produced no audio"}

            # Rounding comes from the phonemes when the voice knows them.
            # from_envelope() cannot: an RMS level carries no vowel shape.
            rnd = float(getattr(self.voice, "last_rounding", 0.0))

            # Frame at the VOICE's rate so 20 ms of audio is 20 ms of mouth.
            n = int(rate * pcm.FRAME_MS / 1000)
            chunks = list(pcm.frames(np.asarray(samples), n))
            envs = [pcm.envelope(f) for f in chunks]

            self.speaking = True
            self.server.publish({"speaking": True, "text": text,
                                 "duration": round(dur, 2)})
            sink = vio.SpeakerSink(device=self.device, rate=rate)

            # The audio is written as fast as PulseAudio will take it, on its
            # own thread. It CANNOT be used as a clock: pacat's stdin is a pipe
            # and it accepted 3.0 s of audio in 0.01 s, so driving the mouth
            # from the write loop animated an entire sentence in ten
            # milliseconds and then left the face still for the rest of it.
            # That is exactly what it looked like - the face stopped long
            # before the voice did.
            def writer():
                try:
                    for f in chunks:
                        sink.write(f)
                finally:
                    sink.close()

            wt = threading.Thread(target=writer)
            wt.daemon = True
            wt.start()

            # The mouth runs on the wall clock instead, offset by however long
            # the sound takes to actually emerge. A2DP is 150-250 ms of that.
            try:
                start = time.time() + self.latency
                step = pcm.FRAME_MS / 1000.0
                for i, e in enumerate(envs):
                    slack = (start + i * step) - time.time()
                    if slack > 0:
                        time.sleep(slack)
                    self.server.publish({"mouth": [round(e, 4), round(rnd, 3)]})
            finally:
                wt.join(timeout=dur + 10)
                self.speaking = False
                self.server.publish({"speaking": False})
                self.server.publish({"mouth": [0.0, 0.0]})
            self.spoken += 1
            print("said %.2fs in %.2fs (%.2fx) : %s"
                  % (dur, synth_s, synth_s / max(dur, 1e-6), text[:60]),
                  flush=True)
            return {"ok": True, "duration": round(dur, 2),
                    "synth": round(synth_s, 2), "voice": self.voice.name}

    def run(self):
        self.server.start()
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
    ap.add_argument("--latency-trim", type=float, default=0.0,
                    help="extra seconds to delay the MOUTH behind the audio. "
                         "PulseAudio cannot see a Bluetooth speaker's own "
                         "buffer, so if the face still leads the sound, add it "
                         "here (e.g. 0.15).")
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
    VoiceService(a.voice, a.device, a.socket, a.latency_trim).run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
