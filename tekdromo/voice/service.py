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

from . import bus, io as vio, pcm, tts

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


def save_choice(name):
    import json
    d = os.path.dirname(CONFIG)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(CONFIG, "w") as f:
        json.dump({"voice": name}, f)

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
CHUNK_RAMP = [90, 140, 200, 260, 320]
# Never start speaking on less than this much buffered audio. With the ramp
# above this is what actually guarantees the producer stays ahead for the rest
# of the reply, at the cost of ~1 s more before the first word.
MIN_LEAD_S = 5.0


def _chunks(text):
    sentences = [x.strip() for x in _SENT.split((text or "").strip()) if x.strip()]
    out, buf = [], ""
    for sent in sentences:
        limit = CHUNK_RAMP[min(len(out), len(CHUNK_RAMP) - 1)]
        if buf and len(buf) + len(sent) + 1 > limit:
            out.append(buf)
            buf = sent
        else:
            buf = (buf + " " + sent).strip()
    if buf:
        out.append(buf)
    return out


class VoiceService(object):

    def __init__(self, voice=None, device=None, path=bus.DEFAULT_PATH,
                 latency_trim=0.0):
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
            parts = _chunks(text)
            if not parts:
                return {"ok": False, "error": "nothing to say"}

            rate = v.rate
            n = int(rate * pcm.FRAME_MS / 1000)
            frames = []                 # grows as synthesis proceeds
            envs = []
            done = threading.Event()
            state = {"synth": 0.0, "audio": 0.0, "rounding": 0.0, "error": None}

            def produce():
                try:
                    for part in parts:
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
            self.server.publish({"speaking": True, "text": text})
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
            dur = state["audio"]
            print("said %.1fs in %.1fs (%.2fx, %d chunks) [%s] %s"
                  % (dur, state["synth"], state["synth"] / max(dur, 1e-6),
                     len(parts), v.name, text[:44].replace("\n", " ")),
                  flush=True)
            return {"ok": True, "duration": round(dur, 2),
                    "synth": round(state["synth"], 2), "chunks": len(parts),
                    "voice": v.name}

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
