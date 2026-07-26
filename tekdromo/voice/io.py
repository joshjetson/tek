"""
Sources and Sinks - the only two I/O concepts in the voice pipeline.

A Source yields frames. A Sink accepts them. Everything else is expressed in
terms of these, which buys three things:

  * A microphone and a WAV file are the same type, so the entire pipeline is
    testable with no hardware. That is not a testing nicety here - it is how
    this was built before any microphone existed.
  * The speaker and the FACE are both Sinks, so the mouth is driven by the very
    same PCM that is played. Lip-sync is correct by construction rather than by
    keeping a second animation path in step with the audio.
  * Adding a stage never requires touching another one.

Audio hardware is reached through PulseAudio's `parec`/`pacat` rather than a
Python binding. That is a deliberate choice: it needs no compiled dependency on
a Python 3.6 / glibc 2.27 box where most audio wheels do not build, and it
routes through PulseAudio, so Bluetooth "just works" and follows the default
sink when the speaker connects or drops.
"""
import subprocess
import threading
import wave

import numpy as np

from . import pcm


# --------------------------------------------------------------------------
class Source(object):
    """Yields fixed-size int16 frames at pcm.RATE. read() returns None at EOF."""

    rate = pcm.RATE

    def read(self):
        raise NotImplementedError

    def __iter__(self):
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def all(self):
        """Drain to a single array. Only for finite sources."""
        got = list(self)
        return np.concatenate(got) if got else pcm.silence(0)

    def close(self):
        pass


class Sink(object):
    """Accepts int16 frames."""

    def write(self, frame):
        raise NotImplementedError

    def close(self):
        pass


# -- sources ---------------------------------------------------------------
class ArraySource(Source):
    """An in-memory array. The backbone of every hardware-free test."""

    def __init__(self, samples, rate=pcm.RATE):
        if rate != pcm.RATE:
            samples = pcm.resample(samples, rate, pcm.RATE)
        self._it = pcm.frames(np.asarray(samples, dtype=pcm.DTYPE))

    def read(self):
        for f in self._it:
            return f
        return None


class WavSource(ArraySource):
    """A WAV file, converted to the pipeline's contract."""

    def __init__(self, path):
        w = wave.open(path, "rb")
        try:
            n, ch, sw, sr = (w.getnframes(), w.getnchannels(),
                             w.getsampwidth(), w.getframerate())
            if sw != 2:
                raise ValueError("%s: need 16-bit PCM, got %d-byte samples"
                                 % (path, sw))
            data = np.frombuffer(w.readframes(n), dtype=np.int16)
        finally:
            w.close()
        if ch > 1:                       # downmix, don't just take channel 0:
            data = data.reshape(-1, ch)  # a mic on one channel would be lost
            data = data.mean(axis=1).astype(np.int16)
        ArraySource.__init__(self, data, sr)


class ToneSource(ArraySource):
    """A sine burst. Useful for proving an output path end to end."""

    def __init__(self, freq=440.0, seconds=1.0, amp=0.3, rate=pcm.RATE):
        t = np.arange(int(rate * seconds), dtype=np.float32) / rate
        ArraySource.__init__(self, pcm.from_float(amp * np.sin(2 * np.pi * freq * t)),
                             rate)


class MicSource(Source):
    """Live capture via PulseAudio's parec.

    Reads exact frame-sized blocks; a short read at the end of a stream means
    the recorder died, which is reported rather than silently treated as
    silence - a mic that has stopped working must not look like a quiet room.
    """

    def __init__(self, device=None, rate=pcm.RATE):
        cmd = ["parec", "--format=s16le", "--rate=%d" % rate, "--channels=1",
               "--latency-msec=50"]
        if device:
            cmd.append("--device=%s" % device)
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self._n = pcm.FRAME * 2

    def read(self):
        buf = self.p.stdout.read(self._n)
        if not buf or len(buf) < self._n:
            return None
        return np.frombuffer(buf, dtype=np.int16)

    def close(self):
        try:
            self.p.terminate()
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


# -- sinks -----------------------------------------------------------------
class NullSink(Sink):
    """Counts frames and nothing else. Lets a pipeline run headless."""

    def __init__(self):
        self.frames = 0

    def write(self, frame):
        self.frames += 1


class ArraySink(Sink):
    def __init__(self):
        self.chunks = []

    def write(self, frame):
        self.chunks.append(np.asarray(frame, dtype=pcm.DTYPE))

    def array(self):
        return (np.concatenate(self.chunks) if self.chunks
                else pcm.silence(0))


class WavSink(Sink):
    def __init__(self, path, rate=pcm.RATE):
        self.w = wave.open(path, "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(rate)

    def write(self, frame):
        self.w.writeframes(np.asarray(frame, dtype=pcm.DTYPE).tobytes())

    def close(self):
        try:
            self.w.close()
        except Exception:
            pass


class SpeakerSink(Sink):
    """Playback via PulseAudio's pacat.

    Rate is a parameter because this is a hardware edge: Piper produces
    22.05 kHz and there is no reason to downsample it to 16 kHz only for
    PulseAudio to convert it back up again. Passing device=None follows the
    default sink, so audio moves to the Bluetooth speaker when it connects
    without anything here knowing that Bluetooth exists.
    """

    def __init__(self, device=None, rate=pcm.RATE, latency_ms=200):
        # Bound the buffer. Without --latency-msec, PulseAudio picks a large
        # target and pacat's stdin swallows an entire utterance in ~10 ms, so
        # nothing downstream can use write progress as a clock. Measured: 3.0 s
        # of audio written in 0.01 s. Anything that needs to know where the
        # audio actually IS must use the wall clock - see service._say.
        cmd = ["pacat", "--format=s16le", "--rate=%d" % rate, "--channels=1",
               "--latency-msec=%d" % latency_ms]
        if device:
            cmd.append("--device=%s" % device)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)

    def write(self, frame):
        try:
            self.p.stdin.write(np.asarray(frame, dtype=pcm.DTYPE).tobytes())
        except (IOError, ValueError):
            pass                    # speaker vanished mid-utterance; keep going

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)   # let the buffer drain, don't clip the end
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


class TeeSink(Sink):
    """Write one stream to several sinks.

    This is the piece that makes lip-sync structural: the audio going to the
    speaker and the audio driving the mouth are not two signals that have to be
    kept in agreement, they are one signal with two consumers.

    A failing sink is dropped rather than propagated - the speaker must not go
    silent because the face is not listening.
    """

    def __init__(self, *sinks):
        self.sinks = [s for s in sinks if s is not None]

    def write(self, frame):
        for s in list(self.sinks):
            try:
                s.write(frame)
            except Exception:
                self.sinks.remove(s)

    def close(self):
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass


class DelaySink(Sink):
    """Hold frames back by a fixed time before passing them on.

    For A2DP: Bluetooth adds 150-250 ms of latency, so a mouth driven from the
    same PCM would move ahead of the sound. Delaying the FACE by the sink's
    reported latency puts them back together. Delaying the audio instead would
    just add that latency twice.
    """

    def __init__(self, sink, seconds):
        self.sink = sink
        self.n = max(0, int(round(seconds * pcm.RATE / pcm.FRAME)))
        self.buf = []

    def write(self, frame):
        self.buf.append(frame)
        if len(self.buf) > self.n:
            self.sink.write(self.buf.pop(0))

    def close(self):
        for f in self.buf:
            self.sink.write(f)
        self.buf = []
        self.sink.close()


def sink_latency(device=None, default=0.20):
    """Seconds between writing a sample and hearing it, as PulseAudio sees it.

    Used to delay the MOUTH so it lines up with the sound. A2DP adds 150-250 ms
    that the face would otherwise run ahead of.

    This is only what PulseAudio can see - a Bluetooth speaker's own internal
    buffer is invisible from here - so it is a floor, not the whole truth, and
    the service allows a manual trim on top.
    """
    try:
        out = subprocess.check_output(["pactl", "list", "sinks"],
                                      stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return default
    cur, best = None, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            cur = line.split(":", 1)[1].strip()
        elif line.startswith("Latency:") and ("usec" in line):
            if device and cur != device:
                continue
            try:
                usec = float(line.split("usec")[0].split(":")[1].strip())
            except (ValueError, IndexError):
                continue
            # Configured latency, not the momentary one, which reads 0 on a
            # suspended sink.
            if usec > 0:
                best = usec / 1e6 if best is None else max(best, usec / 1e6)
    return best if best else default


def pump(source, sink, close=True):
    """Move every frame from a Source to a Sink. The whole pipeline, one line."""
    n = 0
    for f in source:
        sink.write(f)
        n += 1
    if close:
        sink.close()
    return n
