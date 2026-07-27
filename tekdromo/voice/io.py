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
import os
import subprocess
import threading
import time
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

    def __init__(self, freq=440.0, seconds=1.0, amp=0.3, rate=pcm.RATE,
                 fade=0.0):
        ArraySource.__init__(self, pcm.tone(freq, seconds, amp, rate, fade), rate)


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


def _pactl_info():
    try:
        return subprocess.check_output(["pactl", "info"],
                                       stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return ""


def default_sink():
    """Name of the sink audio is currently going to, or None."""
    for line in _pactl_info().splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip() or None
    return None


def default_monitor():
    """CONCRETE name of the current default sink's monitor, or None.

    Deliberately not "@DEFAULT_MONITOR@". PulseAudio resolves that magic name
    ONCE, when the stream is created, and never moves the stream afterwards -
    so a recorder started before the Bluetooth speaker connected stays bound to
    the analog monitor for as long as it lives, watching a device nothing plays
    through. parec does not exit in that state either, so a reconnect-on-death
    loop never notices. That is exactly how the waveform panel came to show a
    flat line while music was playing.

    Resolving to a real name means the caller can COMPARE it with what it is
    connected to, and reconnect when it changes.
    """
    sink = default_sink()
    return sink + ".monitor" if sink else None


def default_source():
    """Name of the default capture source, or None if it is only a monitor.

    A monitor is not a microphone: taking one as an input is how "the mic is
    dead" gets diagnosed for a machine whose mic is fine.
    """
    for line in _pactl_info().splitlines():
        if line.startswith("Default Source:"):
            name = line.split(":", 1)[1].strip()
            return name if name and not name.endswith(".monitor") else None
    return None


def input_sources():
    """Every capture source that is not a monitor, the default one first."""
    try:
        out = subprocess.check_output(["pactl", "list", "sources", "short"],
                                      stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return []
    names = [l.split("\t")[1] for l in out.splitlines() if "\t" in l]
    names = [n for n in names if not n.endswith(".monitor")]
    first = default_source()
    if first in names:
        names = [first] + [n for n in names if n != first]
    return names


def source_alive(name, seconds=0.6, distinct=8):
    """Does this capture source deliver a VARYING signal?

    Distinct values, not amplitude. A dead input is not silent, it is
    CONSTANT - the webcam that used to be on this box sat at -32758 forever
    and passed every "is there a signal" test that looked at level. The Tegra
    onboard input, which has nothing plugged into it, returns a flat line.

    Bounded by a deadline on a non-blocking pipe. A plain read() on a source
    that never delivers hangs forever, which in the ear would look exactly
    like a quiet room.
    """
    import fcntl
    import select
    p = None
    try:
        p = subprocess.Popen(
            ["parec", "-d", name, "--format=s16le", "--rate=%d" % pcm.RATE,
             "--channels=1", "--latency-msec=50"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fd = p.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL,
                    fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        end = time.time() + seconds
        buf = b""
        while time.time() < end:
            if not select.select([fd], [], [], max(0.0, end - time.time()))[0]:
                continue
            try:
                chunk = os.read(fd, 8192)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
        if len(buf) < 2000:
            return False
        x = np.frombuffer(buf[:len(buf) // 2 * 2], dtype=np.int16)
        return int(len(np.unique(x))) >= distinct
    except Exception:
        return False
    finally:
        if p is not None:
            try:
                p.kill()
                p.wait()
            except Exception:
                pass


_SRC_CACHE = {"t": 0.0, "name": None, "cands": None}


def working_source(probe=0.6, ttl=0.0):
    """A capture source that actually works, or None.

    `ttl` reuses the last answer for that many seconds - but the cache is ALSO
    invalidated whenever the set of capture sources changes, which is the only
    event that can make the answer different. So callers can poll this as often
    as they like and it costs one `pactl list sources short` until a device is
    plugged or unplugged.

    That matters more than it looks. Probing opens and kills a recorder on
    every candidate, and doing that on a timer churns PulseAudio while A2DP is
    streaming. PulseAudio's default exit-idle-time is 20 s, so any moment with
    no client at all takes the Bluetooth speaker down with it - 96 daemon
    restarts were logged in one day. exit-idle-time is now -1, but there is
    still no reason to open recorders nobody asked for.

    The PulseAudio default is tried first but is NOT trusted. On this box the
    microphone is built into the webcam, so every camera replug - now a routine
    event rather than a fault - tears the source down, and PulseAudio moves the
    default to the Tegra onboard input, which has no microphone attached. It
    does not move back when the camera returns. The observed result was two
    parec streams sitting on a dead input while the real microphone was idle,
    with the ear reporting itself healthy and hearing nothing at all.

    So each candidate is probed until one delivers a varying signal.
    """
    now = time.time()
    cands = input_sources()
    if (ttl and _SRC_CACHE["name"] and _SRC_CACHE["cands"] == cands
            and now - _SRC_CACHE["t"] < ttl):
        return _SRC_CACHE["name"]
    found = None
    for name in cands:
        if source_alive(name, probe):
            found = name
            break
    if found is None:
        found = cands[0] if cands else None
    _SRC_CACHE["t"] = now
    _SRC_CACHE["name"] = found
    _SRC_CACHE["cands"] = cands
    return found


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
