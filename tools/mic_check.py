#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the microphone actually producing sound, or just producing samples?

The last webcam's mic answered every "is it there" question correctly - device
present, source listed, samples flowing - while being dead hardware pinned at
-32758. So this asks the only question that matters: does the signal MOVE?

Two traps, both of which produced a confident wrong answer once:

  * `abs()` on int16 overflows at -32768, so a railed signal reports a
    perfectly healthy amplitude. Everything here converts to int32 first.
  * `proc.stdout.read(n)` blocks forever if the source never delivers n bytes,
    which looks identical to a hung script. Reads are bounded and the recording
    length is set by parec, not by how much we ask for.

    python3 tools/mic_check.py [--source NAME] [--seconds 3]
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np

RATE = 16000


def sources():
    out = subprocess.run(["pactl", "list", "sources", "short"],
                         stdout=subprocess.PIPE).stdout.decode()
    return [l.split("\t")[1] for l in out.splitlines() if "\t" in l]


def pick():
    """Whatever PulseAudio calls the default source.

    NOT "the first non-monitor in the list": that picks the Tegra onboard
    input, which has nothing plugged into it and returns a flat line, and then
    reports a perfectly good USB mic as dead. It also has to agree with
    MicSource, which passes no device and therefore gets the default - testing
    a different source than the one the voice service uses would make this tool
    worse than useless.
    """
    out = subprocess.run(["pactl", "info"], stdout=subprocess.PIPE).stdout.decode()
    for line in out.splitlines():
        if line.startswith("Default Source:"):
            name = line.split(":", 1)[1].strip()
            if name and not name.endswith(".monitor"):
                return name
    for name in sources():                  # fall back, least-bad order
        if not name.endswith(".monitor") and "platform-sound" not in name:
            return name
    return None


def record(source, seconds):
    """int16 mono at RATE. parec decides the length, so this cannot hang."""
    cmd = ["parec", "-d", source, "--format=s16le", "--rate=%d" % RATE,
           "--channels=1", "--latency-msec=100"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    want = RATE * 2 * seconds
    buf = b""
    try:
        while len(buf) < want:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
    finally:
        p.kill()
        p.wait()
    return np.frombuffer(buf[:want], dtype=np.int16)


def report(x, label):
    ok = True
    if len(x) < RATE // 2:
        print("  %s: only %d samples - the source gave us nothing" % (label, len(x)))
        return False
    # int32 FIRST: abs(-32768) is -32768 in int16 and a railed mic then looks
    # like a loud one.
    v = x.astype(np.int32)
    dc = float(v.mean())
    ac = v - dc
    rms = float(np.sqrt((ac.astype(np.float64) ** 2).mean()))
    peak = int(np.abs(v).max())
    uniq = int(len(np.unique(v)))
    railed = int((np.abs(v) >= 32700).sum())

    print("  %s" % label)
    print("    samples %d   DC offset %+.1f   peak %d" % (len(v), dc, peak))
    print("    RMS about the mean %.1f   distinct values %d   railed %d"
          % (rms, uniq, railed))

    # A dead mic is not silent - it is CONSTANT. That is the distinction the
    # last one defeated.
    if uniq <= 4:
        print("    -> DEAD: the signal never changes (%d distinct values)" % uniq)
        ok = False
    elif rms < 1.0:
        print("    -> digital silence: samples flow but carry no signal")
        ok = False
    elif railed > len(v) // 10:
        print("    -> railed: %d%% of samples are at the limit"
              % (100 * railed // len(v)))
        ok = False
    else:
        print("    -> alive")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--seconds", type=int, default=3)
    a = ap.parse_args()

    print("input sources:")
    for s in sources():
        print("   %s%s" % (s, "   (monitor)" if s.endswith(".monitor") else ""))
    src = a.source or pick()
    if src is None:
        print("no capture source at all")
        return 2
    print("\nrecording %ds from %s" % (a.seconds, src))
    quiet = record(src, a.seconds)
    ok = report(quiet, "ambient")

    print("\nNow make some noise - talk, clap - for %d seconds." % a.seconds)
    sys.stdout.flush()
    loud = record(src, a.seconds)
    ok2 = report(loud, "while you were making noise")

    if ok and ok2:
        q = np.sqrt((quiet.astype(np.float64) ** 2).mean())
        l = np.sqrt((loud.astype(np.float64) ** 2).mean())
        print("\n  loud/quiet RMS ratio: %.2fx" % (l / max(q, 1e-9)))
        if l > q * 1.5:
            print("  -> the mic RESPONDS to sound. It is genuinely working.")
        else:
            print("  -> level barely changed; either it was quiet both times "
                  "or the mic is not picking up the room.")
    print("\nMIC " + ("OK" if ok and ok2 else "PROBLEM"))
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
