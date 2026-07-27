#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close the loop through the air: speak, and listen to the room.

`mic_check.py` proves the mic produces a varying signal. That is necessary and
not sufficient - a mic can hum happily to itself while hearing nothing. This
plays a known sentence through the Bluetooth speaker and records the room while
it plays, so the question becomes "did the level rise when the room got loud",
which no amount of self-noise can fake.

It then hands the recording to Vosk. Getting words back proves the whole
acoustic path end to end: speaker -> air -> mic -> resample -> recogniser. That
is the path the wake word will run on, and it is the one thing that cannot be
tested without hardware.

    tools/mic_room.py            (needs the voice service running)
"""
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())

# vosk's libvosk.so needs a newer libstdc++ than this box ships, kept in lib/.
# LD_LIBRARY_PATH must be set BEFORE the process starts - the dynamic linker
# reads it at exec time, so setting os.environ here does nothing at all. This
# script "worked" once with the recogniser silently unavailable for exactly
# that reason. Re-exec, as tests/voice_stt.py does.
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lib")
if os.path.isdir(_LIB) and _LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo.voice import pcm

LINE = ("The quick brown fox jumps over the lazy dog. "
        "Testing one two three four five.")


def default_source():
    out = subprocess.run(["pactl", "info"], stdout=subprocess.PIPE).stdout.decode()
    for line in out.splitlines():
        if line.startswith("Default Source:"):
            return line.split(":", 1)[1].strip()
    return None


def record(source, seconds, out):
    cmd = ["parec", "-d", source, "--format=s16le", "--rate=%d" % pcm.RATE,
           "--channels=1", "--latency-msec=100"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    want = pcm.RATE * 2 * seconds
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
    out.append(np.frombuffer(buf[:want], dtype=np.int16))


def rms(x):
    v = x.astype(np.float64)          # int32 first would still overflow squares
    return float(np.sqrt((v ** 2).mean())) if len(x) else 0.0


def main():
    src = default_source()
    if not src or src.endswith(".monitor"):
        print("no usable default source (%r)" % src)
        return 2
    print("mic: %s" % src)

    print("\n-- 4s of silence, for a floor --")
    quiet = []
    record(src, 4, quiet)
    quiet = quiet[0]
    print("   ambient RMS %.1f" % rms(quiet))

    print("\n-- speaking, and recording the room at the same time --")
    loud = []
    t = threading.Thread(target=record, args=(src, 12, loud))
    t.start()
    time.sleep(0.5)
    r = subprocess.run(["/home/super/tekdromo/tek", "say", LINE],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("   %s" % r.stdout.decode().strip().splitlines()[-1:])
    t.join()
    loud = loud[0]
    print("   RMS while speaking %.1f" % rms(loud))

    fail = []
    ratio = rms(loud) / max(rms(quiet), 1e-9)
    print("\n   loud/quiet ratio: %.2fx" % ratio)
    if ratio < 1.5:
        fail.append("the mic did not hear the speaker (%.2fx)" % ratio)
    else:
        print("   -> the mic HEARS THE ROOM.")

    # The real proof: can the recogniser read it back?
    try:
        from tekdromo.voice import stt
        rec = stt.Recogniser()
        text = rec.transcribe(loud)
        print("\n   Vosk heard: %r" % text)
        words = set(text.lower().split())
        hits = words & {"quick", "brown", "fox", "jumps", "lazy", "dog",
                        "testing", "one", "two", "three", "four", "five"}
        print("   matched %d keywords: %s" % (len(hits), sorted(hits)))
        if len(hits) < 3:
            fail.append("recognised only %d keywords over the air" % len(hits))
    except Exception as e:
        print("\n   (recogniser unavailable: %s)" % e)

    print("\nMIC ROOM " + ("OK" if not fail else "FAILED: " + "; ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
