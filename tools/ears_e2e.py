#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Say the wake word out loud, as a person would, and see if it answers.

The trick is playing the audio with `paplay` instead of through the voice
service. The service mutes its own ear while it is speaking - it has to, or it
answers its own replies forever - so anything spoken via `tek say` is
deliberately unhearable. Going straight to PulseAudio leaves `speaking` False,
so the microphone picks the words out of the room exactly as it would pick up a
person standing there.

This exercises the entire path with nothing stubbed: Piper -> speaker -> air ->
mic -> VAD -> wake grammar -> free decode -> brain -> speech. It costs one
model call and about half a minute.

    tools/ears_e2e.py ["something to ask"]
"""
import json
import os
import subprocess
import sys
import time
import wave

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lib")
if os.path.isdir(_LIB) and _LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tekdromo.voice import bus, tts

WAV = "/tmp/tek-wake-test.wav"


def ask(msg, timeout=30.0):
    c = bus.Client(bus.DEFAULT_PATH, timeout=timeout)
    try:
        return c.request(msg) or {}
    finally:
        try:
            c.close()
        except Exception:
            pass


def synth_to_wav(text, path):
    v = tts.load()
    samples, rate = v.synth(text)
    samples = np.asarray(samples, dtype=np.int16)
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(int(rate))
    w.writeframes(samples.tobytes())
    w.close()
    return len(samples) / float(rate)


def main():
    line = " ".join(sys.argv[1:]) or "Hey Tek, what day of the week is it?"

    try:
        before = ask({"cmd": "ears"})
    except Exception as e:
        print("cannot reach the voice service: %s" % e)
        return 2
    if not before.get("listening"):
        print("the service is not listening (tek ears on)")
        return 2
    print("before:  utterances=%s wakes=%s commands=%s"
          % (before.get("utterances"), before.get("wakes"),
             before.get("commands")))

    print("\nsynthesising %r" % line)
    secs = synth_to_wav(line, WAV)
    print("   %.1fs of audio" % secs)

    print("\nplaying it into the room with paplay (NOT through the service,\n"
          "so the ear is not muted and hears it as it would hear a person)")
    t0 = time.time()
    subprocess.call(["paplay", WAV])

    # Wait for the utterance to be segmented, decided on, and answered.
    fail = []
    heard = None
    for _ in range(60):
        time.sleep(1.0)
        st = ask({"cmd": "ears"})
        if st.get("commands", 0) > before.get("commands", 0):
            heard = st.get("last_heard")
            print("\n   heard after %.1fs: %r" % (time.time() - t0, heard))
            break
        if st.get("wakes", 0) > before.get("wakes", 0) and heard is None:
            heard = "(woke, still waiting for the command)"
    else:
        st = ask({"cmd": "ears"})
        print("\n   nothing was taken as a command.")
        print("   utterances=%s wakes=%s commands=%s"
              % (st.get("utterances"), st.get("wakes"), st.get("commands")))
        if st.get("wakes", 0) == before.get("wakes", 0):
            fail.append("the wake word was never recognised over the air")
        else:
            fail.append("it woke but never got a command out of the audio")

    if not fail:
        print("\n   waiting for it to answer out loud ...")
        for _ in range(90):
            time.sleep(1.0)
            s = ask({"cmd": "status"})
            if s.get("speaking"):
                print("   it is speaking.")
                break
        else:
            fail.append("it heard the command but never said anything")

    print("\n-- what the service logged --")
    out = subprocess.run(["journalctl", "-u", "tek-voice", "-n", "25",
                          "--no-pager", "-o", "cat"],
                         stdout=subprocess.PIPE).stdout.decode()
    for l in out.splitlines():
        if any(k in l for k in ("ears:", "event speech", "said ")):
            print("   | " + l)

    print("\nEARS E2E " + ("OK" if not fail else "FAILED: " + "; ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
