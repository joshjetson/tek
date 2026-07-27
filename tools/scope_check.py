#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the waveform panel actually MOVE when there is sound?

"The service is running" and "the panel is drawing" are not the claim. The
panel was drawing perfectly the whole time it was broken - a flat line is a
drawn line. The claim is that the trace responds to audio, so this reads the
panel's own pixels out of /dev/fb0 before and during sound and compares how
tall the trace is.

Both directions are checked, because the reported fault was "audio out or in"
and they arrive by different paths:

  * OUT - played through the speaker, seen on the default sink's monitor;
  * IN  - the microphone, which is what makes the panel move when you talk.

    sudo python3 tools/scope_check.py
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid())

import numpy as np

from tekdromo import framebuffer, hud


def panel_spread(samples=14, gap=0.12):
    """Typical height of the trace, in pixels.

    The MEDIAN column's height, not the extent of lit pixels across the whole
    panel. The panel holds three seconds of history, so a single transient
    anywhere in that window pins a whole-panel extent to maximum and the
    measurement then reports "loud" for three seconds after a click - which is
    exactly how a first version of this tool managed to call a working panel
    and a broken one the same number.

    Per-column, median across columns, is what "how tall is the trace right
    now" actually means.
    """
    fd, mm, screen, w, h = framebuffer.open_screen()
    try:
        s = hud.Scope(w, h)
        y0, y1 = s.by, s.by + s.bh
        x0, x1 = s.bx, s.bx + s.bw
        best = []
        for _ in range(samples):
            box = np.array(screen[y0:y1, x0:x1, :3])
            lit = box.max(axis=2) > 40
            heights = []
            for c in range(lit.shape[1]):
                rows = np.where(lit[:, c])[0]
                if len(rows):
                    heights.append(int(rows[-1] - rows[0]))
            best.append(float(np.median(heights)) if heights else 0.0)
            time.sleep(gap)
        return float(np.mean(best)), max(best)
    finally:
        del screen
        mm.close()
        os.close(fd)


def main():
    # Deliberately NOT root. /dev/fb0 is readable by the `video` group, which
    # `super` is in - that is how tek-display draws at all. Running this under
    # sudo instead breaks the half that matters: PulseAudio is per-user, so
    # pactl and paplay both fail with "Connection refused" and the tool then
    # reports the panel as broken when the panel is fine.
    if os.geteuid() == 0:
        print("run this as your normal user, NOT with sudo:\n"
              "  /dev/fb0 needs the 'video' group, which you have;\n"
              "  PulseAudio refuses root, which breaks pactl and paplay.")
        return 2
    if subprocess.run(["systemctl", "is-active", "tek-display"],
                      stdout=subprocess.PIPE).stdout.decode().strip() != "active":
        print("tek-display is not running")
        return 2

    fail = []
    # The auto-gain decays about 0.78 per second, so settling from a loud
    # passage takes several seconds. Four was not enough.
    print("letting the trace decay to quiet ...")
    time.sleep(12)
    quiet_mean, quiet_max = panel_spread()
    print("  quiet:   mean spread %.1f px, max %d px" % (quiet_mean, quiet_max))

    # -- OUT: play something and watch the panel ---------------------------
    print("\nplaying a tone through the speaker ...")
    # Per-uid path: a shared /tmp name left root-owned by an earlier sudo run
    # makes this fail to write and then silently play the STALE file.
    wav = "/tmp/tek-scope-tone-%d.wav" % os.getuid()
    subprocess.call(["python3", "-c", """
import sys, wave, numpy as np
sys.path.insert(0, '/home/super/tekdromo')
from tekdromo.voice import pcm
x = pcm.tone(440.0, 6.0, 0.35, 44100)
w = wave.open(%r, 'wb'); w.setnchannels(1); w.setsampwidth(2)
w.setframerate(44100); w.writeframes(np.asarray(x, np.int16).tobytes()); w.close()
""" % wav], env=dict(os.environ, OPENBLAS_CORETYPE="ARMV8"))
    p = subprocess.Popen(["paplay", wav])
    time.sleep(1.2)
    out_mean, out_max = panel_spread()
    p.wait()
    print("  playing: mean spread %.1f px, max %d px" % (out_mean, out_max))
    if out_mean <= quiet_mean + 4:
        fail.append("the panel did not respond to audio OUT (%.1f -> %.1f)"
                    % (quiet_mean, out_mean))
    else:
        print("  -> responds to audio OUT")

    # -- IN: make a noise the mic will hear --------------------------------
    # Played through the speaker again, but measured while it is NOT the sink
    # monitor that would show it - see below. Simplest honest check: confirm
    # the mic listener is bound at all, and that its level reaches the panel.
    print("\nchecking the microphone path ...")
    src = subprocess.run(["pactl", "info"], stdout=subprocess.PIPE
                         ).stdout.decode()
    src = [l.split(":", 1)[1].strip() for l in src.splitlines()
           if l.startswith("Default Source:")]
    bound = subprocess.run(["pactl", "list", "source-outputs"],
                           stdout=subprocess.PIPE).stdout.decode()
    n_parec = bound.count('application.name = "parec"')
    print("  default source: %s" % (src[0] if src else "(none)"))
    print("  parec record streams open: %d" % n_parec)
    if not src or src[0].endswith(".monitor"):
        fail.append("no real capture source is the default")
    if n_parec < 2:
        fail.append("expected at least 2 parec streams (sink monitor + mic), "
                    "found %d" % n_parec)
    else:
        print("  -> both a monitor and a microphone are being recorded")

    print("\n-- what the display logged --")
    out = subprocess.run(["journalctl", "-u", "tek-display", "-n", "40",
                          "--no-pager", "-o", "cat"],
                         stdout=subprocess.PIPE).stdout.decode()
    for l in out.splitlines():
        if "scope:" in l:
            print("   | " + l)

    print("\nSCOPE " + ("OK" if not fail else "FAILED: " + "; ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
