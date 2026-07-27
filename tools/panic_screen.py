#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the console actually come BACK? Read the panel and find out.

The user-visible promise is not "the service stopped", it is "I can see a
terminal again". Those are different claims, and only the second one matters at
2am with no network. Display.close deliberately leaves the last frame on the
panel, so if the VT switch failed to make fbcon repaint, the face would still
be sitting there over a console that is technically running - which looks
exactly like a panic key that did nothing.

So: sample /dev/fb0 before and after, and also time how long the stop takes,
because a panic key that needs twenty seconds gets pressed again and again.

    sudo python3 tools/panic_screen.py
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy as np

from tekdromo import framebuffer, panic


def sample():
    """(lit pixel count, mean level) for the whole panel."""
    fd, mm, screen, w, h = framebuffer.open_screen()
    try:
        img = np.array(screen[:, :, :3])
        return int((img.max(axis=2) > 40).sum()), float(img.mean()), w * h
    finally:
        del screen
        mm.close()
        os.close(fd)


def active(unit):
    return subprocess.run(["systemctl", "is-active", unit],
                          stdout=subprocess.PIPE).stdout.decode().strip()


def main():
    if os.geteuid() != 0:
        print("needs root")
        return 2
    fail = []

    if active("tek-display") != "active":
        subprocess.call(["systemctl", "start", "tek-display"])
        time.sleep(8)
    time.sleep(2)

    lit_before, mean_before, total = sample()
    print("display up:   %d lit px (%.2f%% of panel), mean level %.2f"
          % (lit_before, 100.0 * lit_before / total, mean_before))
    if lit_before < 500:
        print("  (panel looks blank already - is the display really drawing?)")

    t0 = time.time()
    subprocess.call(["systemctl", "stop", "tek-display"])
    stop_s = time.time() - t0
    print("stop took:    %.2f s" % stop_s)
    if stop_s > 3.0:
        fail.append("stopping took %.1fs - too slow for a panic key" % stop_s)

    # The frame is STILL on the panel at this point, on purpose. Prove it,
    # because that is precisely why restore_console has to exist.
    lit_held, _, _ = sample()
    print("after stop:   %d lit px  <- the frame is still there, as designed"
          % lit_held)

    t0 = time.time()
    panic.restore_console()
    print("repaint took: %.2f s" % (time.time() - t0))
    time.sleep(1.0)
    lit_after, mean_after, _ = sample()
    print("after chvt:   %d lit px (%.2f%% of panel), mean level %.2f"
          % (lit_after, 100.0 * lit_after / total, mean_after))

    # A text console is overwhelmingly black with a few lit glyphs. The face
    # fills far more of the panel than a login prompt does.
    if lit_after >= lit_held * 0.5:
        fail.append("the panel still looks like the face (%d -> %d lit px)"
                    % (lit_held, lit_after))
    else:
        print("console repainted: %d -> %d lit px (%.0f%% of the image gone)"
              % (lit_held, lit_after, 100.0 * (1 - float(lit_after) / max(lit_held, 1))))

    subprocess.call(["systemctl", "start", "tek-display"])
    time.sleep(8)
    print("restored:     tek-display=%s" % active("tek-display"))
    print("PANIC SCREEN " + ("OK" if not fail else "FAILED: " + "; ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
