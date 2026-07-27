#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fire the panic key at the LIVE system and put everything back.

tests/panic_unit.py proves the watcher decodes keys and fires its callback.
That is not the same as proving the installed service stops the display: the
service runs as root in its own cgroup, and `systemctl stop` from inside it is
the step most likely to be quietly denied. So this presses ESC three times on
a real virtual keyboard and checks that tek-display actually went away.

Disruptive by design - it stops the display for a few seconds and restarts it.
Run it deliberately, as root:

    sudo python3 tools/panic_e2e.py
"""
import fcntl
import os
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tekdromo import panic

UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502


def active(unit):
    out = subprocess.run(["systemctl", "is-active", unit],
                         stdout=subprocess.PIPE).stdout
    return out.decode().strip()


def keyboard():
    fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
    fcntl.ioctl(fd, UI_SET_EVBIT, panic.EV_KEY)
    fcntl.ioctl(fd, UI_SET_KEYBIT, panic.KEY_ESC)
    os.write(fd, struct.pack("80sHHHHi" + "i" * 256, b"tekdromo-panic-e2e",
                             0x03, 0x1234, 0x5678, 1, 0, *([0] * 256)))
    fcntl.ioctl(fd, UI_DEV_CREATE)
    return fd


def tap(fd):
    for val in (1, 0):
        os.write(fd, struct.pack(panic.EVENT_FMT, 0, 0,
                                 panic.EV_KEY, panic.KEY_ESC, val))
        os.write(fd, struct.pack(panic.EVENT_FMT, 0, 0, 0, 0, 0))
    time.sleep(0.06)


def main():
    if os.geteuid() != 0:
        print("needs root (it drives /dev/uinput)")
        return 2

    fail = []
    print("before:  tek-panic=%s  tek-display=%s"
          % (active("tek-panic"), active("tek-display")))
    if active("tek-panic") != "active":
        print("FAIL: the panic service is not running - nothing to test")
        return 1
    started_up = active("tek-display") == "active"
    if not started_up:
        subprocess.call(["systemctl", "start", "tek-display"])
        time.sleep(6)

    kb = keyboard()
    try:
        time.sleep(3.0)                 # let the service rescan and find it
        print("pressing ESC x3 ...")
        for _ in range(3):
            tap(kb)
        for _ in range(20):             # up to 10s for the stop to land
            time.sleep(0.5)
            if active("tek-display") != "active":
                break
        state = active("tek-display")
        print("after:   tek-display=%s" % state)
        if state == "active":
            fail.append("the display did NOT stop")
        # Restart=always must not resurrect it: an explicit stop is not a
        # failure, so systemd should leave it down.
        time.sleep(4)
        if active("tek-display") == "active":
            fail.append("the display came back on its own (Restart=always won)")
        else:
            print("stayed down through Restart=always: OK")
    finally:
        try:
            fcntl.ioctl(kb, UI_DEV_DESTROY)
        except Exception:
            pass
        os.close(kb)
        print("restarting the display ...")
        subprocess.call(["systemctl", "start", "tek-display"])
        time.sleep(8)
        print("restored: tek-display=%s" % active("tek-display"))

    print("PANIC E2E " + ("OK" if not fail else "FAILED: " + "; ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
