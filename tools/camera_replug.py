#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Really unplug the camera, and check the face starts tracking again.

tests/boot_camera.py fakes the devices, so it proves the supervision logic and
nothing about the hardware. This does the opposite: it deauthorizes the camera
on the USB bus, which makes the kernel tear the device down and remove
/dev/videoN exactly as pulling the plug does, then authorizes it again.

That distinction matters here. The defect this exists to catch survived a test
suite that passed: the tracker reopened with the device index it was built
with, and its worker was wrapped in a bare `except Exception: pass`, so a
swapped camera left the head still and the log empty. Only a real replug shows
that.

Health is judged by the snapshot the display writes from live frames, not by
"the service is active" - the whole failure mode was a service that stayed
perfectly active while its tracker was dead.

    sudo python3 tools/camera_replug.py
    sudo python3 tools/camera_replug.py --hold    # come back on a NEW node

--hold keeps the old /dev/videoN open across the unplug so the kernel cannot
reuse that minor number, which forces the camera to re-enumerate one index
higher. That is the case that actually broke: the camera was swapped, came back
as video1, and a tracker reopening video0 never found it again. Verified to
produce exactly that - `camera: /dev/video1 delivering (open #3)`.
"""
import argparse
import os
import subprocess
import sys
import time

SNAPSHOT = os.path.expanduser("~/.cache/tekdromo/seen.jpg")
SYSFS = "/sys/bus/usb/devices"
VENDOR_PRODUCT = ("046d", "085c")          # Logitech C922 Pro Stream


def find_usb():
    """sysfs name of the camera, e.g. "1-2"."""
    for name in sorted(os.listdir(SYSFS)):
        base = os.path.join(SYSFS, name)
        try:
            with open(os.path.join(base, "idVendor")) as f:
                v = f.read().strip()
            with open(os.path.join(base, "idProduct")) as f:
                p = f.read().strip()
        except IOError:
            continue
        if (v, p) == VENDOR_PRODUCT:
            return name
    return None


def authorize(name, value):
    with open(os.path.join(SYSFS, name, "authorized"), "w") as f:
        f.write("1" if value else "0")


def nodes():
    return sorted(n for n in os.listdir("/dev") if n.startswith("video"))


def snap_age():
    try:
        return time.time() - os.path.getmtime(SNAPSHOT)
    except OSError:
        return None


def wait_fresh(limit=45.0, need=2):
    """Wait until the display writes NEW snapshots from live frames.

    Two of them, not one: a single fresh file could be the one written just
    before the unplug. Two consecutive new mtimes mean frames are genuinely
    flowing again.
    """
    seen, last = 0, None
    end = time.time() + limit
    while time.time() < end:
        try:
            m = os.path.getmtime(SNAPSHOT)
        except OSError:
            m = None
        if m is not None and m != last:
            if last is not None:
                seen += 1
                if seen >= need:
                    return time.time() - (end - limit)
            last = m
        time.sleep(0.5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", action="store_true",
                    help="hold the current node open across the unplug, so the "
                         "camera is forced to come back on a different index")
    a = ap.parse_args()

    if os.geteuid() != 0:
        print("needs root (it writes to sysfs)")
        return 2
    name = find_usb()
    if name is None:
        print("camera %s:%s not on the USB bus" % VENDOR_PRODUCT)
        return 2
    if subprocess.run(["systemctl", "is-active", "tek-display"],
                      stdout=subprocess.PIPE).stdout.decode().strip() != "active":
        print("tek-display is not running - nothing to test")
        return 2

    fail = []
    print("camera is USB %s, nodes=%s" % (name, nodes()))

    print("\n-- baseline: is it tracking now? --")
    if wait_fresh(limit=30.0) is None:
        print("FAIL: no fresh snapshots before we even started")
        return 1
    print("   snapshots are updating (age %.1fs)" % (snap_age() or -1))

    print("\n-- unplug --")
    before = nodes()
    held = None
    if a.hold and before:
        # An open fd keeps the minor number allocated even after the device is
        # gone, so the replugged camera has to take the next index up.
        held = os.open("/dev/" + before[0], os.O_RDONLY)
        print("   holding /dev/%s open to force a new index" % before[0])
    authorize(name, False)
    for _ in range(20):
        time.sleep(0.5)
        if nodes() != before:
            break
    print("   nodes: %s -> %s" % (before, nodes()))
    if nodes() == before:
        fail.append("the device node never went away - not a real unplug")

    time.sleep(6.0)                       # let the tracker notice and give up

    print("\n-- replug --")
    t0 = time.time()
    authorize(name, True)
    for _ in range(40):
        time.sleep(0.5)
        if nodes():
            break
    print("   nodes came back as %s after %.1fs" % (nodes(), time.time() - t0))
    if held is not None:
        if nodes() == before:
            print("   (the index did not actually change - nothing forced)")
        os.close(held)

    took = wait_fresh(limit=60.0)
    if took is None:
        fail.append("tracking did NOT resume after the replug")
    else:
        print("   tracking resumed %.1fs after the replug" % (time.time() - t0))

    print("\n-- what the display logged --")
    out = subprocess.run(["journalctl", "-u", "tek-display", "-n", "40",
                          "--no-pager", "-o", "cat"],
                         stdout=subprocess.PIPE).stdout.decode()
    for line in out.splitlines():
        if "camera" in line.lower():
            print("   | " + line)

    print("\nCAMERA REPLUG " + ("OK" if not fail else "FAILED: " + "; ".join(fail)))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
