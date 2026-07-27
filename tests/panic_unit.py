# -*- coding: utf-8 -*-
"""The escape hatch. If this is wrong, the machine becomes a brick.

Two halves:

* The chord state machine, exhaustively, as pure logic.
* An END-TO-END check through the real kernel input stack: a virtual keyboard
  is created with uinput *after* the watcher is already running, ESC is pressed
  for real, and the watcher must fire. That ordering is the whole point - the
  keyboard gets plugged in after things have gone wrong, so a watcher that
  enumerated devices once at startup would pass every unit test and still be
  useless on the night you need it.

The uinput half needs root and is skipped without it, loudly.
"""
import fcntl
import os
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tekdromo import panic

FAIL = []


def check(name, cond, extra=""):
    print("  %-56s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


# -- the chord -------------------------------------------------------------
c = panic.Chord(window=2.0)
check("one press does nothing", c.press(0.0) is None)
check("two presses do nothing", c.press(0.1) is None)
check("three presses inside the window stop the display",
      c.press(0.2) == "stop")

c = panic.Chord(window=2.0)
for t in (0.0, 0.1, 0.2, 0.3):
    r = c.press(t)
check("carrying on past three does not re-fire on the fourth", r is None, r)
check("the fifth press escalates to silence", c.press(0.4) == "quiet")
check("the counter resets after escalating", c.press(0.5) is None)

# Presses spread out must NOT fire - this is what stops an ordinary session at
# the console (or a child at the keyboard) from killing the display.
c = panic.Chord(window=2.0)
check("presses outside the window never fire",
      [c.press(t) for t in (0.0, 1.5, 3.0, 4.5, 6.0, 7.5)] == [None] * 6)

c = panic.Chord(window=2.0)
c.press(0.0)
c.press(0.1)
check("a stale press ages out instead of counting",
      c.press(2.5) is None, c.times)

# The window slides: three presses inside ANY 2s window is the contract, even
# if they are spread right across it.
c = panic.Chord(window=2.0)
check("a sliding window still catches three spread across two seconds",
      [c.press(t) for t in (0.0, 1.0, 1.9)][-1] == "stop")
# ...but the span really is measured, so three presses spanning 2.1s do not.
c = panic.Chord(window=2.0)
check("three presses spanning longer than the window do not fire",
      [c.press(t) for t in (0.0, 1.9, 2.1)][-1] is None)

check("the chord defaults are the documented ones",
      (panic.STOP_AT, panic.QUIET_AT) == (3, 5),
      (panic.STOP_AT, panic.QUIET_AT))

# -- the event decoding ----------------------------------------------------
check("input_event is the size the kernel writes", panic.EVENT_SIZE == 24,
      panic.EVENT_SIZE)


def ev(typ, code, val):
    return struct.pack(panic.EVENT_FMT, 0, 0, typ, code, val)


fired = []
w = panic.Watcher(lambda what: fired.append(what))
# A real fd, because drop() closes it - a None here would test a state the
# kernel can never produce and hide whether the close path actually works.
_r, _w = os.pipe()
w.fds["/fake"] = _r


class FakeRead(object):
    """Stand in for os.read on an event device."""

    def __init__(self, blob):
        self.blob = blob

    def __call__(self, fd, n):
        return self.blob


real_read = os.read
try:
    # A realistic burst: press/release pairs with the SYN events the kernel
    # interleaves, all delivered in ONE read - which is what actually happens
    # when someone hits a key three times quickly.
    blob = b"".join([ev(panic.EV_KEY, panic.KEY_ESC, 1), ev(0, 0, 0),
                     ev(panic.EV_KEY, panic.KEY_ESC, 0), ev(0, 0, 0)] * 3)
    os.read = FakeRead(blob)
    w.feed("/fake", 0.0)
    check("three presses arriving in a single read still fire",
          fired == ["stop"], fired)

    # Autorepeat: holding ESC down emits value 2 forever. If that counted, the
    # display would die every time someone rested a book on the keyboard.
    fired[:] = []
    w.chord = panic.Chord()
    os.read = FakeRead(ev(panic.EV_KEY, panic.KEY_ESC, 1)
                       + ev(panic.EV_KEY, panic.KEY_ESC, 2) * 40)
    w.feed("/fake", 0.0)
    check("autorepeat does not count as presses", fired == [], fired)

    # Other keys must be ignored entirely.
    fired[:] = []
    w.chord = panic.Chord()
    os.read = FakeRead(b"".join(ev(panic.EV_KEY, k, 1) for k in (30, 48, 46)))
    w.feed("/fake", 0.0)
    check("other keys are ignored", fired == [], fired)

    # A torn tail (fewer than 24 bytes left) must not crash or misdecode.
    fired[:] = []
    w.chord = panic.Chord()
    os.read = FakeRead(ev(panic.EV_KEY, panic.KEY_ESC, 1) * 2 + b"\x00" * 9)
    w.feed("/fake", 0.0)
    check("a partial trailing event is discarded, not misread",
          fired == [], fired)

    # An unplugged device must be dropped, not raise.
    def boom(fd, n):
        raise OSError(19, "No such device")

    os.read = boom
    w.feed("/fake", 0.0)
    check("a device unplugged mid-read is dropped quietly",
          "/fake" not in w.fds, list(w.fds))
finally:
    os.read = real_read


# -- end to end through the real kernel input stack ------------------------
UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502


def make_keyboard():
    """A real virtual keyboard via uinput. Returns the fd."""
    fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
    fcntl.ioctl(fd, UI_SET_EVBIT, panic.EV_KEY)
    fcntl.ioctl(fd, UI_SET_KEYBIT, panic.KEY_ESC)
    # struct uinput_user_dev: name[80], input_id{4 x u16}, ff_effects_max,
    # then abs{max,min,fuzz,flat}[64] each.
    dev = struct.pack("80sHHHHi" + "i" * 256, b"tekdromo-panic-test",
                      0x03, 0x1234, 0x5678, 1, 0, *([0] * 256))
    os.write(fd, dev)
    fcntl.ioctl(fd, UI_DEV_CREATE)
    return fd


def tap(fd, code=panic.KEY_ESC):
    for val in (1, 0):
        os.write(fd, struct.pack(panic.EVENT_FMT, 0, 0, panic.EV_KEY, code, val))
        os.write(fd, struct.pack(panic.EVENT_FMT, 0, 0, 0, 0, 0))   # SYN
    time.sleep(0.05)


if os.geteuid() != 0:
    print("  (not root - skipping the real-keyboard test; run with sudo)")
elif not os.path.exists("/dev/uinput"):
    print("  (no /dev/uinput - skipping the real-keyboard test)")
else:
    seen = []
    watcher = panic.Watcher(lambda what: seen.append(what), rescan=0.5)
    t = threading.Thread(target=watcher.run)
    t.daemon = True
    t.start()
    time.sleep(1.0)                       # watcher is up, no keyboard yet

    kb = None
    try:
        kb = make_keyboard()              # <- plugged in AFTER the watcher
        # Give udev time to create the node and the watcher time to rescan.
        deadline = time.time() + 6.0
        while time.time() < deadline and not any(
                "tekdromo-panic-test" in open(p).read()
                for p in ["/proc/bus/input/devices"]):
            time.sleep(0.2)
        time.sleep(1.2)

        check("the watcher picked up a keyboard plugged in after it started",
              len(watcher.fds) > 0, list(watcher.fds))

        for _ in range(3):
            tap(kb)
        time.sleep(0.8)
        check("ESC x3 on a real device fires the panic", seen == ["stop"], seen)

        for _ in range(2):
            tap(kb)
        time.sleep(0.8)
        check("two more presses escalate to silence",
              seen == ["stop", "quiet"], seen)

        # And the negative: slow presses must not fire.
        seen[:] = []
        watcher.chord = panic.Chord()
        for _ in range(4):
            tap(kb)
            time.sleep(1.1)
        check("slow presses on a real device never fire", seen == [], seen)
    finally:
        watcher.running = False
        if kb is not None:
            try:
                fcntl.ioctl(kb, UI_DEV_DESTROY)
            except Exception:
                pass
            os.close(kb)

print("PANIC " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
