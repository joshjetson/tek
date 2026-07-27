#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The way out.

The display owns /dev/fb0 and paints straight over the text console, so while
it runs there is no visible terminal. Worse, it deliberately *keeps* the last
frame when it exits (see Display.close - it makes restarts invisible), so even
stopping it leaves the face sitting on the panel. Add a machine whose network
did not come up and there is no way in at all: the only recovery is to
blind-type a login you cannot see. That happened. This module exists so it
cannot happen again.

    ESC ESC ESC     stop the display, hand the console back
    ESC x5          stop the voice as well

Every design choice here is about the failure case, not the happy path:

* **It is a separate process.** A panic key living inside the thing you are
  escaping from is not a panic key: the case you need it for is precisely the
  one where that process is wedged.
* **It imports nothing from this project** - not even numpy. The escape hatch
  must not be able to fail for the same reason the thing it rescues failed.
* **It rescans /dev/input.** The keyboard is normally plugged in *after* things
  have gone wrong, so enumerating once at startup would miss the only keyboard
  that matters.
* **Autorepeat (value 2) does not count.** Otherwise leaning on the ESC key
  fires it.
* **The window is measured on the monotonic clock.** This box sets its clock
  from the network a minute into boot; a wall-clock jump mid-chord would
  otherwise either swallow the presses or fire on its own.
* **Stopping the display is only half the job** - the console is forced to
  repaint by switching VT away and back.
"""
import glob
import os
import select
import struct
import subprocess
import sys
import time

# struct input_event: two longs of timestamp, then type/code/value.
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)         # 24 on aarch64
EV_KEY = 0x01
KEY_ESC = 1
PRESS = 1                                       # 0 release, 2 autorepeat

STOP_AT = 3                 # ESC x3 -> console back
QUIET_AT = 5                # ESC x5 -> and silence
WINDOW = 2.0                # seconds the presses must fall inside
RESCAN = 2.0                # how often to look for a newly plugged keyboard

BANNER = ("\n\n*** TEKDROMO PANIC - display stopped, the console is yours ***\n"
          "    bring it back with:  sudo systemctl start tek-display\n\n")


class Chord(object):
    """Counts presses inside a sliding window.

    Pure and side-effect free so the thing that decides to kill your display
    can be tested exhaustively without a keyboard, a display, or root.
    """

    def __init__(self, window=WINDOW, stop_at=STOP_AT, quiet_at=QUIET_AT):
        self.window = window
        self.stop_at = stop_at
        self.quiet_at = quiet_at
        self.times = []

    def press(self, when):
        """-> None, "stop" or "quiet"."""
        self.times = [t for t in self.times if when - t < self.window]
        self.times.append(when)
        n = len(self.times)
        # Order matters: the escalation is reached by carrying on past the
        # first trigger, so "quiet" must be tested before "stop".
        if n >= self.quiet_at:
            self.times = []
            return "quiet"
        if n == self.stop_at:
            return "stop"
        return None


class Watcher(object):
    """Watches every input device for the chord, including ones plugged in
    later. `on_fire(what)` is called with "stop" or "quiet"."""

    def __init__(self, on_fire, rescan=RESCAN, chord=None):
        self.on_fire = on_fire
        self.rescan = rescan
        self.chord = chord if chord is not None else Chord()
        self.fds = {}                           # path -> fd
        self.running = True

    # -- devices -----------------------------------------------------------
    def scan(self):
        """Open any new event device, forget any that vanished.

        Every device is watched, not just the ones that look like keyboards.
        Filtering would need EVIOCGBIT and would then have to be right about
        what a keyboard is - and this box already has a webcam that registers
        as one. Nothing but a keyboard ever sends KEY_ESC, so the filter buys
        nothing and could only exclude the device you need.
        """
        for path in sorted(glob.glob("/dev/input/event*")):
            if path in self.fds:
                continue
            try:
                self.fds[path] = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                pass                            # not readable; skip quietly
        for path in list(self.fds):
            if not os.path.exists(path):
                self.drop(path)

    def drop(self, path):
        try:
            os.close(self.fds.pop(path))
        except (OSError, KeyError):
            pass

    def feed(self, path, now):
        """Read whatever is pending on one device and count any ESC presses."""
        try:
            data = os.read(self.fds[path], EVENT_SIZE * 64)
        except (OSError, KeyError):
            self.drop(path)                     # unplugged between select and read
            return
        for i in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
            _, _, typ, code, val = struct.unpack(EVENT_FMT,
                                                 data[i:i + EVENT_SIZE])
            if typ == EV_KEY and code == KEY_ESC and val == PRESS:
                what = self.chord.press(now)
                if what:
                    self.on_fire(what)

    def run(self):
        last_scan = 0.0
        while self.running:
            now = time.monotonic()
            if now - last_scan >= self.rescan:
                self.scan()
                last_scan = now
            if not self.fds:
                time.sleep(0.2)
                continue
            try:
                ready = select.select(list(self.fds.values()), [], [],
                                      self.rescan)[0]
            except (select.error, OSError, ValueError):
                # A device disappeared out from under select. Rebuild and
                # carry on; a panic key that dies when you unplug a mouse is
                # worse than useless.
                for path in list(self.fds):
                    self.drop(path)
                last_scan = 0.0
                continue
            by_fd = dict((fd, path) for path, fd in self.fds.items())
            for fd in ready:
                path = by_fd.get(fd)
                if path is not None:
                    self.feed(path, time.monotonic())


# -- the action -------------------------------------------------------------
def _run(cmd):
    try:
        return subprocess.call(cmd) == 0
    except OSError:
        return False


def restore_console(tty="/dev/tty1"):
    """Make the text console visible again.

    Stopping the display is not enough. It leaves the last frame on the panel
    on purpose, and fbcon has no idea its pixels were overwritten, so it never
    repaints. Switching VT away and back is what forces a full redraw.
    """
    here = "1"
    try:
        here = subprocess.check_output(["fgconsole"]).decode().strip() or "1"
    except Exception:
        pass
    other = "2" if here != "2" else "3"
    _run(["chvt", other])
    time.sleep(0.4)
    _run(["chvt", here])
    try:
        with open(tty, "w") as f:
            f.write(BANNER)
    except (IOError, OSError):
        pass


def panic(what="stop"):
    """Stop the display (and optionally the voice), then repaint the console.

    `systemctl stop` beats the unit's Restart=always: an explicit stop is not
    a failure, so systemd will not bring it back.
    """
    units = ["tek-display.service"]
    if what == "quiet":
        units.append("tek-voice.service")
    _run(["systemctl", "stop"] + units)
    restore_console()
    return units


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="tekdromo.panic",
        description="ESC x%d stops the display; ESC x%d silences it too."
                    % (STOP_AT, QUIET_AT))
    ap.add_argument("--now", choices=("stop", "quiet"),
                    help="panic immediately instead of watching the keyboard")
    ap.add_argument("--window", type=float, default=WINDOW,
                    help="seconds the presses must fall within")
    a = ap.parse_args(argv)

    if a.now:
        print("panic: stopped %s" % ", ".join(panic(a.now)), flush=True)
        return 0

    def fire(what):
        print("panic: ESC x%d -> %s" % (STOP_AT if what == "stop" else QUIET_AT,
                                        what), flush=True)
        print("panic: stopped %s" % ", ".join(panic(what)), flush=True)

    w = Watcher(fire, chord=Chord(window=a.window))
    print("panic watcher up: ESC x%d = console back, ESC x%d = also silence"
          % (STOP_AT, QUIET_AT), flush=True)
    w.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
