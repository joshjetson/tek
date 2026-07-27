"""The camera must attach when it appears - and STAY attached across a replug.

Two defects, both invisible on a running machine, both found only by a real
unplug:

1. Boot: systemd starts us before USB enumeration finishes, so /dev/video0 does
   not exist yet. The original code checked once during load() and gave up
   permanently.

2. Replug: the attach loop returned at the first success, on the stated grounds
   that the tracker handled unplug/replug by itself. It did not - it reopened
   using the device index it was constructed with, and its worker was wrapped
   in a bare `except Exception: pass`. Swapping the camera for a different one
   left the head permanently still, with nothing in the log to say why.

Devices are faked here rather than unplugged: the real module would fight the
running service for the camera, and this is about the attach and supervision
logic, not about capture.
"""
import os
import sys
import threading
import time
import types

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tekdromo
from tekdromo import app

FAIL = []


def check(name, cond, extra=""):
    print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


# -- a fake camera module --------------------------------------------------
present = []                    # which /dev/videoN exist right now
started = []                    # every Tracker that was start()ed

fake = types.ModuleType("tekdromo.camera")


class _Tracker(object):
    def __init__(self, device=None, **kw):
        # device is now a PREFERENCE, not an assumption: the real Tracker
        # re-discovers the node itself on every open.
        self.device = device
        self.alive = True
        self._t = None

    def start(self):
        started.append(self)
        self._t = self          # stands in for the worker thread
        return self

    def is_alive(self):
        return self.alive

    def stop(self):
        self.alive = False


fake.Tracker = _Tracker
fake.Follower = lambda: "follower"
fake.video_devices = lambda: list(present)
sys.modules["tekdromo.camera"] = fake
tekdromo.camera = fake

# A Display without a framebuffer: only the attach loop is under test.
d = app.Display.__new__(app.Display)
d.running = True
d.cam = d.follow = None

th = threading.Thread(target=d._wait_for_camera, kwargs={"period": 0.05})
th.daemon = True
th.start()

time.sleep(0.4)
check("no device -> does not attach, keeps polling",
      d.cam is None and th.is_alive())

# -- boot: the device enumerates late --------------------------------------
present[:] = [1]                # NOT video0: after a USB reset it came back as 1
time.sleep(0.4)
check("device appears -> tracker attached", d.cam is not None)
check("follower set before tracker (pose() guards on cam)",
      d.follow == "follower")
check("started exactly once", len(started) == 1, len(started))

# The supervisor must NOT exit here. The previous version did, and that is
# the entire defect - this check used to assert the opposite.
check("the supervisor keeps running after attaching", th.is_alive())

# -- steady state: it must not thrash --------------------------------------
first = d.cam
time.sleep(0.5)
check("a healthy tracker is left alone",
      d.cam is first and len(started) == 1, len(started))

# -- the tracker dies (the failure that used to be permanent) --------------
d.cam.alive = False
time.sleep(0.5)
check("a dead tracker is noticed and replaced", d.cam is not first)
check("the replacement was started", len(started) == 2, len(started))
check("the old tracker was stopped", first.alive is False)

# -- unplug: no devices at all ---------------------------------------------
second = d.cam
present[:] = []
second.alive = False
time.sleep(0.4)
check("with no device present it does not build a tracker",
      len(started) == 2, len(started))
check("and it is still supervising, not exited", th.is_alive())

# -- replug at a DIFFERENT index -------------------------------------------
present[:] = [2]
time.sleep(0.4)
check("replug at a new index re-attaches", len(started) == 3, len(started))
check("the new tracker discovers the node itself, not a stale index",
      d.cam.device is None, d.cam.device)

d.running = False
print("BOOT CAMERA " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
