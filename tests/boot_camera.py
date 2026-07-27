"""The camera must attach when it appears, not only if it is already there.

This is the boot case: systemd starts us before USB enumeration finishes, so
/dev/video0 does not exist yet. The old code checked once during load() and
gave up permanently - a defect invisible on a running machine, because by the
time anyone tests it the device is long since present. So the device is faked
here rather than unplugged.
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


def check(name, cond):
    print("  %-46s %s" % (name, "OK" if cond else "FAIL"))
    if not cond:
        FAIL.append(name)


# Fake camera module: the real one would fight the running service for
# /dev/video0, and this test is about the attach logic, not about capture.
started = []
fake = types.ModuleType("tekdromo.camera")


class _Tracker(object):
    def __init__(self, device=0, **kw):
        # Must accept the device index the real Tracker now takes. Without it
        # the constructor raised TypeError, which _wait_for_camera catches and
        # retries - so the failure looked like "the camera never appeared".
        self.device = device

    def start(self):
        started.append(self.device)
        return self


fake.Tracker = _Tracker
fake.Follower = lambda: "follower"
sys.modules["tekdromo.camera"] = fake
tekdromo.camera = fake

# A Display without a framebuffer: only the attach loop is under test.
d = app.Display.__new__(app.Display)
d.running = True
d.cam = d.follow = None

# The device index is NOT assumed to be 0: after a USB reset with the old node
# still held, this camera came back as /dev/video1 and a hardcoded video0 would
# never have found it again.
present = [False]
real_find = app._find_camera
app._find_camera = lambda: (1 if present[0] else None)

th = threading.Thread(target=d._wait_for_camera, kwargs={"period": 0.05})
th.daemon = True
th.start()

time.sleep(0.4)
check("no device -> does not attach, keeps polling", d.cam is None and th.is_alive())

present[0] = True                       # the device enumerates
time.sleep(0.4)
check("device appears -> tracker attached", d.cam is not None)
check("follower set before tracker (pose() guards on cam)", d.follow == "follower")
check("attach thread exits once attached", not th.is_alive())
check("started exactly once", len(started) == 1)
check("the discovered index is passed through, not assumed to be 0",
      started == [1])

d.running = False
app._find_camera = real_find
print("BOOT CAMERA " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
