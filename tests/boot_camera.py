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
    def start(self):
        started.append(time.time())
        return self


fake.Tracker = _Tracker
fake.Follower = lambda: "follower"
sys.modules["tekdromo.camera"] = fake
tekdromo.camera = fake

# A Display without a framebuffer: only the attach loop is under test.
d = app.Display.__new__(app.Display)
d.running = True
d.cam = d.follow = None

present = [False]
real_exists = os.path.exists
app.os.path.exists = lambda p: present[0] if p == "/dev/video0" else real_exists(p)

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

d.running = False
app.os.path.exists = real_exists
print("BOOT CAMERA " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
