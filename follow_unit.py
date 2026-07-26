"""Unit-test the follower against synthetic input, before trusting a live run.
The windup bug only showed up on real data; this makes it reproducible."""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import tekcam

fail = 0

# 1. a steady target must be reached and HELD, at both a sane and a silly dt
for dt in (0.025, 0.25):
    f = tekcam.Follower()
    for _ in range(int(6.0 / dt)):
        gx, gy, hy = f.update(dict(present=True, x=0.35, y=-0.20), dt)
    ok = abs(gx - 0.35) < 0.05 and abs(gy + 0.20) < 0.05
    print("dt=%.3f steady -> gaze=(%+.3f,%+.3f) %s" % (dt, gx, gy, "ok" if ok else "FAIL"))
    fail += not ok

# 2. must never pin at the rail when the target is near centre
f = tekcam.Follower()
for i in range(400):
    gx, gy, _ = f.update(dict(present=True, x=0.05, y=0.10), 0.25)
ok = abs(gx) < 0.2 and abs(gy) < 0.25
print("large-dt near-centre -> gaze=(%+.3f,%+.3f) %s" % (gx, gy, "ok" if ok else "FAIL"))
fail += not ok

# 3. losing the face must return to centre, not stick
f = tekcam.Follower()
for _ in range(200):
    f.update(dict(present=True, x=0.9, y=0.7), 0.03)
for _ in range(300):
    gx, gy, hy = f.update(dict(present=False, x=0, y=0), 0.03)
ok = abs(gx) < 0.05 and abs(gy) < 0.05 and abs(hy) < 0.05
print("face lost -> gaze=(%+.3f,%+.3f) head=%+.3f %s" % (gx, gy, hy, "ok" if ok else "FAIL"))
fail += not ok

# 4. no overshoot on a step input (critical damping, not underdamped)
f = tekcam.Follower()
peak = 0.0
for _ in range(300):
    gx, _, _ = f.update(dict(present=True, x=1.0, y=0.0), 0.02)
    peak = max(peak, gx)
ok = peak <= 1.001
print("step response peak=%.4f (want <=1.0, no overshoot) %s" % (peak, "ok" if ok else "FAIL"))
fail += not ok

# 5. head must lag the eyes
f = tekcam.Follower()
for _ in range(12):
    gx, _, hy = f.update(dict(present=True, x=1.0, y=0.0), 0.03)
ok = abs(hy) < abs(gx)
print("head lags eyes: gaze=%+.3f head=%+.3f %s" % (gx, hy, "ok" if ok else "FAIL"))
fail += not ok

print("\n%s" % ("ALL PASS" if not fail else "%d FAILURES" % fail))
