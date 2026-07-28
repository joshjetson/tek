"""
The display application.

Its one job is that the screen never goes blank. Five ways it used to stop:

 1. Long black gap at startup - geometry is disk-cached, keyed by a hash of the
    source files so it self-invalidates. Cold ~4s, warm ~0.02s. The framebuffer
    is painted before any of that, so there is a boot screen instead of black.
 2. An exception killed the process - the frame loop catches per-frame errors,
    leaves the last good frame up, and continues.
 3. systemd's start limit - disabled in the unit ([Unit], not [Service]).
 4. The kernel console blanker - FBIOBLANK re-asserted every 30s.
 5. A wedged model - a watchdog thread repaints the last good frame.

Split out of the old tekrun.py, whose main() had grown to 130 lines doing
setup, camera wiring and rendering all at once.
"""
import argparse
import hashlib
import math
import os
import threading
import time
import traceback

import cv2
import numpy as np

from . import (contour, framebuffer, geometry, hud, phosphor, rig, speech,
               starfield, voice_link)
from .voice import bus

# How long a presence change must persist before it counts as an event. Below
# this, a detector glitch becomes an arrival.
WATCH_STABLE = 2.0

# How long an expression takes to arrive, and to let go. Both far slower than
# the 0.45s used for presence: presence is a fact that has just changed, a mood
# is something a face settles into. Letting go is slower still, because
# snapping back to neutral the instant the last word lands reads as the face
# switching off rather than as a conversation ending.
MOOD_BLEND = 0.9
MOOD_RELAX = 1.4
SNAPSHOT = os.path.join(os.path.expanduser("~/.cache/tekdromo"), "seen.jpg")
# Refresh the on-disk frame this often while someone is in view.
SNAPSHOT_EVERY = 8.0
# Face crops are written faster: `tek face enrol` needs a handful of distinct
# captures, and one every 8s would make enrolling take a minute and a half.
CROP_EVERY = 1.5
FACE_CROP = os.path.join(os.path.expanduser("~/.cache/tekdromo"), "crop.png")

# The waveform panel mixes what the speaker plays with what the mic hears, and
# the mic is far the quieter of the two: a person across the room peaks around
# 0.1 where the sink monitor peaks near 0.5. The panel auto-scales to its own
# recent maximum, so without a lift the mic trace is a flat line whenever
# anything has played recently.
SCOPE_MIC_GAIN = 4.0
# Noise gate for the microphone feed, subtracted before the gain.
#
# Without it the panel is never quiet. The trace auto-scales to its own recent
# maximum, so if the only thing arriving is room hiss, room hiss is what fills
# the panel - a permanently jittering full-height trace, which is a worse lie
# than the flat line it replaced. Measured on the C922 in this room: ambient
# peaks sit at a median of 0.0029 and a 90th percentile of 0.0042, while the
# speaker playing measures 0.1218 - twenty-nine times higher. A gate just above
# ambient costs real sound nothing.
#
# It belongs here rather than in Scope: a noise floor is a property of a
# microphone, not of a display, and Scope must keep raising the gain on genuinely
# quiet music.
SCOPE_MIC_FLOOR = 0.006
# How long the answer to "which input actually works" is reused. Finding it
# costs a short recording per candidate, and the watchdog asks every 2s.
SCOPE_MIC_TTL = 30.0

# Process start, captured at import so the startup report measures what the
# user actually waits for - from exec to a picture - not from some later point.
T_START = time.time()

CACHE_DIR = os.path.expanduser("~/.cache/tekdromo")
SRC = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
       for f in ("anatomy.py", "field.py", "contour.py", "rig.py")]


# ---------------------------------------------------------------------------
def _src_hash():
    h = hashlib.sha256()
    for p in SRC:
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


# Device discovery lives in camera.video_devices(): the tracker has to
# re-discover on every reopen anyway, so a second copy here could only drift
# out of step with the one that matters.


def cache_warm():
    """Is the geometry already on disk? Decides whether a wait is worth
    announcing: warm is ~1.3s to a picture, cold is ~5s."""
    return os.path.exists(os.path.join(CACHE_DIR, _src_hash() + ".npz"))


def load_geometry():
    """Disk-cached static geometry; rebuilds itself when the source changes."""
    path = os.path.join(CACHE_DIR, _src_hash() + ".npz")
    if os.path.exists(path):
        try:
            d = np.load(path)
            return d["v"], d["e"], d["n"], True
        except Exception:
            pass                                  # corrupt cache: rebuild
    # lips=True: the RESTING face must be complete. Regions only take over an
    # area once their controls actually move.
    v, e, n = contour.build(lips=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp.npz"                       # savez appends .npz
    np.savez_compressed(tmp, v=v, e=e, n=n)
    os.replace(tmp, path)                         # atomic
    return v, e, n, False


# ---------------------------------------------------------------------------
class Display:
    """Owns the panel, the model and the frame loop."""

    def __init__(self, args):
        self.a = args
        self.running = True
        self.last_frame = None
        self.last_frame_t = 0.0
        self.fd, self.mm, self.screen, self.w, self.h = framebuffer.open_screen()
        framebuffer.unblank(self.fd)
        self._unblanked = time.time()
        self.statics = phosphor.build_statics(self.w, self.h)
        # Only announce a wait long enough to be worth announcing. A warm start
        # is 1.3s; painting a banner over it would REPLACE the picture the
        # previous process left on the panel with a splash screen, turning an
        # invisible restart into a visible one. A cold start is ~5s, where a
        # blank panel would just look broken.
        if not cache_warm():
            self.banner("BUILDING GEOMETRY")

    # -- setup ------------------------------------------------------------
    def banner(self, msg):
        self.screen[:] = phosphor.erase_bgra(self.w, self.h, 0.25, self.statics)
        img = np.zeros((self.h, self.w, 3), np.uint8)
        cv2.putText(img, "TEKTRONIX 4014", (int(self.w * .10), int(self.h * .44)),
                    cv2.FONT_HERSHEY_SIMPLEX, .62, (40, 150, 70), 2, cv2.LINE_AA)
        cv2.putText(img, msg, (int(self.w * .10), int(self.h * .52)),
                    cv2.FONT_HERSHEY_SIMPLEX, .85, (70, 220, 110), 2, cv2.LINE_AA)
        self.screen[..., :3] = img
        self.screen[..., 3] = 255

    def load(self):
        v, e, n, warm = load_geometry()
        print("geometry %s (%d edges)" % ("cached" if warm else "built", len(e)),
              flush=True)
        # Hand the rig the cached geometry, unpunched - it drops edges per
        # frame and only for regions whose controls have moved.
        self.face = rig.Face(static=(v, e, n))
        self.face.express("neutral", blend=0.01)
        self.warm_done = False
        # Both of these are built on a background thread once the picture is
        # up - see _background_init. Neither is needed to draw a face, and
        # together they were 3.5s of the startup wait.
        self.stars = None

        self.cam = self.follow = None
        # Subscribes to the voice service for mouth frames. Retries forever in
        # its own thread, so the voice service can start later, be restarted,
        # or never exist, without the display noticing.
        self.mouth = voice_link.MouthLink().start()
        self.clock = None if self.a.no_clock else hud.Clock(self.w, self.h)
        self.scope = None if self.a.no_scope else hud.Scope(self.w, self.h)
        self.face_panel = None if self.a.no_facepanel else hud.FacePanel(self.w, self.h)
        # Camera-event state. Transitions are debounced here; the policy of
        # whether to act on them lives in the voice service.
        self._watch_state = False
        self._pending_state = False
        self._pending_since = None
        self._event_q = []
        return self

    # -- watchdog ---------------------------------------------------------
    def _watchdog(self, period=2.0):
        while self.running:
            time.sleep(period)
            if self.last_frame is not None and \
                    time.time() - self.last_frame_t > period:
                try:
                    self.screen[:] = self.last_frame
                except Exception:
                    pass

    # -- per-frame --------------------------------------------------------
    def pose(self, t, dt):
        """Head orientation: idle sway, overridden by whoever is being looked at."""
        idle = self.a.yaw * (0.72 * math.sin(2 * math.pi * t / 9.3)
                             + 0.28 * math.sin(2 * math.pi * t / 3.7))
        rx = -0.045 + 0.030 * math.sin(2 * math.pi * t / 6.1)
        if self.cam is None:
            return rx, idle
        st = self.cam.state()
        gx, gy, hy = self.follow.update(st, dt)
        self.face.set(gaze_x=gx, gaze_y=gy)
        # Presence drives the expression only when speech is not. A mood set
        # for a reply has to outrank "somebody is in the room", or walking into
        # frame mid-sentence would wipe the face back to attentive.
        if st["present"] != getattr(self, "_seen", False):
            self._seen = st["present"]
            if getattr(self, "_mood", None) is None:
                self.face.express("attentive" if st["present"] else "neutral",
                                  blend=0.45)
        w = 1.0 if st["present"] else 0.0         # presence weight
        self._watch(st, t)
        return rx - 0.10 * gy * w, idle * (1.0 - 0.75 * w) + hy * w

    MOOD_BLEND = MOOD_BLEND
    MOOD_RELAX = MOOD_RELAX

    def _apply_mood(self):
        """Wear the expression the reply asked for, while it is being said.

        The rig has had `amused`, `concerned`, `confused`, `happy` and
        `surprised` since it was written, and nothing ever set them - the face
        said something wry with a completely neutral expression. The controls,
        the presets and the blend were all already here; the only missing piece
        was anything deciding which one to wear.

        Edge-triggered, not per-frame: `express()` restarts its blend from the
        current pose each time it is called, so calling it every frame would
        freeze the ramp at its first step and the change would never arrive.

        The ramp is SLOW on purpose. A face that snaps between expressions
        reads as a slideshow of faces rather than as one face changing its
        mind, and the whole point of a 0.27ms warm rig is that it can afford to
        move gradually.
        """
        mood = self.mouth.mood if self.mouth.speaking else None
        if mood == getattr(self, "_mood", None):
            return
        self._mood = mood
        if mood:
            self.face.express(mood, blend=self.MOOD_BLEND)
        else:
            # Speech ended. Hand the face back to the presence state machine,
            # which _watch's edge trigger will not do on its own because
            # presence has not changed.
            self.face.express("attentive" if getattr(self, "_seen", False)
                              else "neutral", blend=self.MOOD_RELAX)

    # -- camera events -----------------------------------------------------
    def _watch(self, st, t):
        """Notice arrivals and departures and hand them to the voice service.

        Runs inside the frame loop, so it does nothing but compare two numbers
        and drop a note on a queue - the snapshot and the socket write happen
        on _event_worker. Anything that can block does not belong here.

        Only debouncing lives here: an arrival has to persist before it counts,
        or a detector glitch becomes an event. The COOLDOWN and the on/off
        switch live in the voice service instead, so `tek watch off` works
        without restarting the display.
        """
        present = bool(st["present"])
        if present != self._pending_state:
            self._pending_state = present
            self._pending_since = t
            return
        if self._pending_since is None or present == self._watch_state:
            return
        if t - self._pending_since < WATCH_STABLE:
            return                       # not settled yet
        self._watch_state = present
        self._pending_since = None
        self._event_q.append({
            "kind": "arrival" if present else "departure",
            "what": ("someone came into view" if present
                     else "whoever was there has gone"),
            "faces": 1 if present else 0,
        })

    def _feed_reader(self, feed):
        """Keep feed.level current from one PulseAudio source. Never returns.

        Reconnecting is driven from OUTSIDE this loop, by _feed_watchdog, which
        closes the source to break the read. That indirection is the whole
        point: the obvious version checks "has the default moved?" between
        reads, and a source that delivers nothing never gets between two reads
        - it sits blocked in the first one forever. Which is exactly what
        happened: at startup the default capture source is the Tegra onboard
        input, which has nothing plugged into it, so the reader bound to a
        device that produces no samples and could never notice that the real
        microphone had since become the default.
        """
        from .voice import io as vio
        while self.running:
            try:
                want = feed["resolve"]()
                if not want:
                    time.sleep(1.0)
                    continue
                src = vio.MicSource(device=want)
                feed["src"], feed["device"] = src, want
                feed["last_data"] = time.time()
                print("scope: following %s" % want, flush=True)
                gain, floor = feed.get("gain", 1.0), feed.get("floor", 0.0)
                for frame in src:
                    if not self.running:
                        break
                    raw = float(np.abs(frame).max()) / 32768.0
                    feed["level"] = max(0.0, raw - floor) * gain
                    feed["last_data"] = time.time()
            except Exception:
                pass
            finally:
                feed["level"], feed["src"] = 0.0, None
                try:
                    src.close()
                except Exception:
                    pass
            if self.running:
                time.sleep(1.0)

    def _feed_watchdog(self, feeds, check=2.0, stale=4.0):
        """Force a feed to reconnect when it is on the wrong device or mute.

        Two independent faults, because either alone leaves a flat line:

        * **Wrong device.** The feeder used to open "@DEFAULT_MONITOR@".
          PulseAudio resolves that magic name once, when the stream is created,
          and never moves an existing stream. The display starts before
          tek-bluetooth has connected the speaker, so the default sink at that
          moment is the analog output - and when the speaker connected and
          became default, every sound went somewhere the scope was not looking.
          It showed a flat line while music played, on every boot. parec stayed
          perfectly healthy throughout, which is why retrying on failure never
          fired: there was no failure, just a stream aimed at the wrong place.

        * **Silent device.** A source can be the right answer and still deliver
          nothing - a dead input, or one that went away without closing the
          stream. Waiting for a read that never comes looks identical to a
          quiet room.

        Closing the source is what breaks the reader's blocking read: parec
        dies, the pipe closes, read() returns None and the reader rebuilds.
        """
        while self.running:
            time.sleep(check)
            now = time.time()
            for feed in feeds:
                src = feed.get("src")
                if src is None:
                    continue
                why = None
                try:
                    if feed["resolve"]() != feed["device"]:
                        why = "default moved off %s" % feed["device"]
                except Exception:
                    pass
                if why is None and now - feed.get("last_data", now) > stale:
                    why = "%s delivered nothing for %.0fs" % (feed["device"],
                                                              stale)
                if why:
                    print("scope: %s - reconnecting" % why, flush=True)
                    feed["level"] = 0.0
                    try:
                        src.close()
                    except Exception:
                        pass

    def _scope_feeder(self):
        """Drive the waveform panel from everything audible.

        Two feeds - what the speaker is playing, and what the microphone hears
        - and ONE pusher on a steady clock taking the louder of the two. A
        single time base matters: if each source scrolled the trace itself the
        horizontal axis would mean different things depending on which was
        active, and two sources at different sample rates would fight.

        The microphone is included because "nothing shows when there is audio
        out or in" is the complaint this fixes, and because with an ear on the
        box, a panel that moves when someone talks to it is the whole point.
        """
        from .voice import io as vio, pcm as vpcm
        # The monitor resolves cheaply by name. The microphone must be PROBED:
        # the mic is built into the webcam, so a camera replug moves the
        # PulseAudio default to the Tegra onboard input - which has nothing
        # plugged into it - and it never moves back. Trusting the default put
        # two recorders on a dead device while the real mic sat idle.
        feeds = [{"resolve": vio.default_monitor, "gain": 1.0, "floor": 0.0,
                  "level": 0.0, "src": None, "device": None, "last_data": 0.0},
                 {"resolve": lambda: vio.working_source(ttl=SCOPE_MIC_TTL),
                  "gain": SCOPE_MIC_GAIN, "floor": SCOPE_MIC_FLOOR,
                  "level": 0.0, "src": None, "device": None, "last_data": 0.0}]
        for feed in feeds:
            t = threading.Thread(target=self._feed_reader, args=(feed,))
            t.daemon = True
            t.start()
        w = threading.Thread(target=self._feed_watchdog, args=(feeds,))
        w.daemon = True
        w.start()

        step = vpcm.FRAME_MS / 1000.0
        nxt = time.time()
        while self.running:
            nxt += step
            delay = nxt - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.time()             # fell behind; do not spiral
            if self.scope is not None:
                self.scope.push_level(max(f["level"] for f in feeds))

    def _event_worker(self):
        """Snapshot and send. Off the render loop entirely.

        Also keeps a recent frame on disk whenever anyone is in view, so
        `tek look` has something to look at. The voice service does not own the
        camera - only this process can - and a manual "look now" that finds a
        stale or missing image is worse than useless.
        """
        last_shot = 0.0
        last_crop = 0.0
        while self.running:
            if not self._event_q:
                now = time.time()
                # Unconditionally, not only when a face is present: `tek
                # look` should work in an empty room too.
                if self.cam is not None and now - last_shot > SNAPSHOT_EVERY:
                    self.cam.snapshot(SNAPSHOT)
                    last_shot = now
                if self.cam is not None and now - last_crop > CROP_EVERY:
                    # The display owns the camera, so enrolment has no other
                    # way to get a face crop.
                    #
                    # aligned_crop(), NOT crop(): this file is the enrolment
                    # gallery's only source, and an unaligned sample is worse
                    # than a missing one. crop() falls back to the raw detector
                    # rectangle when landmarks fail, which is right for the HUD
                    # and silently poisonous here - a gallery built from
                    # fallbacks reports as a successful enrolment and then
                    # fails to recognise anybody.
                    #
                    # Writing nothing when alignment fails is what makes the
                    # file's mtime meaningful: enrolment waits for a NEW crop,
                    # so "no fresh crop" now means "that pose could not be
                    # captured" and it can say so instead of storing rubbish.
                    try:
                        crop = self.cam.aligned_crop()
                        if crop is not None and crop.size:
                            cv2.imwrite(FACE_CROP, crop)
                    except Exception:
                        pass
                    last_crop = now
                time.sleep(0.25)
                continue
            ev = self._event_q.pop(0)
            self._event_q[:] = []            # only the newest matters
            try:
                if self.cam is not None and ev["kind"] == "arrival":
                    shot = self.cam.snapshot(SNAPSHOT)
                    if shot:
                        ev["image"] = shot
                # Who the recogniser settled on, for the journal and for the
                # per-person half of recall. Read here rather than in _watch
                # because _watch runs on the frame loop and this is a voted
                # label that is still settling at the instant of arrival - by
                # the time the snapshot is taken it has had its 4-second window.
                if self.cam is not None:
                    who = getattr(self.cam, "name", None)
                    if who and who != "UNKNOWN":
                        ev["person"] = who
                ev["when"] = time.strftime("%A %H:%M")
                c = bus.Client(bus.DEFAULT_PATH, timeout=10)
                c.send({"cmd": "event", "event": ev})
                c.close()
            except Exception:
                pass                         # voice service down: not our problem

    def _wait_for_camera(self, period=2.0):
        """Attach a tracker, and keep one alive for as long as we run.

        This began as a single os.path.exists check during load(), which is
        wrong in the case that matters: at boot, USB enumeration has not
        finished when systemd starts us, so /dev/video0 does not exist yet and
        the head would never track again until someone restarted the service.

        It then became a loop that RETURNED at the first successful attach, on
        the stated grounds that "Tracker's own loop handles unplug/replug for
        good". That was an assertion, not a measurement, and it was wrong: the
        tracker reopened using the device index it was built with, and its
        worker was wrapped in a bare `except Exception: pass`, so a camera
        swapped for a different one left the head permanently still.

        So this is now a supervisor and never returns. Tracker survives an
        unplug by itself (see camera._loop), and if it somehow does not, this
        notices a tracker that has stopped delivering frames and replaces it.
        Two layers, because the first one has already been believed once
        without being true.
        """
        from . import camera
        while self.running:
            try:
                cam = self.cam
                if cam is not None and not (cam._t is not None
                                            and cam._t.is_alive()):
                    # The worker died despite everything. Say so ONCE - the
                    # previous failure was invisible precisely because nothing
                    # did - then drop it, so a camera that stays unplugged does
                    # not reprint this every couple of seconds forever.
                    print("camera: tracker thread is gone, rebuilding",
                          flush=True)
                    cam.stop()
                    self.cam = cam = None
                if cam is not None or not camera.video_devices():
                    time.sleep(period)
                    continue
                follow = camera.Follower()
                fresh = camera.Tracker().start()
                # follow first: pose() reads self.cam as the guard, so the
                # follower must already exist when cam becomes non-None.
                self.follow, self.cam = follow, fresh
                print("[%5.2fs] camera attached" % (time.time() - T_START),
                      flush=True)
            except Exception:
                traceback.print_exc()
            time.sleep(period)

    def _background_init(self):
        """Everything that is not needed to draw the FIRST frame.

        Startup used to be 9.5s to a picture. Two of those seconds were the rig
        rebuilding geometry the caller already had; the remaining 3.5s is this -
        pose warming (~3.0s) and the star field (~0.4s). Neither changes what
        the first frame looks like, so both happen behind the running picture.

        Order matters: stars first, because 0.4s later the backdrop is simply
        there, whereas warming has to finish entirely before it helps anything.

        Speech is held back until warm_done, so no un-warmed mouth pose can
        hitch mid-word. A cold pose costs ~53ms - three frames - which is
        exactly the stutter this whole cache exists to prevent.
        """
        try:
            if not self.a.no_stars:
                self.stars = starfield.Backdrop(self.w, self.h)
                print("[%5.2fs] starfield ready" % (time.time() - T_START),
                      flush=True)
            self.face.warm(verbose=False)
            print("[%5.2fs] poses warm (%d) - speech enabled"
                  % (time.time() - T_START,
                     sum(len(r.cache.d) for r in self.face.regions.values())),
                  flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            self.warm_done = True

    def run(self):
        threading.Thread(target=self._watchdog, daemon=True).start()
        threading.Thread(target=self._background_init, daemon=True).start()
        threading.Thread(target=self._event_worker, daemon=True).start()
        if self.scope is not None:
            threading.Thread(target=self._scope_feeder, daemon=True).start()
        if not self.a.no_camera:
            threading.Thread(target=self._wait_for_camera, daemon=True).start()
        t0 = prev = tick = time.time()
        frames = errors = 0
        first = True
        while self.running:
            try:
                now = time.time()
                dt = min(0.1, now - prev)
                prev, t = now, now - t0

                # The mouth, in priority order: real speech if the voice
                # service is saying something, synthetic babble only if it was
                # asked for, otherwise shut. Held closed until the poses are
                # warm - a cold mouth pose costs ~53ms, three frames.
                if self.warm_done:
                    openness, rounding = self.mouth.mouth()
                    if openness > 0.0 or self.mouth.speaking:
                        self.face.speak(openness, rounding)
                    elif self.a.babble:
                        self.face.speak(*speech.synthetic(t))
                    else:
                        self.face.speak(0.0, 0.0)
                    self._apply_mood()
                rx, ry = self.pose(t, dt)
                v, e, n = self.face.update(t, dt)
                pts = geometry.build_pts_culled(v, e, n, self.w, self.h,
                                                (rx, ry, 0.0), self.a.dist,
                                                -0.05, "and", self.a.fov)
                # HUD panels emit the same (N,2,2) segments the head does, so
                # they are simply concatenated and go through ONE render pass.
                # Drawing them separately would mean a second bloom and a
                # second LUT, and two looks that drift apart.
                panels, faint = [], []
                if self.clock is not None:
                    panels.append(self.clock.points())
                    # Unlit seven-segment strokes go to the renderer's dim
                    # layer, not the bright one.
                    faint.append(self.clock.dim_points())
                if self.scope is not None:
                    panels.append(self.scope.points())
                if self.face_panel is not None:
                    # None when nobody is there, which the panel draws as a
                    # "no signal" cross rather than an empty box.
                    self.face_panel.update(
                        self.cam.face_points() if self.cam is not None else None,
                        self.cam.who() if self.cam is not None else None)
                    panels.append(self.face_panel.points())
                if panels:
                    pts = np.concatenate([pts] + panels) if len(pts) \
                        else np.concatenate(panels)
                dim = None
                if faint:
                    dim = np.concatenate(faint)
                    if not len(dim):
                        dim = None
                frame = phosphor.render_bgra(pts, self.w, self.h, self.statics,
                                             dim=dim,
                                             dim_level=hud.Clock.GHOST_LEVEL)
                if self.stars is not None:
                    frame = self.stars.under(frame, t)
                self.screen[:] = frame
                self.last_frame, self.last_frame_t = frame, time.time()
                frames += 1
                if first:
                    # The number that matters at boot: exec -> a picture.
                    # Logged every start so a regression shows up in the
                    # journal instead of only under a benchmark.
                    print("[%5.2fs] FIRST FRAME" % (time.time() - T_START),
                          flush=True)
                    first = False
                # Cap the rate. Uncapped we render as fast as the CPU allows,
                # which on a 2GB Nano at MAXN with a USB hub, wifi, bluetooth
                # and a camera attached pushed the input rail hard enough to
                # trip soctherm OC ALARM 0x00000001. Nothing above ~30fps is
                # visible on this panel, so the extra draw buys nothing.
                if self.a.max_fps > 0:
                    slack = (1.0 / self.a.max_fps) - (time.time() - now)
                    if slack > 0:
                        time.sleep(slack)
            except Exception:
                # One bad frame is not fatal - but silently swallowing a
                # PERSISTENT failure is worse. A stale import once had this
                # loop crash-looping at 0 fps with 198 errors and nothing
                # useful in the journal, because only the first 3 tracebacks
                # were printed. Re-report periodically so a stuck loop is
                # always visible.
                errors += 1
                if errors <= 3 or errors % 200 == 0:
                    traceback.print_exc()
                    print("frame error #%d - still failing" % errors, flush=True)
                time.sleep(0.05)

            now = time.time()
            if now - tick >= 10.0:
                print("%.1f fps  errors=%d" % (frames / (now - tick), errors),
                      flush=True)
                frames, tick = 0, now
            if now - self._unblanked > 30.0:
                framebuffer.unblank(self.fd)
                self._unblanked = now

    def close(self):
        """Shut down WITHOUT clearing the panel.

        This used to zero the framebuffer on exit, which meant every restart
        showed black for the whole gap - process teardown, RestartSec, and the
        new process's startup - even though the hardware was perfectly capable
        of just holding the last picture. Leaving the image up makes a restart
        very nearly invisible, and it is what a storage tube does anyway: the
        image persists on the phosphor until something erases it.

        Use --clear-on-exit when running by hand and you want your terminal
        back.
        """
        self.running = False
        self.mouth.stop()
        if self.cam is not None:
            self.cam.stop()
        try:
            if self.a.clear_on_exit:
                self.screen[:] = 0
        finally:
            del self.screen
            self.mm.close()
            os.close(self.fd)
        print("exited cleanly", flush=True)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="tekdromo")
    ap.add_argument("--yaw", type=float, default=0.30)
    ap.add_argument("--dist", type=float, default=16.0)
    ap.add_argument("--fov", type=float, default=11.4)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-stars", action="store_true")
    ap.add_argument("--no-clock", action="store_true",
                    help="hide the clock/date panel")
    ap.add_argument("--no-scope", action="store_true",
                    help="hide the audio waveform panel")
    ap.add_argument("--no-facepanel", action="store_true",
                    help="hide the camera landmark-face panel")
    ap.add_argument("--max-fps", type=float, default=30.0,
                    help="frame cap. Rendering flat-out burns power for no "
                         "visible benefit and this board has thin current "
                         "headroom (see the OC ALARM note in app.run).")
    ap.add_argument("--clear-on-exit", action="store_true",
                    help="blank the panel on exit. Off by default so a service "
                         "restart holds the last picture instead of going "
                         "black - see Display.close.")
    ap.add_argument("--babble", action="store_true",
                    help="move the mouth with synthetic syllables when nothing "
                         "is actually being said. Off by default now that real "
                         "speech drives it - babbling while silent reads as a "
                         "malfunction rather than as life.")
    a = ap.parse_args(argv)

    import signal
    d = Display(a).load()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: setattr(d, "running", False))
    try:
        d.run()
    finally:
        d.close()
