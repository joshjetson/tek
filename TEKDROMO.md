# TEKDROMO

A Tektronix 4014 storage-tube vector display, running on a Jetson Nano 2GB,
rendering an animated human face that talks and (as of now) watches the room.

Long-term intent: an always-on household presence — the face is the UI. Not a
smart-speaker puck; a thing with a location, an aesthetic, and continuity.

---

## 1. Platform, and what it forces

| | |
|---|---|
| Board | NVIDIA Jetson Nano **2GB** (t210), 4× Cortex-A57 @1.48GHz, 128-core Maxwell (sm_53) |
| OS | Ubuntu 18.04.6, L4T **R32.5.2**, kernel 4.9.201-tegra |
| Python | **3.6.9** — no walrus `:=`, no f-string `=`, no dataclasses by default |
| glibc | 2.27 — most modern prebuilt binaries will not run |
| CUDA | 10.2, cuDNN 8.0, TensorRT 7.1.3 |
| OpenCV | **4.5.5 built from source with CUDA**, installed to `/usr/local` |
| Storage | SD card: **22.6 MB/s sequential, ~635 IOPS**. This is the real bottleneck |
| Display | HDMI 1024×600 (panel native), `/dev/fb0`, BGRA, no X |
| Free RAM | ~900MB–1.1GB with the display running |

### Environment traps — read these before debugging anything

1. **`OPENBLAS_CORETYPE=ARMV8` is mandatory.** Without it `import numpy` and
   `import cv2` die with **SIGILL**. numpy 1.19.5's OpenBLAS misdetects the
   A57. It is set in `/etc/environment` (login only) *and* explicitly in the
   systemd unit. Any new service or cron job must set it too.
2. **`cv2.data` does not exist** in a source build. Cascades are at
   `/usr/local/share/opencv4/haarcascades/`.
3. **systemd is 237.** `StandardOutput=append:` is unsupported (silently
   ignored). `StartLimitIntervalSec`/`StartLimitBurst` must be in **`[Unit]`**,
   not `[Service]` — putting them in `[Service]` is silently ignored.
4. **No desktop.** X/lightdm is disabled, default target is `multi-user`.
   `tek-display` owns `/dev/fb0` continuously, so tty1 is painted over. SSH is
   unaffected. `sudo systemctl stop tek-display` to get the console back.
5. **Console blanking** (`consoleblank`) will switch the panel off after 10
   min of no keypress. Disabled via kernel cmdline *and* re-asserted in-process
   with `FBIOBLANK` every 30s.
6. **L4T is pinned at 32.5.2.** 32.7.6 exists and is the last release for t210,
   deliberately not taken — it rewrites the bootloader, and there is no backup
   of the 93GB on the card. Do it at SSD-migration time, not before.

---

## 2. Rendering

### The aesthetic is a real constraint, not decoration

A **Direct-View Storage Tube** is genuinely vector: an electron beam walks
point-to-point and the phosphor holds the charge. Consequences:

* **No fills, no shading, ever.** Only line art.
* **No scanlines** — it does not scan. Adding them is the classic fake tell.
* **Constant intensity** — stored charge is binary-ish, so no depth-cued
  brightness by default.
* Erase is all-or-nothing: a bright green flash, then start over.

### Pipeline (`tekvector.py`, `tekfb.py`)

Geometry → project → back-face cull → `cv2.polylines` → multi-scale bloom →
phosphor LUT → straight into an mmap'd `/dev/fb0`.

Hard-won performance facts:

| lesson | detail |
|---|---|
| Batch line drawing | 814 `cv2.line` calls = 50ms; one `polylines` = **2.9ms** |
| Composite via LUT | 8 full-frame float passes = 140ms; single-channel + 256-entry LUT = **13ms** |
| **GPU is not always faster** | bloom at 512×300: CUDA **29ms**, CPU pyramid **6.8ms**. Kernel-launch overhead dominates at small sizes; the GPU won 4.2x at 1080p. **Measure, don't assume** |
| Framebuffer beats X | `imshow` pushed 2.7MB/frame through the X socket; the panel is BGRA with no padding, identical to OpenCV's layout, so frames memcpy in with zero conversion |

Overall: **4.6 → 45 fps** across those changes.

---

## 3. The two faces

### `face_1` — measured reconstruction (`faces/face_1/`, `tekanat.py`)

Dense lat/long ovoid + ~14 anatomical displacements + a tangential warp field +
explicit feature curves. Built by **measuring a reference image** against a
normalised grid (unit = crown-to-chin height, origin = chin on the midline).

Final: silhouette ratio **0.996**, landmark mean error **3.8%** over 11
landmarks. ~6,000 edges, 29fps with a talking mouth.

**Why it was abandoned as the base:** feature curves drawn on an undeformed
mesh always read as a decal. Four separate attempts at a convincing nose ridge
all failed for this reason. Kept because the measurement work is sound and the
talking-mouth code is a useful reference.

### `face_2` — FDL contour head (`tekfdl.py`) ← **THIS IS BASE FACE v1**

An **implicit height field sliced into iso-contours**:

```
z(x,y) = skull + forehead + brow + nose + cheeks + lips + chin
         − eyes − philtrum − nostrils
for z = .95 down to −.25 step −.05:  extract contours → emit vectors
```

This is the key architectural win. **Contours are level sets of the real
surface, so they flow around every feature for free.** Everything face_1 tried
to fake with overlays happens automatically here.

Anatomy is from research, not guesswork:
* **Loomis**: a head is a sphere with the sides cut *flat* — an ellipse cannot
  be a head. The silhouette is an explicit landmark profile with flat side
  planes.
* Widest point is the **zygomatic arch** (below the eye line), not the cranium.
* **Five eyes wide**; mouth two eyes wide; vertical thirds.
* **Gonial angle** — the jaw is a slope *break*, not a smooth taper.
* Ears span **brow line → nose base**, tilted 18°.

**Everything is one renderer.** Neck and ears were originally ring-and-meridian
meshes and read as bolted-on; both are now fields:
* Neck is unioned into the main field with `max()`, which also gives correct
  jaw-over-neck occlusion for free.
* Ears have their **own lateral field** — `x` as a function of `(y,z)` —
  because a `z(x,y)` height field cannot express something that protrudes
  sideways. Their contours are level sets of sideways protrusion.

---

## 4. The expression rig (`tekrig.py`)

```
EXPRESSIONS   named presets → control values
     ↓
CONTROLS      10 named scalars — the ONLY animation state
     ↓
REGIONS       a bbox + a field function + which controls touch it
     ↓
FIELD/CACHE   re-contour ONE region, memoised by quantised controls
```

Why it works: the face is a field, so an expression is just different numbers
in the equation — no blendshapes, no skinning. A full rebuild is ~4s, but an
expression only disturbs a small box, and **the field outside the box is
unchanged so contours still meet the border exactly**.

**The DRY line:** every region shares one contour/cache/compose pipeline. Only
the field maths differs. Adding a feature = one field function + one `REGIONS`
entry; caching, border-splitting and compositing are inherited.

Warm cost: **0.27 ms/frame**.

* Add an expression → one line in `EXPRESSIONS`
* Add a control → one line in `CONTROLS` + a term in a region's field function
* Blink is **not** an expression — it is a reflex on its own timer that clamps
  `eye_open`, so it works during any expression and during speech.

API: `face.express(name)`, `face.speak(openness, rounding)`,
`face.set(**controls)`, `face.update(t) -> (verts, edges, normals)`.

---

## 5. The runner (`tekdromo/app.py`) — the display must never stop

Seven failure modes, each handled:

1. **Startup black gap** — geometry disk-cached, keyed by a hash of the source
   files so it self-invalidates. Cold 4.2s → **warm 0.019s**.
2. **Exception killed the process** — per-frame errors are caught, last good
   frame stays up, loop continues. A *persistent* failure is re-reported
   periodically; only printing the first three tracebacks once hid a loop
   spinning at 0 fps with 198 errors.
3. **systemd start limit** — `Restart=always` still gave up after 5 failures.
   Now disabled (in `[Unit]`, which is where systemd 237 actually reads it).
   Verified: **7 consecutive SIGKILLs, all recovered**.
4. **Console blanker** — `FBIOBLANK` every 30s.
5. **A wedged model** — watchdog thread repaints the last good frame if the
   main loop misses its deadline.
6. **Blanking the panel on exit** — `close()` used to zero the framebuffer, so
   every restart was black for the whole gap. It now leaves the last picture
   up; the hardware holds it with no process running at all (verified: service
   stopped, panel mean 80.8, 49.6% of pixels lit). Which is what a storage tube
   does anyway. `--clear-on-exit` restores the old behaviour for hand runs.
7. **Camera never re-attaching** — `load()` checked `/dev/video0` exactly once.
   At boot, USB enumeration has not finished when systemd starts us, so the
   head would never track again until someone restarted the service. Now a
   thread waits for the device to appear. Invisible on a running machine —
   only a real reboot or a replug would have shown it, so `tests/boot_camera.py`
   fakes the device instead.

   **The fix was half a fix, and said so in a docstring.** `_wait_for_camera`
   returned at the first successful attach, on the written grounds that "once
   started, Tracker's own loop handles unplug/replug for good". It did not.
   Swapping the camera for a new one stopped face tracking permanently, and
   there were three reasons, of which the device index was the least
   interesting:

   * `Tracker._loop` wrapped its entire body in `except Exception: pass`, so
     one exception — including one raised while reopening a device that had
     just vanished — ended the thread silently and forever;
   * the reopen used `self.device`, the index captured at construction;
   * `VideoCapture.isOpened()` was treated as proof a device works, when some
     nodes open and never deliver a frame.

   Now: the tracker re-discovers on every open and accepts a node only once it
   has actually grabbed a frame; nothing but `stop()` can end the worker; and
   `_wait_for_camera` is a permanent supervisor that replaces a dead tracker.
   Two layers, because the single-layer version was believed once already
   without being true.

   Verified by really unplugging it — `tools/camera_replug.py` deauthorizes the
   device on the USB bus. With `--hold` the old node is kept open so the camera
   is forced onto a different index, reproducing the original failure exactly.
   Recovery measured at 4.9 s, and the log shows the path:
   `no frames for 3s on /dev/video0 - re-discovering` →
   `/dev/video1 delivering (open #3)`.

8. **No way out** — the one that actually bit a human. Every mode above is
   about keeping the display *alive*; none of them asked what happens when you
   need it to stop. The display writes to `/dev/fb0` without owning a VT, so
   `Ctrl+Alt+F2` repaints the console and the display covers it again on the
   next frame; failure mode 6 above means even stopping the service leaves the
   face on the panel; and the wifi was a user-scoped NetworkManager profile, so
   the box had no network until someone logged in. Those three are each
   defensible alone and together they made the machine unusable: the only way
   back in was blind-typing a login roughly a hundred times.

   Fixed in two independent layers, because one of them will eventually be
   broken by something else:
   * the wifi profiles are now system-scoped with `psk-flags=0`, so **SSH works
     before anyone logs in** — this is the real fix;
   * `tek-panic.service` watches every input device for **ESC ×3** and stops
     the display, then forces fbcon to repaint by switching VT away and back.
     Measured: 0.22 s to stop, 0.44 s to repaint, 127,576 → 25,267 lit pixels.

   The general lesson is the one worth keeping: *an availability invariant with
   no escape hatch is a liveness bug wearing a hat.* "The display must never
   stop" was enforced so thoroughly — `Restart=always`, no start limit, holding
   the last frame across restarts — that it defeated its own operator.
   `tools/check_boot.sh` now checks both layers, since both fail only at boot
   and look perfect from an established session.

### Time to first frame — 9.47s → 1.24s

Measured end to end, from `exec` to a picture on the panel:

| | before | after |
|---|---|---|
| import package | 1.06s | 1.06s |
| geometry (cached) | 0.04s | 0.04s |
| `rig.Face()` | 4.47s | **0.14s** |
| starfield | 0.41s | *background* |
| pose warming | 3.07s | *background* |
| phosphor statics | 0.45s | 0.45s |
| **first frame** | **9.47s** | **1.24s** |

Three changes, in order of size:

* `rig.Face()` was calling `contour.build()` itself — rebuilding the exact
  geometry every caller already had on disk, then discarding it. It now takes
  `static=`.
* Pose warming (~3.0s) and the star field (~0.4s) moved behind the running
  picture. Neither changes what the first frame looks like. Speech is gated on
  `warm_done` so an un-warmed mouth pose can't hitch mid-word — a cold pose
  costs ~53ms, three frames, which is the exact stutter the cache exists to
  prevent.
* The startup banner is only painted when the geometry cache is *cold*. On a
  warm start it would replace the picture the previous process left on the
  panel with a splash screen — turning an invisible restart into a visible one.

Cold (first boot ever, or after a source change): 4.79s to first frame.

Every start logs these numbers to the journal, so a regression shows up in
`journalctl -u tek-display` rather than only under a benchmark.

### Verifying boot survival without rebooting

`tools/check_boot.sh` — unit validity, enablement and the `multi-user.target`
symlink, start-limit settings, path existence, `OPENBLAS_CORETYPE` in the *unit*
(not just the login shell), `video` group membership, then a genuine cold start
with every cache cleared and a warm start after it.

Two traps it hit, both of which produced a confident wrong answer first:
`sudo` with no tty silently did nothing, so the "cold start" measured a service
that had never stopped; and `journalctl --since` has one-second granularity, so
a window opened right after a restart still contains the *outgoing* process's
lines. It now reports by `_PID`.

---

## 5b. Voice — ears and a mouth

Built and tested with no microphone and no speaker attached, because at the
time there were none.

### The DRY spine

Every stage of a voice loop either consumes audio or produces it, so there is
**one** PCM contract — 16 kHz, mono, `int16`, 20 ms frames — and **one** pair of
abstractions: `Source` yields frames, `Sink` accepts them. Everything else is
expressed in those terms, which buys three things that are not just tidiness:

* A microphone and a WAV file are the same type, so `tests/voice_loopback.py`
  exercises the real pipeline with zero hardware.
* **The mouth is a Sink.** The audio going to the speaker and the audio driving
  the face are not two signals kept in agreement — they are one signal with two
  consumers. Lip-sync is a property of the topology.
* Wake word and transcription will be one engine with two grammars, not two
  models.

`pcm.RATE` is 16 kHz because that is native for Vosk *and* Whisper, so the
recognition path needs no conversion at all. Resampling happens only at
hardware edges.

### Piper on a stack that officially cannot run it

The official wrapper needs Python 3.9+; this box has 3.6.9 and glibc 2.27. Two
risks were predicted and both evaporated on measurement:

| Predicted | Actual |
|---|---|
| espeak-ng 1.49.2 (2018) too old — build 1.52 from source | **100% phoneme coverage**, 0 unmapped. No build needed. |
| ORT 1.10 (last cp36 aarch64 wheel) won't load a 2023 VITS export | Loads clean |

`piper-phonemize` has no Python 3.6 build and is the actual blocker — but it is
only a C++ shim over espeak-ng, which is already in apt. So the path is
`espeak-ng --ipa` → the model's own `phoneme_id_map` → onnxruntime, and the
dependency disappears. **0.72× real-time**: faster than speech.

espeak's `--ipa` drops punctuation and emits a newline instead, but the model
was *trained with* punctuation phonemes — they are its pause cues. Clauses are
phonemised separately and the punctuation put back, which is what gives it
sentence rhythm instead of a flat monotone.

A side benefit: `speech.from_envelope()` hardcodes `rounding=0` and says why —
"real viseme shape needs phoneme information, which an envelope does not
carry." The Piper path *has* the phonemes, so the mouth rounds on /u/, /o/, /w/.

### The mouth ran 300× too fast

`pacat`'s stdin is a pipe with no backpressure: it accepted **3.0 s of audio in
0.01 s**. Driving the mouth from the write loop therefore animated an entire
sentence in ten milliseconds and left the face still for the rest of it —
reported from across the room as *"it stopped before you stopped"*.

The mouth now runs on the wall clock, offset by the sink latency PulseAudio
reports (232 ms here, mostly A2DP). `tests/voice_lipsync.py` pins it: the mouth
stream must last as long as the audio, paced at ~20 ms.

### Bluetooth on JetPack

Three faults, each masking the next:

1. NVIDIA's drop-in starts `bluetoothd` with `--noplugin=audio,a2dp,avrcp` —
   A2DP disabled outright. The override must sort **after** `nv-` in filename
   order; systemd applies drop-ins lexicographically by filename and `/etc`
   does not automatically beat `/lib`.
2. The D-Bus policy denies uid 1000 access to `org.bluez`, so PulseAudio cannot
   register a media endpoint. Surfaces as `Protocol not available` from BlueZ
   and `No default controller available` from `bluetoothctl` — neither of which
   points at permissions. Group membership cannot fix a *running* PulseAudio.
3. `module-suspend-on-idle` suspends the sink, the speaker sees no stream and
   drops the link.

`tek-bluetooth.service` reasserts all three continuously. Verified by forcing a
disconnect: recovered in ~10 s.

---

## 6. File map

```
tekvector.py   geometry primitives, projection, storage-tube rasteriser
tekfb.py       framebuffer plumbing, phosphor LUT, bloom
tekhead.py     build_pts_culled() — projection + back-face culling
tekfdl.py      BASE FACE v1: the FDL field, contour generator, neck, ears
tekrig.py      expression rig (controls / regions / expressions / cache)
tekrun.py      the display service entry point
tekanat.py     face_1 (superseded, kept for reference)
faces/face_1/  snapshot + README with its measurements
faces/face_2/  snapshot of base v1
nano-tune-REVERT.md   every system change made, with revert steps
/etc/systemd/system/tek-display.service
```

Control: `systemctl {start,stop,restart} tek-display`, logs via
`journalctl -u tek-display -f`.

---

## 7. Working practices that mattered

These were learned the hard way in this project; ignoring them cost hours.

* **Measure, don't eyeball.** Every "looks about right" judgement here was
  wrong. Build the comparison harness (normalised grid, matched-scale crops)
  and quote numbers.
* **Verify the extraction before fitting to it.** A threshold-based eye
  measurement merged the eye with the nose-bridge shadow and put the eyes 37%
  too close together. Another apparent "jaw corner" in the reference silhouette
  turned out to be where the *ears* end — fitting it would have welded a false
  landmark into the model permanently.
* **Profile, don't guess.** Twice the assumed bottleneck was wrong; both times
  the real cost was a per-point Python loop, not the numerics.
* **Check that an edit landed.** A heredoc printed its success message
  unconditionally while its string replacements silently failed to match, and
  two "completed" changes had done nothing. `grep` for the symbol.
* **Look at the render before claiming anything.** A caption once described
  contours "converging at the bridge" when they visibly did not.

---

## 8. Where it is now, and what is next

**Working:** base face v1 live at ~41fps, talking, blinking, expression rig
wired, display survives kills and reboots.

**Just added:** USB camera — Adesso CyberTrack H4 (`0c45:636b`), UVC,
`/dev/video0`, 640×480 MJPG @ 30fps. Face detection confirmed working.
MJPG is required — YUYV at that resolution saturates USB 2.0.

**In progress:** camera → gaze. Detector output is already in the rig's
coordinate range (−1..+1), so it feeds `gaze_x`/`gaze_y` directly.
Haar detect costs **147ms**, far too slow per-frame; plan is a background
thread at ~5Hz on a subsampled frame with smoothing, and later TensorRT (the
GPU is idle and this is the workload it is actually good at).

**Known weak spots:** ear shape is generic (the *field* is right, the profile
is bland); expression amplitudes are too subtle and want a tuning pass;
FDL contours are organic where the reference is crisp quad topology.

**Deliberately not done:** local LLMs (sm_53 + CUDA 10.2 cannot build modern
llama.cpp kernels; it would fall back to 4 weak cores). The intended split is
**the Nano is the senses and the always-on body; the brain is an API call.**

---

## Licence

[Apache License 2.0](LICENSE) — see the
[README](README.md#licence) for why that one, and
[README §11](README.md#11-privacy-consent-and-what-this-is-not) for the
deployment responsibilities that come with pointing a camera and a microphone
at people.
