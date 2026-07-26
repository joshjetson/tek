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

## 5. The runner (`tekrun.py`) — the display must never stop

Five failure modes, each handled:

1. **Startup black gap** — geometry disk-cached, keyed by a hash of the source
   files so it self-invalidates. Cold 4.2s → **warm 0.019s**. A boot screen is
   painted before any of that.
2. **Exception killed the process** — per-frame errors are caught, last good
   frame stays up, loop continues.
3. **systemd start limit** — `Restart=always` still gave up after 5 failures.
   Now disabled (in `[Unit]`). Verified: **7 consecutive SIGKILLs, all
   recovered**.
4. **Console blanker** — `FBIOBLANK` every 30s.
5. **A wedged model** — watchdog thread repaints the last good frame if the
   main loop misses its deadline.

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
