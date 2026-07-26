# Jetson Nano 2GB tuning — what changed, 2026-07-25

Every change is reversible. Backups of replaced files: `/root/nano-tune-backup/`

## Results

| | before | after |
|---|---|---|
| Processes | 268 | 237 |
| MemAvailable | 870 MB | ~1.15 GB |
| Disk used | 94 G (85%) | 93 G (84%), 1.17 GB freed |
| Failed units | 2 | 0 |
| I/O scheduler | cfq | noop |
| Boot (est.) | ~19 s | ~14 s (apt-daily's 33 s no longer blocks) |

---

## 1. Desktop session (user-level, no root)

Killed and disabled via `Hidden=true` overrides in `~/.config/autostart/`:

- **xscreensaver + the `distort` GL hack** — ~93 MB, and it kept the GPU/CPU
  busy permanently on an idle desktop. Replaced with DPMS blanking at 10 min.
- **nvpmodel_indicator** — 9 copies of one tray icon, ~136 MB combined.
- **compton** — GL compositor; pure GPU overhead on this hardware.
- **update-manager** (82 MB) / **update-notifier** / **gnome-software-service**
- **zeitgeist** (3 procs, logs activity to the SD card) / **deja-dup-monitor** / **goa-daemon**
- GNOME/Unity leftovers inert under LXDE: unity-settings-daemon,
  nautilus-autostart, gnome-screensaver, gnome-welcome-tour, gnome-initial-setup-*,
  gsettings-data-convert, user-dirs-update-gtk, xdg-user-dirs,
  unity-fallback-mount-helper
- No matching hardware/use: print-applet, blueman, clipit, indicator-application,
  indicator-messages, spice-vdagent, snap-userd, onboard, orca

**Revert one:** `rm ~/.config/autostart/<name>.desktop`
**Revert all:** `grep -l nano-tune ~/.config/autostart/*.desktop | xargs rm`
Screensaver: re-add `@xscreensaver -no-splash` to `~/.config/lxsession/LXDE/autostart`

## 2. I/O

- **Scheduler cfq → noop.** cfq optimises for rotational seek latency, which
  costs CPU and reorders pointlessly on flash.
  Persisted: `/etc/udev/rules.d/60-flash-scheduler.rules` (delete to revert)
- **`noatime` on /** — removes a metadata write on every file *read*.
  In `/etc/fstab`; original at `/root/nano-tune-backup/fstab.orig`

## 3. Memory tuning — `/etc/sysctl.d/99-nano-tuning.conf`

Swap here is zram (priority 5) first, 4 GB SD swapfile (priority -1) last resort.

- `vm.swappiness = 100` — compressed-RAM swap is far cheaper than evicting page
  cache and re-reading from a 22 MB/s card. Default 60 assumes a spinning disk.
- `vm.page-cluster = 0` — disables swap readahead. Faulting in 8 pages per miss
  is pointless for zram (no seek cost) and wastes decompression CPU.
- `vm.vfs_cache_pressure = 50` — hold dentry/inode cache harder.
- `vm.dirty_ratio = 10` / `dirty_background_ratio = 5` — at the defaults, 20% of
  2 GB = 400 MB of dirty pages could flush to a slow SD card in one burst,
  which is a classic multi-second UI freeze.

**Revert:** `sudo rm /etc/sysctl.d/99-nano-tuning.conf && sudo reboot`

## 4. Services disabled

Re-enable any with `sudo systemctl enable --now <name>`

- **Full KVM/QEMU virtualisation stack** on a 2 GB ARM SBC: `libvirtd`,
  `libvirt-guests`, `qemu-kvm`, `ebtables`, `ubuntu-fan`, `dnsmasq`, `spice-vdagentd`
- **`rpcbind` / `portmap`** — Sun RPC portmapper, only needed for NFS/NIS
- **`ModemManager`** (no modem), **`nvweston`** (Wayland; this box runs X11/LXDE)
- **`whoopsie` / `kerneloops` / `apport`** — crash telemetry that writes dumps to SD
- **`ureadahead`** — masked; it failed on every single boot on this kernel
- **`apt-daily.timer` / `apt-daily-upgrade.timer`** — was 33 s of boot plus
  unpredictable CPU/IO storms. Manual `apt update && apt upgrade` is unaffected.

## 5. Fixed: mystic.service (Mystic BBS)

Was in a failed state with two separate bugs:

1. **Stale `semaphore/mis.bsy`** from June 2025. `mis` refuses to start while it
   exists, so it exited instantly; `Restart=always` respawned it until systemd
   hit the start limit and gave up. Lock cleared, and an `ExecStartPre` now
   removes it on every start so this cannot recur.
2. **`StandardOutput=append:`** requires systemd 240+; this box runs 237, so it
   was silently ignored — which is why `mis.log` was always empty. Now logs to
   the journal: `journalctl -u mystic -f`

Still **enabled but not started** — I didn't start a network-facing service
without asking. Start it with: `sudo systemctl start mystic`
Original unit: `/root/nano-tune-backup/mystic.service.orig`

## 6. Disk — 1.17 GB freed

- `/var/cuda-repo-10-2-local-10.2.89` (1013 MB) and the three
  `/var/visionworks-*-repo` dirs — local `.deb` installer archives. CUDA 10.2
  and VisionWorks remain installed (`/usr/local/cuda-10.2/bin/nvcc` verified);
  these were only the packages they were installed *from*. Matching
  `/etc/apt/sources.list.d/` entries removed too, so `apt update` stays clean.
- Purged config-only leftovers: `dc`, `libsane-hpaio`, `oem-config`, `oem-config-gtk`
- Journal capped at 50 MB; superseded snap revisions removed (core18/2961,
  core20/2683, snapd/25205)

---

## Not touched (deliberately)

- **`vsftpd`** and **`rsync.service`** are both enabled and listening. Plausibly
  tied to the BBS, so I left them. If not, disabling both is free.
- **`snapd`** + 5 loop mounts, ~21 MB resident, only there for `lazygit` 0.34
  (quite old). `sudo snap remove lazygit && sudo apt purge snapd` would reclaim
  it — lazygit is also available as a plain binary.
- **`ollama`** is enabled and running with a 637 MB model. Idle it's only ~13 MB,
  but a model load will hard-swap a 2 GB box.
- **`bluetooth`**, **`avahi`** — left alone in case peripherals/discovery are used.
- **Clocks** — already optimal: `nvpmodel` MAXN, EMC pinned at 1600 MHz (max),
  CPU 1.2–1.479 GHz on `schedutil`. `jetson_clocks` would gain ~nothing here.
  (`nvpmodel -q` prints a harmless `fan mode is not set` warning — no fan.)

## The real remaining bottleneck: the SD card

Measured after tuning:

- Sequential read: **22.6 MB/s**
- 4K direct read: **2.6 MB/s (~635 IOPS)**

The bus negotiated **SDR104 SDXC (~104 MB/s capable)**, so the interface is not
the limit — the card is. It's a Longsys "USD00" (`manfid 0xad`, `oemid 0x4c53`),
mfg 01/2024, i.e. a low-grade card.

No amount of software tuning fixes 635 IOPS. Two real options:

1. **Boot rootfs from a USB 3.0 SSD.** L4T R32.5 supports it on the Nano.
   Typically 6–10x the sequential throughput and 50x+ the random IOPS. This is
   by far the largest available speedup and would also solve the 84%-full disk.
2. **A genuine A2-rated card** (SanDisk Extreme / Samsung Pro Endurance) —
   cheaper, maybe 2-3x random, but still an order of magnitude behind a SSD.

---
---

# Part 2 — Updates, same day

## Fixed: broken apt source

`/etc/apt/sources.list` line 50 had a Docker entry with `[arch=amd64]` on an
arm64 board — never installable, and it duplicated the correct `arch=arm64`
entry in `sources.list.d/docker.list`. That was the cause of the nine
"configured multiple times" warnings on every `apt update`. Removed; `apt update`
is now silent. Original: `/root/nano-tune-backup/sources.list.orig`

## Applied

- **9 packages upgraded** — that was everything the bionic archive still offers:
  apport, apport-gtk, python3-apport, python3-problem-report, distro-info-data,
  ubuntu-advantage-tools, ubuntu-pro-client, ubuntu-pro-client-l10n,
  wireless-regdb. Now **0 upgradable, 0 failed units, `dpkg --audit` clean.**
- **snapd 2.72 → 2.76.1.** core18/core20 were already current.
- **`ubuntu-fan` purged** via autoremove (it was the only autoremove candidate).
- **`/etc/default/apport` reset to `enabled=0`** — the apport upgrade flipped it
  back to 1. The init service was already disabled, so this is belt-and-braces.

## Deliberately NOT updated, with reasons

- **pip 21.3.1** — this is the *last* pip that supports Python 3.6, and
  `python3` here is 3.6.9. Upgrading pip would break it outright. Correct ceiling.
- **Python** is 3.6.9 with glibc 2.27. The deadsnakes PPA is enabled but no newer
  interpreter is installed. A newer python3.x could be added *alongside*, but
  changing the default `python3` breaks apt on Ubuntu — don't.
- **Docker 24.0.2**, **OpenJDK 11.0.19** — no newer builds exist in the bionic
  repos; Docker dropped 18.04 support. These are current-for-bionic.
- **ollama 0.8.0** — newer releases likely need a newer glibc than 2.27, and
  CUDA 10.2 is too old for modern ollama GPU paths. Working; left alone.
- **lazygit 0.34 (snap)** — that snap's publisher stopped updating. A plain
  binary from upstream would be far newer if you care.

## L4T BSP: 32.5.2 → 32.7.6 — DEFERRED (deliberately)

Verified available: `repo.download.nvidia.com/jetson/t210 r32.7` serves up to
**32.7.6-20241104234540** (JetPack 4.6.6, Nov 2024), kernel 4.9.337 vs the
current 4.9.201. Enabling it is a one-line change to
`/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` (`r32.5` → `r32.7`).

**Not done, on purpose.** Reasoning:

- **No performance benefit.** It's CVE and bugfix work, not optimization. The
  actual bottleneck is 635-IOPS storage.
- **It's a dead-end.** t210 is EOL, so 32.7.6 is the final release that will ever
  exist. You'd take a bootloader-rewrite risk to land on a terminal version — it
  does not put the board on a maintained track.
- **The downside is real:** `nvidia-l4t-bootloader` writes the boot partitions
  (mmcblk0p2–p14). A failed write means no boot, and recovery is reflashing the
  card — which wipes the 93 GB on it. No backup exists, and 19 GB free is not
  enough to make one locally.

**Do it at SSD migration time instead.** Installing JetPack 4.6.6 fresh onto a
USB SSD gets you 32.7.6 *and* the storage speedup in one move, with the SD card
untouched as a rollback. Strictly better than an in-place upgrade now.

## Ubuntu Pro / ESM

18.04 hit end of standard support April 2023 — that is why only 9 packages were
upgradable. ESM (`esm-infra` + `esm-apps`) is **available but not attached**, and
is free for up to 5 personal machines. It unlocks security updates for thousands
of packages until **April 2028**.

Pre-flight check done — **safe to attach on this board**: no `linux-generic` or
`linux-image-*` meta-packages are installed, the kernel comes from
`nvidia-l4t-kernel`, and boot is via extlinux (no grub). So ESM has no path to
install a non-Tegra kernel. (`linux-firmware` is present but is only blobs;
Tegra firmware ships separately in `nvidia-l4t-firmware`.)

To attach:

    # token from https://ubuntu.com/pro/dashboard
    sudo pro attach <TOKEN>
    sudo apt update && sudo apt upgrade

To detach later: `sudo pro detach`

ESM was **declined** for now — box is internal, behind network gear. `vsftpd` and
`rsync` were left running for the same reason.

---
---

# Part 3 — numpy/OpenCV fix, cleanup, CUDA rebuild

## FIXED: numpy and OpenCV were completely broken

`import numpy` and `import cv2` both died instantly with **SIGILL (illegal
instruction)**. This is the well-known Jetson/aarch64 fault: numpy 1.19.5's
bundled OpenBLAS misdetects the Cortex-A57 and emits an instruction the core
cannot execute. Any vision or ML work on this board was impossible.

Fixed by adding to `/etc/environment`:

    OPENBLAS_CORETYPE=ARMV8

Verified working: numpy 1.19.5, cv2 4.1.1, matmul OK.

**Note:** `/etc/environment` is only read at *login*. Existing shells/tmux panes
need `export OPENBLAS_CORETYPE=ARMV8` or a fresh login. Anything started by
systemd should be fine after a reboot.

Original: `/root/nano-tune-backup/environment.orig`

## SSH keys backed up

`/root/ssh-keys-backup-20260725.tar.gz` (8789 bytes, mode 600, 11 files) —
contains `~/.ssh/` (authorized_keys, id_rsa, id_rsa.pub, known_hosts) and all
three `/etc/ssh/ssh_host_*` keypairs.

**This is on the same SD card, so it only protects against accidental deletion,
not card failure. Copy it off the box.**

Also fixed: `~/.ssh/authorized_keys` was mode **664** (group-writable). sshd's
`StrictModes` can refuse a group-writable authorized_keys — a latent lockout.
Now `600`.

## Deleted

`/home/super/arpusa_stuff` (12.3 GB — `arp-scrapers`, `arpusa-portal`) at the
owner's request; dead project. Checked for keys/credentials first — the only
`.pem`/credential hits were library boilerplate (certifi CA bundles, twisted
test fixtures). Disk went **84% → 73%, 31 GB free**.

## OpenCV 4.5.5 + CUDA — build

Build script: `/home/super/build-opencv-cuda.sh`   Log: `/home/super/opencv-build.log`
Sources: `/home/super/opencv_src/{opencv,opencv_contrib}`

Why 4.5.5: it's the sweet spot for this toolchain — CUDA 10.2, gcc 7.5,
cmake 3.10, cuDNN 8.0. Newer OpenCV wants a newer cmake/CUDA; older (4.2–4.4)
breaks against cuDNN 8's changed API.

Confirmed by cmake before committing to the compile:

    NVIDIA CUDA:      YES (ver 10.2, CUFFT CUBLAS FAST_MATH)
    NVIDIA GPU arch:  53          <- Maxwell GM20B, the Nano's GPU
    cuDNN:            YES (8.0.0)
    GStreamer:        YES (1.14.5)
    numpy:            1.19.5 (from ~/.local)
    install path:     /usr/local/lib/python3.6/dist-packages/cv2/python-3.6

### Three non-obvious things this build had to get right

1. **Build as `super`, not root.** numpy is pip-installed under
   `~/.local`, which `sudo` cannot see. Building entirely under sudo silently
   produces an OpenCV with **no Python bindings at all**. Only `make install`
   runs as root.
2. **`OPENCV_PYTHON3_INSTALL_PATH` must be forced to `dist-packages`.** OpenCV
   defaults to `/usr/local/lib/python3.6/site-packages`, which Debian/Ubuntu
   never put on `sys.path`. Left alone, the build succeeds after ~4 hours and
   `import cv2` *still* returns NVIDIA's 4.1.1.
3. **`WITH_NVCUVID=OFF`, `WITH_GSTREAMER=ON`.** On Tegra the hardware video
   engines are not reached via NVCUVID — they're reached through GStreamer
   (`nvv4l2decoder`, `nvarguscamerasrc`). GStreamer is what actually unlocks
   NVENC/NVDEC and the CSI camera.

Also: `CUDA_ARCH_BIN=5.3` only (no PTX) — building one arch instead of many cuts
compile time and binary size a lot. `-j2` because each nvcc unit wants ~1GB and
`-j4` OOMs on 2GB. Tests/samples/java/docs off, saving hours.

### Coexistence / rollback

The new build installs to `/usr/local`, which precedes
`/usr/lib/python3.6/dist-packages` on `sys.path`, so it shadows NVIDIA's 4.1.1
**without removing the `libopencv` debs**. To roll back, delete
`/usr/local/lib/python3.6/dist-packages/cv2` (plus `/usr/local/lib/libopencv_*`)
and 4.1.1 is live again. Nothing L4T depends on was touched.

### The desktop is currently stopped

`lightdm` was stopped to free ~350 MB of RAM for the compile.

    sudo systemctl start lightdm      # bring the desktop back

Leaving it off permanently is worth considering — it's the single largest RAM
saving available on a 2GB board (see the vision-workload notes).

### Finishing the build

    cd /home/super/opencv_src/opencv/build
    sudo make install && sudo ldconfig

Then verify (in a **fresh** shell so `/etc/environment` applies):

    python3 -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"

Expect `4.5.5 1`. A `0` for the device count means CUDA didn't take.

### RESULT: built and installed successfully

Build took **~10.5 hours** (00:55 → 11:30), not the 3–5 I estimated — the CUDA
contrib modules are the bulk of it. Installed clean, `ldconfig` refreshed.

    cv2 version : 4.5.5
    cv2 module  : /usr/local/lib/python3.6/dist-packages/cv2/__init__.py
    CUDA devices: 1
    Device 0    : NVIDIA Tegra X1, 1979Mb, sm_53, Driver/Runtime 10.20/10.20
    DNN CUDA backend: available
    GStreamer 1.14.5 / v4l2 / cuDNN 8.0.0: all YES

### Measured CPU vs GPU (1080p, benchmark at ~/verify-opencv-cuda.py)

    operation                CPU (ms)   GPU (ms)   speedup
    GaussianBlur 31x31          141.4       33.8      4.2x
    Resize 2x cubic              80.1      111.6      0.7x   <- GPU SLOWER
    BGR->Gray                     1.5        3.3      0.5x   <- GPU SLOWER

**Read this before writing GPU code.** The 128-core Maxwell only wins on
compute-heavy work. Two reasons the cheap ops lose:

1. **Kernel launch overhead** (~1-3 ms) swamps any op that only takes a few ms
   on CPU. BGR->Gray costs 1.5 ms on CPU — the GPU can't dispatch that fast.
2. **The GPU shares the same LPDDR4 as the CPU.** There is no separate VRAM and
   no bandwidth advantage, so memory-bound ops (resize, colour convert) gain
   nothing. Only arithmetic-dense ops (big convolutions, DNN inference) win.

Practical rule: **upload once, chain many heavy ops on-GPU, download once.**
Round-tripping per operation will be slower than staying on CPU. Note the
benchmark above *excludes* transfer time — include it and the losing cases get
worse still.

The real payoff here is not `cv2.cuda.*` for basic filtering — it's the **DNN
CUDA backend** for neural-net inference, plus GStreamer reaching NVENC/NVDEC.

---
---

# Part 4 — Tektronix vector display, and going headless

## Files

    tekvector.py   geometry + still renderer (PNG / MP4)
    tekscreen.py   live renderer via X (cv2.imshow) — superseded
    tekfb.py       live renderer straight to /dev/fb0, no X   <- what runs now
    tekctl.sh      manual start/stop/status
    verify-opencv-cuda.py   CPU vs GPU benchmark
    /etc/systemd/system/tek-display.service   starts tekfb at boot

## THE DESKTOP IS GONE

    systemctl stop/disable lightdm
    systemctl set-default multi-user.target      # boots to console now

To undo:

    sudo systemctl set-default graphical.target
    sudo systemctl enable --now lightdm

**Caveat worth knowing:** `tek-display` continuously owns `/dev/fb0`, so the
tty1 text console is painted over and effectively invisible while it runs. SSH
is unaffected. If you need the local console back:

    sudo systemctl stop tek-display

## Optimisation, measured at every step

| stage | fps (apple, 814 edges) |
|---|---|
| first working version, via X | 4.6 |
| batched polylines + LUT composite, via X | 16.5 |
| framebuffer + CPU pyramid bloom + vectorised projection | 43.8 |
| grain views + folded scale | **45.5** |

Nearly 10x. What actually mattered, in order:

1. **Per-line Python loop → one `cv2.polylines`**: 50.2 ms → 2.9 ms. 814 calls
   each crossing the Python/C boundary.
2. **Composite → single-channel + 256-entry LUT**: 139.8 ms → ~13 ms. It was
   doing ~8 full-frame 3-channel float passes and was memory-bandwidth bound.
   Phosphor colour, white-core saturation and screen tint all depend only on
   intensity, so they collapse into a lookup.
3. **CUDA bloom → CPU pyramid**: 29.3 ms → 6.8 ms. See below.
4. **Vectorised projection**: the per-edge Python loop in `build_segments` cost
   7 ms/frame; `build_pts` does it as one fancy-index in ~0.5 ms.
5. **X → framebuffer**: `imshow` was pushing 2.7 MB through the X socket per
   frame. The panel is BGRA with no padding, identical to OpenCV's byte order,
   so frames now go into an mmap with no conversion at all.
6. **grain via views, not `np.roll`**: roll copied 2.4 MB/frame. A double-height
   buffer sliced at a random offset copies nothing.

## The uncomfortable finding: CUDA lost

Measured, bloom at 1024x600:

    3x CUDA blur, half-res, upload/download each ... 29.3 ms
    same but one upload/download, combine on GPU ... 28.6 ms   (so: not transfers)
    pure CPU, half-res                           ... 14.2 ms
    CPU pyramid, wide blurs at quarter res       ...  6.8 ms   <- chosen

**The GPU is 4x slower than the CPU for this workload.** That is consistent with
the earlier 1080p benchmark rather than contradicting it: the GPU won 4.2x on a
31x31 blur at 1920x1080, but at 512x300 kernel-launch overhead dominates and
NEON on the A57s simply wins. The crossover is resolution-dependent.

So the 10-hour CUDA build is *not* what makes this display fast. It still earns
its place for the **DNN CUDA backend** (neural-net inference, where the GPU wins
decisively) and for full-resolution image work — but reach for it based on
measurement, not on the assumption that GPU beats CPU.

## Display mode

With X gone the framebuffer sits at **1024x600**, not the 1280x720 X was using.
`/sys/class/graphics/fb0/modes` lists `D:1024x600p-60` first — `D` = EDID
detailed timing — so 1024x600 is almost certainly the panel's native mode and X
was driving it off-native with the panel scaling. Modes up to 1920x1080p-60 are
available if a different one is wanted (`fbset`).

## Result

    RAM used       1218 MB -> 988 MB
    RAM available   751 MB -> 1099 MB     (+348 MB)
    processes         241 -> 224
    Xorg                1 -> 0
