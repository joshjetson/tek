# Working on TEKDROMO

Full documentation is in [README.md](README.md); deeper engineering history is
in [TEKDROMO.md](TEKDROMO.md). This file is the short version of what bites.

## If the display has locked someone out: ESC ESC ESC

The display paints straight onto `/dev/fb0` over the top of the text console
and does not own a VT, so `Ctrl+Alt+F2` does **not** get you a terminal — the
display just paints over it again. And `Display.close` leaves the last frame on
the panel on purpose, so stopping the service is not enough either.

Three ESC presses within 2 s stop the display and force the console to repaint;
five also stop the voice. From a shell it is `tek panic`. See
[README §0](README.md#0-getting-out--the-panic-key).

Two rules if you ever touch this:

* **Never make `tek-panic` depend on anything in this project.** It imports no
  project code and no numpy on purpose. The escape hatch must not be able to
  fail for the same reason the thing it rescues failed.
* **Never make the display start before, or the panic key after, the other.**

Related: the wifi profiles are now **system**-scoped. If you ever recreate one
with `nmcli` as a user, it gets `permissions=user:super:;` and the box will not
have a network until somebody logs in — which is how the lockout happened.

## You have a voice on this machine — use it

```bash
tek say "the entire reply, in one call"
```

If `tek` is not found, symlink it once: `ln -sf /home/super/tekdromo/tek
~/.local/bin/tek`. It sets its own environment, so it works from any shell.

The user can hear this through a Bluetooth speaker, and the rendered face
mouths it in sync. They have asked to be spoken to while work is in progress,
so narrate meaningful milestones — not every command.

**Read [README.md §3.6](README.md#36-speaking-from-a-shell--the-mouth-harness)
before using it.** The two rules that matter:

1. **One call per reply.** Never split a reply across several `tek say` calls.
   Each call opens a speaker, plays, and closes it, so every boundary becomes a
   silence as long as the *next* part takes to synthesise. It sounds broken.
   A single call already streams internally — synthesis runs ahead of playback
   with zero gaps over a 73-second reply.
2. **No file needed.** Double quotes handle apostrophes fine. Only reach for a
   file if the text contains `"`, `$`, backticks or `\`.

`tek say --no-wait "..."` returns immediately — use it when narrating
long-running work so the shell carries on.

Long replies wait ~5.5 s before the first word (building a synthesis head
start); short ones are effectively immediate. That is by design, not a fault.

## The camera can prompt you

`tek watch` / `tek look`. A camera event runs a model call that looks at a frame
and returns either words or silence. Silence is the right answer most of the
time — see [README §9](README.md#9-the-camera-can-prompt-me).

If you change the brain: it must run in a NEUTRAL cwd. Running it in this
project puts CLAUDE.md in scope, so it learns it has a voice and may speak for
itself as well as returning words to speak.

## Audio traps

* **The speaker "randomly disconnecting" is almost always PulseAudio.** Its
  default `exit-idle-time` is 20 s, so with no client connected the daemon
  exits and takes A2DP with it; the next client autospawns a new one. 96
  daemon restarts were logged in one day. Now `-1` in
  `/etc/pulse/daemon.conf`. Check with `pulseaudio --dump-conf | grep
  exit-idle` before suspecting BlueZ.
* **Never trust the PulseAudio default source.** The mic is inside the webcam,
  so a camera replug moves the default to the Tegra onboard input, which has
  nothing plugged into it, and it never moves back. Use
  `io.working_source()`, which probes for a *varying* signal — a dead input is
  not silent, it is constant.
* **Never use `@DEFAULT_MONITOR@`.** PulseAudio resolves it once, at stream
  creation, and never moves the stream. The display starts before the speaker
  connects, so it binds to the analog monitor and stays there — a flat
  waveform panel on every boot. Use `io.default_monitor()` and re-check it.
* **Do not probe audio devices on a timer.** Probing opens and kills a
  recorder on every candidate. Cache it — `working_source` invalidates on the
  device list changing, which is the only thing that can alter the answer.
* **`abs()` on int16 overflows at -32768**, so a railed mic reports a healthy
  amplitude. Convert to int32 first.
* **`parec` stdout `read(n)` blocks forever** if the source never delivers n
  bytes, which looks exactly like a hung script.

## Environment traps

* **`OPENBLAS_CORETYPE=ARMV8` is mandatory.** Without it `import numpy` and
  `import cv2` die with SIGILL. Set it in every service, script and cron job.
* **`XDG_RUNTIME_DIR=/run/user/1000`** is needed to reach the voice socket and
  PulseAudio. `tek-display` and `tek-voice` must agree on it or they silently
  never connect.
* **`LD_LIBRARY_PATH=/home/super/tekdromo/lib`** is required for anything using
  `vosk` — its `libvosk.so` needs a newer libstdc++ than this box has. The
  system runtime is deliberately untouched.
* **Python is 3.6.9.** No walrus `:=`, no f-string `=`, no
  `subprocess.capture_output`, no `ast.end_lineno`.
* **systemd is 237.** `StartLimitIntervalSec` belongs in `[Unit]`; in
  `[Service]` it is silently ignored. Drop-ins apply in lexicographic *filename*
  order across all directories — `/etc` does not automatically beat `/lib`.

## Memory lives in Postgres, in a container

`tekdromo/memory/`, backed by `deploy/docker-compose.yml` on port 5433. Bring it
up with `docker-compose up -d` then `tek memory migrate`. See
[README §9b](README.md#9b-memory--it-remembers).

Two rules:

* **A dead journal must never break the face.** Everything in `memory/` degrades
  to "no memory" and returns a default rather than raising — but it *logs*, and
  keeps `last_error` for `tek memory status`. Silent degradation is the bug; the
  brain-returning-None-quietly incident is why.
* **Do not import the submodules by their short names.** `from tekdromo.memory
  import migrate` gives you the convenience *function*, not the module — the
  package exports `migrate_mod`, `db_mod`, `recall_mod`, `store_mod` for that.

## Do not break the display

`tek-display` renders at ~30 fps and the project's central invariant is that it
**never stops**. Anything heavy runs in a separate process. Before claiming a
change is safe, check the frame rate actually held:

```bash
journalctl -u tek-display -n 5 | grep fps
```

## Measure, do not assume

This codebase is full of comments recording things that were measured and did
*not* help — CUDA bloom being 4× slower than a CPU pyramid at this resolution,
numpy fancy-indexing being 3.5× slower than `cv2.LUT`. Several real bugs here
were found only because a number was checked rather than eyeballed, and at
least one shipped because a code comment asserted behaviour nobody had
verified (`pacat`'s stdin has no backpressure; it accepted 3.0 s of audio in
0.01 s).

Run the suite before and after:

```bash
for t in tests/*.py; do printf "%-22s " "$t"; python3 "$t" 2>&1 | tail -1; done
```

## Shell

`pkill -f <pattern>` will match **this session's own command line** if the
pattern appears in it, and kills the shell. Target processes by PID or by
`/proc/*/exe` instead.

## Git

Commit messages here explain *why*, with measured numbers, and do not mention
Claude.
