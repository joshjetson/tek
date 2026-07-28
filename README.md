# TEKDROMO

A **Tektronix 4014 storage-tube vector display**, running on a Jetson Nano 2GB,
rendering an animated human face that watches the room, listens, and talks.

Not a smart-speaker puck. A thing with a location, an aesthetic, and continuity
— the face *is* the interface.

<sub>Everything below was built on the target hardware: 4× Cortex-A57 @1.48GHz,
2GB RAM, Python 3.6.9, glibc 2.27. Those constraints shaped nearly every
decision here, and where something looks odd it is usually because the obvious
approach does not exist on this box.</sub>

---

## Contents

0. [**Getting out — the panic key**](#0-getting-out--the-panic-key) ← if you are staring at a face you cannot dismiss
1. [What it is](#1-what-it-is)
2. [Quick start](#2-quick-start)
3. [The voice](#3-the-voice) — [all 38 Piper voices](#34-every-english-piper-voice) · [**how to speak (read this first)**](#36-speaking-from-a-shell--the-mouth-harness)
3b. [**Listening** — the ear](#3b-listening--the-ear)
4. [Architecture](#4-architecture)
5. [Hardware notes and traps](#5-hardware-notes-and-traps)
6. [Testing](#6-testing)
7. [Services](#7-services)
8. [Measured results](#8-measured-results)
9. [The camera can prompt me](#9-the-camera-can-prompt-me)
10. [What is next](#10-what-is-next)

---

## 0. Getting out — the panic key

### Press ESC three times.

The display stops and the text console comes back, in about **0.7 s**. Five
presses instead of three stops the voice as well.

```
ESC ESC ESC        console back
ESC ESC ESC ESC ESC   console back, and silence

sudo systemctl start tek-display     # bring the face back
```

From a shell — local or over SSH — the same thing is `tek panic`
(or `tek panic quiet`).

### Why this needs to exist

This was found the hard way. The machine rebooted and became genuinely
unusable:

* The display writes straight to `/dev/fb0`, over the top of the text console.
  It does not own a VT, so **`Ctrl+Alt+F2` does not help** — switching consoles
  repaints the screen and the display simply paints over it again 33 ms later.
* `Display.close` deliberately leaves the last frame on the panel, because it
  makes service restarts invisible ([§8](#8-measured-results)). So even
  stopping the display leaves the face sitting there.
* The wifi was configured as a **user** connection, so it did not come up until
  somebody logged in — and nobody could see the login prompt to log in.

The only way back in was to blind-type a username and password roughly a
hundred times until they happened to land in the right fields.

### What was changed

| | Before | After |
|---|---|---|
| Wifi profiles | `permissions=user:super:;` — needs a login session | system-scoped, connects at boot |
| Wifi secret | (already `psk-flags=0`, fine) | unchanged, verified |
| Escape hatch | none | `tek-panic.service`, ESC ×3 |
| tty1 | login prompt, typed blind | `agetty --autologin super` |

`tty1` autologin means a shell is already waiting the moment the console comes
back, instead of a login prompt you cannot see. It is a physical-access-only
box on a home LAN; **SSH still requires a password**.

The NetworkManager change is the important half: **SSH now works before anyone
logs in**, which is the real fix. The panic key is the fallback for when the
network is also gone.

### How the panic key is built

`tekdromo/panic.py`, run as root by `tek-panic.service`. Every choice in it is
about the failure case rather than the happy path:

* **It is a separate process.** A panic key inside the thing you are escaping
  from is not a panic key — the case you need it for is the one where that
  process is wedged.
* **It imports nothing from this project, not even numpy.** The escape hatch
  must not be able to fail for the same reason the thing it rescues failed.
* **It rescans `/dev/input` every 2 s.** The keyboard gets plugged in *after*
  things go wrong, so enumerating once at startup would miss the only keyboard
  that ever matters. It watches every device rather than guessing which are
  keyboards — nothing but a keyboard sends `KEY_ESC`, and this box already has
  a webcam that registers as one.
* **Autorepeat does not count** (`value == 2`), so leaning on the key does
  nothing.
* **The window is measured on the monotonic clock**, because this box sets its
  clock from the network a minute into boot and a wall-clock jump mid-chord
  would otherwise fire it by itself.
* **It forces the console to repaint** by switching VT away and back. Stopping
  the display is only half the job.

Measured on the live system: stop 0.22 s, repaint 0.44 s, and the panel goes
from 127,576 lit pixels to 25,267 — the face gone, a console in its place.
`tests/panic_unit.py` proves the chord logic and drives a **real virtual
keyboard through the kernel input stack**, created *after* the watcher starts,
because that ordering is the entire point. `tools/panic_e2e.py` fires it at the
live service.

### Still stuck?

`Alt+SysRq` is fully enabled (`kernel.sysrq = 1`). `Alt+SysRq+R E I S U B` is
the last resort — it will reboot the box without corrupting the filesystem.

---

## 1. What it is

Three services that together make a face which can hold a conversation.

| Service | Job |
|---|---|
| `tek-display` | Renders the head to `/dev/fb0` at ~30 fps. Never stops. |
| `tek-voice` | Speech in and out. Owns the voice, the speaker, the mouth stream. |
| `tek-bluetooth` | Keeps the Bluetooth speaker connected and audio routed to it. |

The aesthetic is a genuine constraint, not decoration. A Direct-View Storage
Tube walks an electron beam point-to-point and the phosphor holds the charge,
so: **no fills, no shading, no scanlines** (it does not scan — adding them is
the classic fake tell), constant beam intensity, and an all-or-nothing erase
flash.

The head is not a mesh. It is an **implicit height field sliced into
iso-contours**, which is the central idea of the whole project:

```
z(x,y) = skull + forehead + brow + nose + cheeks + lips + chin
         − eyes − philtrum − nostrils
for z = .95 down to −.25 step −.05:   extract contours → emit vectors
```

Because contours are level sets of the real surface, **they flow around every
feature for free**. An earlier attempt drew feature curves onto an undeformed
mesh and every one of them read as a decal.

<sub>[↑ Contents](#contents)</sub>

---

## 2. Quick start

Put `tek` on your PATH once (it sets its own environment — `OPENBLAS_CORETYPE`,
`XDG_RUNTIME_DIR`, `LD_LIBRARY_PATH` — so it works from any shell, cron job or
service):

```bash
ln -sf "$PWD/tek" ~/.local/bin/tek
```

```bash
tek say "hello"                 # speak; the face mouths it (see 3.6 first)
tek status                      # which voice, is it speaking
tek voices                      # what is installed and working
tek audition --piper            # hear every Piper model, out loud
tek voice en_US-kusal-medium    # set the default, permanently
tek listen                      # watch the mouth stream live
```

A fresh checkout has **no voice models and no speech model** — they are large
binaries, deliberately not in git:

```bash
tools/fetch_voice.sh --list                   # all English Piper voices
tools/fetch_voice.sh en_US-kusal-medium       # ~61MB each
# speech recognition model (~68MB):
curl -sSL -o m.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q m.zip -d models/ && rm m.zip
```

<sub>[↑ Contents](#contents)</sub>

---

## 3. The voice

### 3.1 Chosen voice

**`en_US-kusal-medium`** — picked by ear, out loud, through the actual Bluetooth
speaker in the actual room. Judging a voice from a waveform or on headphones is
a different test.

Change it any time; nothing is baked in:

```bash
tek voice en_GB-alba-medium
```

The choice is stored in `~/.config/tekdromo/voice.json` — outside the repo,
because it is a per-machine preference rather than code. The same checkout on a
box with different speakers may want a different voice.

### 3.2 Shortlist

Three finalists came out of listening to 20 voices:

| Voice | | Why |
|---|---|---|
| `en_US-kusal-medium` | **chosen** | US English |
| `en_GB-alba-medium` | shortlisted | Scottish — the most distinctive of the three |
| `en_US-amy-medium` | shortlisted | US English, warmer and softer than lessac |

### 3.3 There are far more than 38

The catalogue lists **38 English voices**, but that undersells it. Seven of them
are *multi-speaker* models containing **1,975 more voices** — `en_US-libritts`
alone holds 904 speakers, `en_GB-vctk` holds 109.

Of the 38, only a handful are genuinely distinct speakers not yet heard:
`danny`, `kathleen`, `reza_ibrahim`. Eight more are simply `low`/`high` quality
variants of voices already sampled.

> **On `high` quality:** medium models synthesise at **0.72× real-time** here.
> High-quality models are larger networks and will land near or above 1.0×,
> which means a pause before speech starts. On this board, medium is the right
> trade.

> **Multi-speaker models are not wired up yet.** They need an extra
> `speaker_id` input that `PiperVoice` does not pass. Small change, not done.

### 3.4 Every English Piper voice

"Sampled" means it was played aloud through the speaker during voice selection.

#### en_US

| Voice | Quality | Speakers | Sampled | Note |
|---|---|---|---|---|
| `en_US-amy-low` | low | 1 | — |  |
| `en_US-amy-medium` | medium | 1 | yes | shortlisted |
| `en_US-arctic-medium` | medium | 18 | — |  |
| `en_US-bryce-medium` | medium | 1 | yes |  |
| `en_US-danny-low` | low | 1 | — |  |
| `en_US-hfc_female-medium` | medium | 1 | yes |  |
| `en_US-hfc_male-medium` | medium | 1 | yes |  |
| `en_US-joe-medium` | medium | 1 | yes |  |
| `en_US-john-medium` | medium | 1 | yes |  |
| `en_US-kathleen-low` | low | 1 | — |  |
| `en_US-kristin-medium` | medium | 1 | yes |  |
| `en_US-kusal-medium` | medium | 1 | yes | **CHOSEN** |
| `en_US-l2arctic-medium` | medium | 24 | — |  |
| `en_US-lessac-high` | high | 1 | — |  |
| `en_US-lessac-low` | low | 1 | — |  |
| `en_US-lessac-medium` | medium | 1 | yes |  |
| `en_US-libritts-high` | high | 904 | — |  |
| `en_US-libritts_r-medium` | medium | 904 | — |  |
| `en_US-ljspeech-high` | high | 1 | — |  |
| `en_US-ljspeech-medium` | medium | 1 | yes |  |
| `en_US-mike-medium` | medium | 1 | yes |  |
| `en_US-norman-medium` | medium | 1 | yes |  |
| `en_US-reza_ibrahim-medium` | medium | 1 | — |  |
| `en_US-ryan-high` | high | 1 | — |  |
| `en_US-ryan-low` | low | 1 | — |  |
| `en_US-ryan-medium` | medium | 1 | yes |  |
| `en_US-sam-medium` | medium | 1 | yes |  |

#### en_GB

| Voice | Quality | Speakers | Sampled | Note |
|---|---|---|---|---|
| `en_GB-alan-low` | low | 1 | — |  |
| `en_GB-alan-medium` | medium | 1 | yes |  |
| `en_GB-alba-medium` | medium | 1 | yes | shortlisted |
| `en_GB-aru-medium` | medium | 12 | — |  |
| `en_GB-cori-high` | high | 1 | — |  |
| `en_GB-cori-medium` | medium | 1 | yes |  |
| `en_GB-jenny_dioco-medium` | medium | 1 | yes |  |
| `en_GB-northern_english_male-medium` | medium | 1 | yes |  |
| `en_GB-semaine-medium` | medium | 4 | — |  |
| `en_GB-southern_english_female-low` | low | 1 | yes |  |
| `en_GB-vctk-medium` | medium | 109 | — |  |

Fetch any of them:

```bash
tools/fetch_voice.sh en_US-ryan-high en_GB-cori-medium
```

### 3.5 Fallback voices

Piper is what ships, but three native synths are kept behind the same `Voice`
interface — `pico` (SVOX Pico), `flite`, `espeak`. A household assistant that
cannot talk is useless, so if a model file is missing or onnxruntime fails to
load, speaking badly beats not speaking.

`espeak` is never an extra dependency: **it is Piper's phonemiser.**

### 3.6 Speaking from a shell — the mouth harness

> **If you are an assistant driving this machine, read this section before
> using the voice.** Two mistakes were made here the hard way and both are
> avoidable.

```bash
tek say "the entire reply, in one call"
```

That is all it takes. No file, no temp buffer, no setup. The text goes from the
command straight to the speaker, and the face mouths it from the same PCM.

#### Rule 1 — pass the whole reply in ONE call

**Do not split a reply across several `tek say` calls.** It is the obvious thing
to do and it sounds broken.

Each call opens a speaker, plays, closes it, and returns. So every boundary
becomes a silence as long as the *next* part takes to synthesise — roughly 0.7×
its spoken length. A listener describes the result as *"too many breaks, it is
not fluid"*, and they are right.

Inside a single call the service already streams properly: it splits the text
into ramped chunks, synthesises **ahead of** playback, and writes everything to
one continuous sink. Measured on a 73-second reply:

| | one call | split across 8 calls |
|---|---|---|
| Gaps over 250 ms | **0** | one at every boundary |
| Median frame gap | 20 ms | 20 ms, with ~1.5 s holes |

#### Rule 2 — expect a beat before long replies, and none before short ones

| Reply length | Time to first word | Why |
|---|---|---|
| A sentence or two | effectively immediate | synthesis finishes before the head-start threshold is reached, so the wait is skipped |
| A paragraph or more | **~5.5 s** | builds `MIN_LEAD_S` of buffered audio first |

That head start is not padding. Synthesis runs at **0.71× real-time**, so each
second of playback buys only 0.4 s of lead — which means the risk of playback
overtaking synthesis is entirely at the *start*, before any lead exists. Without
it, a long reply breaks up three or four times in the first half and is smooth
thereafter.

The effect is that conversation paces itself about right: quick answers come
back instantly, a considered one takes a beat before it starts.

#### Shell quoting

Double quotes are fine, and so are apostrophes inside them:

```bash
tek say "I don't think that's the real problem."
```

Avoid `"`, `$`, backticks and `\` in the text — the shell will eat them. If the
wording needs any of those, write it to a file and pass it in:

```bash
tek say "$(tr '\n' ' ' < reply.txt)"
```

A file is **never** required for ordinary sentences. Reaching for one by default
is just caution about quoting, not a limitation of the harness.

#### Other flags

```bash
tek say --no-wait "working on it"        # return immediately; narrate progress
tek say --voice pico "compare this"      # one-off voice, default unchanged
tek listen                               # watch the mouth stream live
```

`--no-wait` is the right choice when narrating long-running work, so the shell
carries on while the sentence plays.

#### Writing for the ear

Text that reads well is not the same as text that *hears* well. Shorter
sentences, less subordinate-clause nesting, and real punctuation — the model was
trained **with** punctuation phonemes, so commas and full stops are literally
its pause cues. A paragraph with no punctuation comes out as a breathless
monotone.

<sub>[↑ Contents](#contents)</sub>

---

## 3b. Listening — the ear

```bash
tek ears            # state: what it is listening to, and what it has heard
tek ears off        # stop listening (the microphone is closed, not ignored)
```

Say **"hey tek"** and then a question — in one breath, or as two. It answers out
loud.

```
mic → Gate → VAD → wake grammar → free decode → brain → speech
```

Proven end to end through real air, nothing stubbed
(`tools/ears_e2e.py` plays the wake word through the speaker so the mic picks it
out of the room exactly as it would pick up a person):

```
ears: woken (1.7s) - waiting 8s for a command
ears: heard 'what day of the week is it' (3.9s)
event speech: saying "It's Monday."
```

Four things it must not do, each of which shaped the design:

**It must not hear itself.** The mic picks the speaker up at ~11× ambient and
Vosk transcribes Piper *perfectly* — the loopback test reads back 12 of 12
keywords. Unchecked, it answers its own replies forever. `Gate` feeds silence to
the segmenter while the face is speaking, plus a 1.2 s tail for A2DP latency and
reverb. Verified by saying "Hey Tek" out loud three times: **0 wakes**.

**It must not transcribe the household.** The wake grammar runs on everything,
but it can only emit its four phrases or `[unk]` — it cannot produce a
transcript. Full decoding happens only after the wake word matches. Local only,
wake-word gated, nothing leaves the house.

**It must not trust "the default microphone".** The mic is inside the webcam, so
a camera replug moves the PulseAudio default to the Tegra onboard input — which
has nothing plugged into it — and it never moves back. That was observed live:
two recorders sitting on a dead device while the real mic was idle, with the ear
reporting itself perfectly healthy and hearing nothing. `io.working_source()`
probes candidates for a *varying* signal, because a dead input is not silent,
it is constant.

**It must not go quietly deaf.** Neither "wrong device" nor "no frames" ends the
stream, so a reader just sits blocked forever. A watchdog closes the source to
break it loose and the reader re-probes.

### The model, and why answers were thin

Answers used to be one flat line because **the prompt asked for that** — "one
or two short sentences", and then `parse()` cut whatever survived at 400
characters. No model was going to fix that. Length is now per event: a camera
greeting is still a sentence or two, an answer is not.

The model was `haiku`, chosen "because latency matters more than depth". That
was never measured. Same prompts, same box, three questions each:

| model | mean |
|---|---|
| `haiku` | **10.5 s** — the *slowest* |
| `sonnet` | 7.7 s |
| `opus` | **7.5 s** — fastest *and* best |

Latency here is dominated by CLI startup and session setup, not by the model,
so the "fast" choice cost quality and bought nothing. Default is now `opus`.

Replies are **spoken as they are written** (`--include-partial-messages`), so
time-to-first-word no longer grows with the length of the answer — which is
what makes depth affordable. Measured live: a 624-character answer began
speaking at 12.7 s and ran 35 s without a gap (15 chunks, producer at 0.92×).

`tools/brain_bench.py` reproduces the model comparison.

A spoken question **skips the camera cooldown**. That cooldown exists to stop
the camera remarking on an ordinary evening; applying it to someone who spoke
directly to you reads as broken, not as restraint.

<sub>[↑ Contents](#contents)</sub>

---

## 4. Architecture

### 4.1 The DRY spine

Every stage of a voice loop either consumes audio or produces it. So there is
**one** PCM contract — 16 kHz, mono, `int16`, 20 ms frames — and **one** pair of
abstractions:

```
Source  yields frames        Sink  accepts frames
```

Everything else is expressed in those terms, which buys three things that are
not merely tidy:

* **A microphone and a WAV file are the same type**, so the entire pipeline was
  built and tested before any microphone existed.
* **The mouth is a Sink.** The audio going to the speaker and the audio driving
  the face are not two signals kept in agreement — they are *one signal with two
  consumers*. Lip-sync is a property of the topology.
* **Wake word and transcription are one model with two grammars**, not two
  engines.

`pcm.RATE` is 16 kHz because that is native for Vosk *and* Whisper, so the
recognition path needs no conversion at all. 20 ms because WebRTC's VAD accepts
only 10/20/30 ms frames. Resampling happens **only** at hardware edges.

### 4.2 Module map

```
tekdromo/
  app.py          display application, frame loop, startup
  anatomy.py      measured shape + FDL constants + blob/ridge primitives
  field.py        the surface equation, neck unioned with max()
  contour.py      marching squares, ears, back shell
  rig.py          expression rig: controls -> regions -> cached contours
  geometry.py     rotate, project, back-face cull
  phosphor.py     bloom, phosphor LUT, the storage-tube look
  starfield.py    amber backdrop, same renderer
  camera.py       face tracking (background thread) + critically-damped follow
  voice_link.py   display end of the voice seam (~30 lines)
  voice/
    pcm.py        THE audio contract + resample/envelope
    io.py         Source/Sink; Mic/Wav/Tone, Speaker/Wav/Null/Tee/Delay
    vad.py        WebRTC VAD segmenter with pre-roll and hang-over
    stt.py        Vosk recognition; wake grammar + free decode
    tts.py        Voice interface; Piper/Pico/Flite/Espeak
    phonemes.py   espeak-ng -> IPA -> phoneme IDs (replaces piper-phonemize)
    bus.py        line-JSON over a Unix socket, shared by both processes
    service.py    the wiring; tek-voice entry point
    cli.py        the `tek` command
```

### 4.3 The expression rig

```
EXPRESSIONS  named presets → control values
     ↓
CONTROLS     10 named scalars — the ONLY animation state
     ↓
REGIONS      a bbox + a field function + which controls touch it
     ↓
CACHE        re-contour ONE region, memoised on quantised controls
```

The face is a field, so an expression is just different numbers in the
equation — no blendshapes, no skinning. A full rebuild is ~4 s, but an
expression only disturbs a small box, and the field outside that box is
unchanged so contours still meet the border exactly. Warm cost: **0.27 ms per
frame**.

Adding an expression is one line. Adding a control is one line plus a term in a
field function. Blink is *not* an expression — it is a reflex on its own timer
that clamps `eye_open`, so it works during any expression and during speech.

### 4.4 Piper without piper-phonemize

The official Piper wrapper needs Python 3.9+; this box has 3.6.9.
`piper-phonemize` has no 3.6 build at all and is the real blocker — but it is
only a C++ shim over espeak-ng, **which is already in apt**. So:

```
text → espeak-ng --ipa → the model's own phoneme_id_map → onnxruntime → PCM
```

and the dependency disappears. Two risks were predicted and both evaporated on
measurement:

| Predicted | Actual |
|---|---|
| espeak-ng 1.49.2 (2018) too old; build 1.52 from source | **100% phoneme coverage**, 0 unmapped. No build needed. |
| ORT 1.10 (last cp36 aarch64 wheel) won't load a 2023 VITS export | Loads clean |

espeak's `--ipa` **drops punctuation** and emits a newline instead, but the
model was *trained with* punctuation phonemes — they are its pause cues. Clauses
are phonemised separately and the punctuation put back, which is what gives it
sentence rhythm rather than a flat monotone.

A side benefit: `speech.from_envelope()` hardcodes `rounding=0` and explains
why — *"real viseme shape needs phoneme information, which an envelope does not
carry."* The Piper path **has** the phonemes, so the mouth rounds on /u/, /o/,
/w/.

<sub>[↑ Contents](#contents)</sub>

---

## 5. Hardware notes and traps

Read these before debugging anything.

1. **`OPENBLAS_CORETYPE=ARMV8` is mandatory.** Without it `import numpy` and
   `import cv2` die with **SIGILL** — numpy 1.19.5's OpenBLAS misdetects the
   A57. Set it in every new service and cron job.
2. **systemd is 237.** `StandardOutput=append:` is silently ignored.
   `StartLimitIntervalSec`/`StartLimitBurst` must be in **`[Unit]`**; in
   `[Service]` they are silently ignored.
3. **systemd drop-ins are applied in lexicographic *filename* order across all
   directories.** `/etc` does **not** automatically beat `/lib`. An override
   named `10-foo.conf` loses to `nv-bar.conf`; it must sort *after*.
4. **JetPack disables A2DP.** NVIDIA ships a drop-in starting `bluetoothd` with
   `--noplugin=audio,a2dp,avrcp`. A speaker can pair but can never carry audio.
5. **The D-Bus policy denies uid 1000 access to `org.bluez`**, so PulseAudio
   cannot register a media endpoint. It surfaces as `Protocol not available`
   from BlueZ and `No default controller available` from `bluetoothctl` —
   neither of which points at permissions. Group membership cannot fix an
   already-running PulseAudio, so the policy names the user.
6. **`vosk`'s `libvosk.so` needs GCC 11+ libstdc++**; this box has GCC 7.5
   (`GLIBCXX_3.4.25`) and only 0.3.44/0.3.45 ship aarch64 wheels, both of which
   fail. A newer libstdc++ built *for* bionic lives in `lib/` and is used by
   `tek-voice` **only**, via `LD_LIBRARY_PATH`. The system runtime is
   deliberately untouched — replacing it globally risks the CUDA/OpenCV stack.
7. **`pacat`'s stdin has no backpressure.** It accepted **3.0 s of audio in
   0.01 s**. Nothing may use write progress as an audio clock.
8. **No desktop.** X/lightdm disabled, default target `multi-user`.
   `tek-display` owns `/dev/fb0` continuously. `sudo systemctl stop tek-display`
   to get the console back.
9. **L4T is pinned at 32.5.2.** 32.7.6 exists and is the last release for t210,
   deliberately not taken — it rewrites the bootloader and there is no backup of
   the card.

<sub>[↑ Contents](#contents)</sub>

---

## 6. Testing

```bash
for t in tests/*.py; do python3 "$t"; done
```

| Test | Covers | Hardware needed |
|---|---|---|
| `smoke.py` | end-to-end frame render | none |
| `holecheck.py` | every expression + blends leave no holes | none |
| `follow_unit.py` | camera follow, including integrator windup | none |
| `boot_camera.py` | camera attaches when it appears, not only if present | none |
| `voice_pcm.py` | framing, resampling, envelope, phoneme coverage | none |
| `voice_loopback.py` | the whole Source/Sink pipeline on stubs | none |
| `voice_bus.py` | protocol framing, dead subscribers, slow clients | none |
| `voice_stt.py` | recognition + VAD, using **Piper as the test signal** | none |
| `voice_watch.py` | the camera-prompt decision gate, on a stub brain | none |
| `hud_unit.py` | clock, scope and face panels | none |
| `panic_unit.py` | the escape hatch, incl. a **real uinput keyboard** | root for the last part |
| `voice_ears.py` | the self-hearing gate, wake/command logic, misheard wake words | none |
| `voice_lipsync.py` | reads `/dev/fb0` while really speaking | display + voice |

Three are disruptive and therefore live in `tools/`, not `tests/`:

| Tool | What it proves that the suite cannot |
|---|---|
| `panic_e2e.py` | the *installed service* stops the display, not just that a callback fired |
| `panic_screen.py` | reads `/dev/fb0` before and after — the console really comes back |
| `camera_replug.py` | deauthorizes the camera on the USB bus: a **real** unplug |
| `mic_check.py` | the mic produces a *varying* signal, not just samples |
| `mic_room.py` | speaks and records the room — the acoustic path, through air |
| `ears_e2e.py` | says the wake word aloud and checks it answers |
| `scope_check.py` | reads the waveform panel out of `/dev/fb0` while sound plays |

`camera_replug.py --hold` is the important one. It holds the old `/dev/videoN`
open across the unplug so the kernel cannot reuse that minor number, which
forces the camera to re-enumerate at a different index — the exact case that
broke when the camera was swapped. All three put things back afterwards.

`voice_stt.py` is the one worth noting: with no microphone available, **Piper
speaks the test sentences and Vosk reads them back**. The loop closes on-box.
That proves wiring, rates, framing, segmentation and grammar — it does *not*
prove acoustic performance, because synthetic speech has no room noise, reverb
or distance.

<sub>[↑ Contents](#contents)</sub>

---

## 7. Services

```bash
systemctl status tek-display tek-voice tek-bluetooth tek-panic
tools/check_boot.sh          # verify boot survival WITHOUT rebooting
```

`tek-panic` is the odd one out: it runs as **root**, has
`DefaultDependencies=no`, and is deliberately *not* ordered after
`tek-display`. It exists to escape the display, so it must come up before it
and survive independently of it — see [§0](#0-getting-out--the-panic-key).

`tek-display` and `tek-voice` **must agree on `XDG_RUNTIME_DIR`** or each looks
for the socket in a different place and they silently never connect.

The display must never stop, and seven failure modes are handled explicitly —
startup gap, per-frame exceptions, systemd's start limit, the console blanker, a
wedged model, blanking on exit, and a camera that has not enumerated yet. See
`TEKDROMO.md` §5.

### The speaker disconnecting is usually PulseAudio, not Bluetooth

If the speaker drops and has to be woken by hand, check this before touching
anything in BlueZ:

```bash
pulseaudio --dump-conf | grep exit-idle
journalctl --since today | grep -oE 'pulseaudio\[[0-9]+\]' | sort -u | wc -l
```

PulseAudio's default `exit-idle-time` is **20 seconds**: with no client
connected it shuts down, and the next client autospawns a fresh daemon. Every
one of those restarts tears down the A2DP link. **96 distinct PulseAudio
processes were logged in a single day** while this was tracked down.

It had run an entire night without a single drop, which is what made it look
like a regression in something else. The reason is that the display holds one
permanent recorder on the sink monitor, so the daemon was never idle — a
fragile thing to rest a speaker connection on. Anything that briefly closes
every stream (restarting a service, probing a device) opens a 20-second window
in which the whole audio stack quietly dies.

`exit-idle-time = -1` in `/etc/pulse/daemon.conf` (backup kept alongside it).
Verified: the full test suite now runs start to finish without the daemon's PID
changing. `tools/check_boot.sh` check 10b guards it.

### Keeping the Bluetooth speaker awake

There is a way the audio dies that is not on the Nano at all: **the speaker has
its own idle timer** and powers itself off when what it receives is digital
silence. Unloading `module-suspend-on-idle` keeps PulseAudio's *stream* open,
which is necessary but not sufficient — something has to actually be played.

```bash
tek keepalive                 # interval, tone, how many sent, idle time
tek keepalive --every 300     # less often
tek keepalive --every 0       # disable
tek keepalive --now           # send one immediately
```

After 120 s of genuine silence the voice service plays a **0.6 s tone at 40 Hz**,
faded in and out. Speech resets the timer and it is skipped while speaking, so a
talkative evening sends none at all.

All four parameters are tunable and persisted, because the right values depend
on the specific speaker and only listening settles them:

```bash
tek keepalive --hz 200 --amp 0.02 --secs 0.25 --every 90
```

**The first attempt did not work, and the reason is worth keeping.** It used
40 Hz, chosen *because* a portable driver cannot reproduce it — which is
self-defeating. A speaker's auto-off detector works on the same post-filter
signal path as its amplifier, so **a tone it cannot reproduce is a tone it
cannot detect**. "Inaudible because unreproducible" and "invisible to the
silence detector" are the same property. It sent 34 tones over three hours and
the speaker switched off anyway.

Ultrasonic is the other intuitive answer and is also wrong here, for a
different reason: children hear well past 18 kHz, so a tone the adults cannot
hear could quietly irritate the kids all day.

So the tone must sit **inside** the range the speaker really plays, and be kept
quiet and brief instead. It is still faded in and out — a waveform starting
mid-cycle is a step discontinuity, and a step contains every frequency, so even
an unobtrusive tone would announce itself with a click.

<sub>[↑ Contents](#contents)</sub>

---

## 8. Measured results

### Where the time and memory actually go

Profiled rather than guessed, with the house empty:

| | cost |
|---|---|
| `phosphor.render_bgra` | **21.75 ms** — 92% of the frame |
| `geometry.build_pts_culled` (1891 edges) | 1.22 ms |
| all four HUD panels together | 0.65 ms |
| `face.update` (the rig) | 0.10 ms |
| Haar detect / landmark fit / LBPH predict | 22.2 / 11.9 / 7.9 ms, at ≤6.7 Hz |

So the renderer *is* the display, and it is already at its floor — its
docstring records four optimisations that were measured and did not help,
including CUDA at 39.1 ms against 21.75 on the CPU. It is memory-bandwidth
bound at 1024×600, and OpenCV's NEON paths are the limit.

OpenCV threading was checked too: only Haar detect benefits (22.2 ms at four
threads against 37.8 at one), so cutting the pool would cost more than the
idle TBB workers do.

Steady state, whole system:

| | |
|---|---|
| `tek-display` | 100% of one core, 238 MB, 28.9 fps |
| `tek-voice` (idle, listening) | **~0% of a core**, 422 MB |
| `tek-panic` / `tek-bluetooth` | ~0%, 4 MB / 1 MB |
| load average | 2.6 of 4 cores · 65 °C |

The ear costs essentially nothing while nobody is talking, which is what the
VAD gate is for. Memory is the binding constraint, as predicted at the start:
665 MB of product against 1971 MB total.

Nothing in this table is an estimate.

| | |
|---|---|
| Render | 4.6 → **45 fps** across the optimisation pass |
| Steady state | **29.0–29.8 fps**, 0 errors, with voice running |
| Time to first frame | 9.47 s → **1.24 s** warm, 4.79 s cold |
| Face reconstruction | silhouette **0.996**, landmark error **3.8%** |
| Expression rig | **0.27 ms/frame** warm |
| Piper synthesis | **0.72× real-time** |
| Recognition (free) | **0.17× real-time** |
| Wake spotting | **0.11× real-time** — 11% of one core, always on |
| Word accuracy | **100%** on synthetic test phrases |
| Lip-sync | 5.92 s of mouth for 5.92 s of audio, paced at 20 ms |
| Bluetooth recovery | forced disconnect → reconnected in **~10 s** |
| Reclaimed | a full core, by disabling an idle-spinning BBS daemon |

Things that were measured and did **not** help, recorded so nobody repeats them:

| Idea | Result |
|---|---|
| CUDA bloom at 512×300 | 29 ms vs **6.8 ms** CPU pyramid — kernel launch dominates |
| numpy fancy-index instead of `cv2.LUT` | 25.1 ms vs **7.1 ms** — 3.5× worse |
| CUDA composite | 39.1 ms, 12.4 ms of it in upload/download alone |
| Box blur instead of gaussian | no change |

<sub>[↑ Contents](#contents)</sub>

---

## 9. The camera can prompt me

The camera does not just steer the head — it can **start a conversation**.

```
camera sees something  ->  debounce  ->  cooldown  ->  a model looks at the
                                                       frame and decides
                                                            |
                                    speaks  <---------------+---> stays silent
```

Silence is a first-class outcome, not a failure. A face that comments on every
arrival is unbearable within a day.

```bash
tek watch                 # is it on, what is the cooldown, how many events
tek watch off             # stop it acting on camera events
tek watch --cooldown 600  # be less talkative
tek look                  # look right now and decide (manual trigger)
tek look --force          # ignore the cooldown
```

### Why recognition broke on a new camera

It started saying UNKNOWN after the webcam was swapped. Leave-one-out on the
existing gallery, threshold 62:

```
median 47.4   p90 61.1   max 62.7   over threshold 1/12
```

That is the *same person* on the *same camera* already failing one sample in
twelve. It was living on the edge and nothing measured it.

`tools/face_diag.py` put those faces through what a different camera does:

| what changed | cost in LBPH distance |
|---|---|
| gamma, contrast | ~2–9 — nothing, the pipeline equalises |
| resolution loss | 54–59 |
| **an 8-pixel shift** | **78–84** |

Never lighting. **Geometry** — and the crop fed to LBPH was the raw Haar
rectangle, which jitters every frame and sits differently on a different lens.

Two fixes, both measured before being written:

* **Eye alignment.** The 68 landmarks already fitted for the HUD face panel
  make this free. A 10px shift goes 98.1 → 36.6; the real gallery's worst case
  goes 62.7 → 52.0, from *above* the threshold to comfortably below.
* **Training augmentation.** Lower-detail copies of each enrolled face, because
  the new camera is wider-angle and puts fewer pixels on a face. Third
  resolution + blur goes 61.0 → 41.4.

Recognition is also **throttled to 2 Hz and voted over a 4-second window**. A
single stray frame no longer wipes the label — which is what "it says unknown"
looks like from the sofa even when most frames are right.

`tools/face_realign.py` migrated the existing gallery, measuring before and
after and refusing to write if alignment had not helped.

**Still worth doing:** re-enrol on the new camera (`tek face enrol JOSH`).
Alignment and augmentation close most of the gap; samples actually taken
through this lens would close the rest.

### Who it thinks people are

Identity comes from **plain English**, not a face-recognition model. Describe
the household in `~/.config/tekdromo/people.md` and that text is handed to the
model along with the picture. No training step, no embeddings, no enrolment —
and someone undescribed is simply not greeted by name.

### Three gates before anything costs money

Every event that gets through is a model call, so:

| Gate | Where | Why |
|---|---|---|
| **Debounce** (2 s) | display | a detector glitch is not an arrival |
| **Cooldown** (180 s) | voice service | so `tek watch off` works without restarting the display |
| **Departures ignored** | voice service | announcing that someone left, to an empty room, is talking to nobody |

### Things that were wrong at first

* **`claude` is not on systemd's PATH.** It lives in `~/.local/bin`, so the
  subprocess never started. The brain caught the `OSError` and returned "no
  comment", which meant a **crash was indistinguishable from a thoughtful
  silence** — the log said `stayed quiet (0.0s)` and the `0.0` was the only
  clue. Failures are now logged loudly and the CLI path is absolute.
* **The brain ran in the project directory**, so it picked up `CLAUDE.md`,
  learned it had a voice, and started trying to run commands it had no
  permission for — turning a 10 s judgement call into 59 s of nothing. It now
  runs in a neutral directory with `--allowed-tools Read`.
* **`--allowed-tools` before the prompt** makes the CLI report the prompt as
  missing. Order matters.

Decision latency is **~10 s**. That is why the trigger fires on arrival rather
than waiting for someone to settle, and why `--brain-model` exists.

<sub>[↑ Contents](#contents)</sub>

---

## 10. What is next

* **The conversation loop** — wake word → transcribe → the same Brain → speak.
  The camera path already proves the shape; a microphone is just another event
  source feeding the same pipeline.
* **A microphone.** Everything above the mic is built and tested; far-field
  pickup in a real room is the open question, and it matters more than model
  choice.
* **Multi-speaker Piper models** — a `speaker_id` input away from ~2,000 more
  voices.
* **Viseme timeline** rather than a per-utterance rounding average, using the
  model's duration predictor.

<sub>[↑ Contents](#contents)</sub>

---

<sub>Deeper engineering notes, including the reasoning behind the face geometry
and the full optimisation history, are in [TEKDROMO.md](TEKDROMO.md).</sub>
