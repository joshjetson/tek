"""
`tek` - the mechanism that gives Claude a voice.

    tek say "hello"        speak through the service (face mouths it)
    tek status             is the service up, which voice, has it spoken
    tek voices             which voices actually work on this machine
    tek listen             print mouth frames - proves the display feed

Everything goes through the voice service, which is the only owner of a Voice.
The CLI has no synthesis and no fallback logic of its own: two places deciding
"how do we speak" is exactly how they drift apart.
"""
import argparse
import os
import socket
import sys

VOICE_ENGINES = ("piper", "pico", "flite", "espeak")

from . import bus


def _normalise(name):
    """Accept a bare Piper model name where a voice name is expected.

    "en_US-kusal-medium" and "piper:en_US-kusal-medium" mean the same thing to
    a person; only one of them used to work, and only in one subcommand. One
    helper, used everywhere a voice can be named.
    """
    n = (name or "").strip()
    if not n or ":" in n:
        return n
    if n in VOICE_ENGINES:
        return n
    return "piper:" + n if "-" in n else n


def _spoken(name):
    """A voice name a voice can actually read out."""
    n = name.split(":", 1)[-1]
    for lang, word in (("en_US-", ""), ("en_GB-", "British ")):
        if n.startswith(lang):
            n = word + n[len(lang):]
            break
    for q in ("-medium", "-high", "-low"):
        if n.endswith(q):
            n = n[:-len(q)]
            break
    return n.replace("_", " ")


def _client(path, timeout=120.0):
    try:
        return bus.Client(path, timeout=timeout)
    except (socket.error, OSError) as e:
        sys.stderr.write(
            "cannot reach the voice service at %s (%s)\n"
            "  start it with:  systemctl --user start tek-voice\n"
            "  or standalone:  python3 -m tekdromo.voice.service\n" % (path, e))
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tek")
    ap.add_argument("--socket", default=bus.DEFAULT_PATH)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("say", help="speak some text")
    p.add_argument("text", nargs="+")
    p.add_argument("--no-wait", action="store_true",
                   help="return immediately instead of waiting for the end")
    p.add_argument("--voice", default=None,
                   help="speak this once in another voice, without changing "
                        "the default")

    p = sub.add_parser("audition",
                       help="hear every voice say the same line, out loud")
    p.add_argument("text", nargs="*",
                   help="what each voice should say (default: a set line)")
    p.add_argument("--all", action="store_true",
                   help="every installed voice, Piper models and native "
                        "engines alike")
    p.add_argument("--piper", action="store_true",
                   help="only the Piper models. This is usually what you want "
                        "when choosing a voice - pico/flite/espeak are "
                        "fallbacks, not candidates.")
    p.add_argument("--voices", default=None,
                   help="comma-separated list to audition exactly, e.g. "
                        "en_US-amy-medium,en_GB-alan-medium")

    p = sub.add_parser("voice", help="set the default voice, permanently")
    p.add_argument("name", help="piper|pico|flite|espeak")

    p = sub.add_parser("watch",
                       help="camera-triggered speech: on, off, or status")
    p.add_argument("state", nargs="?", choices=["on", "off"], default=None)
    p.add_argument("--cooldown", type=float, default=None,
                   help="minimum seconds between acting on events")

    p = sub.add_parser("look",
                       help="look at the camera NOW and decide whether to speak")
    p.add_argument("--force", action="store_true",
                   help="ignore the cooldown")

    p = sub.add_parser("keepalive",
                       help="inaudible tone that stops the speaker sleeping")
    p.add_argument("--every", type=float, default=None,
                   help="seconds of silence before nudging it; 0 disables")
    p.add_argument("--hz", type=float, default=None,
                   help="tone frequency. Must be something the speaker can "
                        "actually reproduce - a tone it cannot play is also a "
                        "tone its silence detector cannot see.")
    p.add_argument("--amp", type=float, default=None,
                   help="amplitude 0..1. Lower is quieter; too low and the "
                        "speaker treats it as silence and sleeps anyway.")
    p.add_argument("--secs", type=float, default=None, help="tone length")
    p.add_argument("--now", action="store_true", help="send one immediately")

    p = sub.add_parser("face", help="who the camera recognises")
    p.add_argument("action", choices=["list", "enrol", "enroll", "forget"])
    p.add_argument("name", nargs="?")
    p.add_argument("--samples", type=int, default=10)

    p = sub.add_parser("ears", help="continuous listening: state, on, or off")
    p.add_argument("state", nargs="?", choices=["on", "off"], default=None)

    p = sub.add_parser("panic",
                       help="stop the display and hand the console back")
    p.add_argument("what", nargs="?", choices=["stop", "quiet"], default="stop",
                   help="stop: just the display. quiet: the voice as well.")

    sub.add_parser("status", help="service state")
    sub.add_parser("voices", help="which voices work here")
    p = sub.add_parser("listen", help="print mouth frames as they are published")
    p.add_argument("--seconds", type=float, default=0)

    a = ap.parse_args(argv)

    if a.cmd == "panic":
        # The same action the ESC chord performs, for when you still have a
        # shell. Deliberately routed through tekdromo.panic rather than calling
        # systemctl here: stopping the display is only half of it - the panel
        # keeps the last frame, so the console has to be forced to repaint.
        import subprocess
        from .. import panic as _panic
        if os.geteuid() != 0:
            return subprocess.call(["sudo", sys.executable, _panic.__file__,
                                    "--now", a.what])
        print("stopped %s" % ", ".join(_panic.panic(a.what)))
        return 0

    if a.cmd == "voices":
        from . import tts
        for name, ok, note in tts.available():
            print("  %-7s %-5s %s" % (name, "OK" if ok else "no", note))
        return 0

    if a.cmd == "say":
        c = _client(a.socket)
        if c is None:
            return 1
        r = c.request({"cmd": "say", "text": " ".join(a.text),
                       "wait": not a.no_wait,
                       "voice": _normalise(a.voice) if a.voice else None})
        c.close()
        if not r or not r.get("ok"):
            sys.stderr.write("failed: %s\n" % (r or {}).get("error", "no reply"))
            return 1
        if r.get("duration"):
            print("spoke %.2fs (%s, synth %.2fs)"
                  % (r["duration"], r.get("voice", "?"), r.get("synth", 0)))
        return 0

    if a.cmd == "audition":
        # Through the real speaker, in the real room. Judging a voice from a
        # waveform or a laptop is not the same test - this one has to sound
        # right on THAT speaker, at that distance.
        from . import tts
        line = " ".join(a.text) if a.text else (
            "This is voice number %d. I would sound like this every day, "
            "reading you the weather, or answering a question from across "
            "the room.")
        c = _client(a.socket, timeout=300.0)
        if c is None:
            return 1
        if a.voices:
            names = [_normalise(n) for n in a.voices.split(",") if n.strip()]
        elif a.piper:
            names = ["piper:" + m for m in tts.piper_models()]
        elif a.all:
            names = tts.all_names()
        else:
            names = tts.ORDER
        for i, name in enumerate(names, 1):
            said = line % i if "%d" in line else line
            print("  %d/%d  %s" % (i, len(names), name))
            # Speak a human label, not the model id: "piper:en_GB-alan-medium"
            # comes out as "piper colon en underscore G B alan medium", which
            # is unusable when the whole point is to judge how it sounds.
            c.request({"cmd": "say", "text": "Voice %d. %s." % (i, _spoken(name)),
                       "voice": name})
            r = c.request({"cmd": "say", "text": said, "voice": name})
            if not r or not r.get("ok"):
                print("       unavailable: %s" % (r or {}).get("error"))
        c.close()
        print("\n  pick one with:  tek voice <name>")
        return 0

    if a.cmd == "voice":
        c = _client(a.socket, timeout=120.0)
        if c is None:
            return 1
        r = c.request({"cmd": "set_voice", "voice": _normalise(a.name)})
        if r and r.get("ok"):
            print("default voice is now %s (%d Hz), saved across restarts"
                  % (r["voice"], r["rate"]))
            c.request({"cmd": "say",
                       "text": "This is now my voice.", "voice": r["voice"]})
            c.close()
            return 0
        sys.stderr.write("failed: %s\n" % (r or {}).get("error", "no reply"))
        c.close()
        return 1

    if a.cmd == "ears":
        c = _client(a.socket, timeout=30.0)   # switching on loads Vosk
        if c is None:
            return 1
        msg = {"cmd": "ears"}
        if a.state:
            msg["on"] = (a.state == "on")
        r = c.request(msg) or {}
        c.close()
        print("  listening    %s" % r.get("listening"))
        print("  wake words   %s" % ", ".join(r.get("wake_words") or ["?"]))
        if "utterances" in r:
            print("  utterances   %s heard, %s were the wake word, "
                  "%s became commands"
                  % (r.get("utterances"), r.get("wakes"), r.get("commands")))
            print("  microphone   %s" % (r.get("device") or "?"))
            print("  mic opens    %s" % r.get("opens"))
            print("  last heard   %r" % (r.get("last_heard"),))
            for m in (r.get("misses") or []):
                print("  near miss    %-16r %4.1fs  peak %.3f%s"
                      % (m.get("got"), m.get("secs", 0), m.get("peak", 0),
                         "   <- too quiet" if m.get("peak", 1) < 0.05 else ""))
            if r.get("armed"):
                print("  ARMED - waiting for a command right now")
        elif r.get("listening"):
            print("  (the ear is starting up)")
        return 0

    if a.cmd == "watch":
        c = _client(a.socket, timeout=15.0)
        if c is None:
            return 1
        msg = {"cmd": "watch"}
        if a.state:
            msg["on"] = (a.state == "on")
        if a.cooldown is not None:
            msg["cooldown"] = a.cooldown
        r = c.request(msg) or {}
        c.close()
        print("  watching   %s" % r.get("watching"))
        print("  cooldown   %.0fs" % (r.get("cooldown") or 0))
        print("  brain      %s" % r.get("brain"))
        print("  events     %s seen, %s acted on" % (r.get("seen"), r.get("acted")))
        if r.get("next_in"):
            print("  next event accepted in %.0fs" % r["next_in"])
        return 0

    if a.cmd == "look":
        # A manual trigger, so the whole perception path can be exercised
        # without waiting for someone to walk past.
        c = _client(a.socket, timeout=180.0)
        if c is None:
            return 1
        if a.force:
            c.request({"cmd": "watch", "cooldown": 0})
        r = c.request({"cmd": "event", "event": {
            "kind": "manual", "faces": 1,
            "what": "you were asked to look at the camera right now",
            "image": os.path.expanduser("~/.cache/tekdromo/seen.jpg")}}) or {}
        c.close()
        if r.get("acted"):
            print("  looking... (it will speak only if it decides to)")
        else:
            print("  did not look: %s" % r.get("reason"))
        return 0

    if a.cmd == "keepalive":
        c = _client(a.socket, timeout=30.0)
        if c is None:
            return 1
        msg = {"cmd": "keepalive"}
        for k in ("every", "hz", "amp", "secs"):
            v = getattr(a, k)
            if v is not None:
                msg[k] = v
        if a.now:
            msg["now"] = True
        r = c.request(msg) or {}
        c.close()
        print("  every      %s" % ("off" if not r.get("every")
                                   else "%.0fs of silence" % r["every"]))
        print("  tone       %.0f Hz, %.0f ms, %.1f%% amplitude"
              % (r.get("hz", 0), (r.get("secs") or 0) * 1000,
                 (r.get("amp") or 0) * 100))
        print("  sent       %s" % r.get("sent"))
        print("  idle for   %.0fs" % (r.get("idle") or 0))
        return 0

    if a.cmd == "face":
        import time as _time
        from .. import recog
        if a.action == "list":
            ppl = recog.people()
            if not ppl:
                print("  nobody enrolled - every face shows as UNKNOWN")
                return 0
            reg = recog.registry()
            print("  %-12s %-8s %-17s %-17s %s"
                  % ("NAME", "SAMPLES", "ENROLLED", "LAST SEEN", "TIMES"))
            for n in ppl:
                e = reg.get(n, {})
                print("  %-12s %-8d %-17s %-17s %s"
                      % (n, len(recog.samples(n)), e.get("enrolled", "?"),
                         e.get("last_seen", "never"), e.get("seen", 0)))
            return 0
        if not a.name:
            sys.stderr.write("need a name\n")
            return 1
        name = a.name.strip().upper()
        if a.action == "forget":
            recog.forget(name)
            print("  forgot %s - photographs and record both deleted" % name)
            return 0

        crop = os.path.expanduser("~/.cache/tekdromo/crop.png")
        print("  enrolling %s - look at the camera and move your head a "
              "little" % name)
        import cv2
        got, seen, t0 = 0, None, _time.time()
        while got < a.samples and _time.time() - t0 < 90:
            try:
                m = os.path.getmtime(crop)
            except OSError:
                _time.sleep(0.5)
                continue
            # Only take a crop we have not already taken. Without this it saves
            # the same picture ten times and the recogniser learns one pose.
            if m == seen:
                _time.sleep(0.3)
                continue
            seen = m
            g = cv2.imread(crop, cv2.IMREAD_GRAYSCALE)
            if g is None or g.size < 100:
                continue
            recog.save_sample(name, g)
            got += 1
            print("    %d/%d" % (got, a.samples))
        if not got:
            sys.stderr.write("  no face crops appeared - is anyone in front "
                             "of the camera?\n")
            return 1
        recog.note_enrolled(name, len(recog.samples(name)))
        print("  enrolled %s with %d samples; the display picks it up within "
              "a few seconds" % (name, got))
        return 0

    if a.cmd == "status":
        c = _client(a.socket, timeout=5.0)
        if c is None:
            return 1
        r = c.request({"cmd": "status"})
        c.close()
        for k in ("voice", "rate", "speaking", "spoken", "load_time"):
            print("  %-10s %s" % (k, (r or {}).get(k)))
        return 0

    if a.cmd == "listen":
        c = _client(a.socket, timeout=None)
        if c is None:
            return 1
        c.subscribe()
        import time
        t0 = time.time()
        for msg in c:
            if "mouth" in msg:
                o, rd = msg["mouth"]
                print("  open %.3f round %.3f  %s" % (o, rd, "#" * int(o * 50)))
            elif "speaking" in msg:
                print("  speaking=%s %s" % (msg["speaking"], msg.get("text", "")))
            if a.seconds and time.time() - t0 > a.seconds:
                break
        c.close()
        return 0

    ap.print_help()
    return 1
