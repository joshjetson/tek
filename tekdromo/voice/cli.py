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
import socket
import sys

from . import bus


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

    sub.add_parser("status", help="service state")
    sub.add_parser("voices", help="which voices work here")
    p = sub.add_parser("listen", help="print mouth frames as they are published")
    p.add_argument("--seconds", type=float, default=0)

    a = ap.parse_args(argv)

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
                       "wait": not a.no_wait})
        c.close()
        if not r or not r.get("ok"):
            sys.stderr.write("failed: %s\n" % (r or {}).get("error", "no reply"))
            return 1
        if r.get("duration"):
            print("spoke %.2fs (%s, synth %.2fs)"
                  % (r["duration"], r.get("voice", "?"), r.get("synth", 0)))
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
