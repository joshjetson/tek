# -*- coding: utf-8 -*-
"""The socket protocol between the voice service and the display.

The display must never be able to stall the voice service, and the voice
service must never be able to stall the display. Both directions are tested,
because the failure mode of getting this wrong is the one thing this project
does not tolerate: a render loop that stops.
"""
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tekdromo.voice import bus

FAIL = []


def check(name, cond, extra=""):
    print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(extra)))
    if not cond:
        FAIL.append(name)


path = os.path.join(tempfile.mkdtemp(prefix="tekbus"), "v.sock")
seen = []


def handler(msg, conn):
    seen.append(msg)
    if msg.get("cmd") == "slow":
        time.sleep(0.3)
    return {"ok": True, "echo": msg.get("cmd")}


srv = bus.Server(path, handler).start()

# -- basics ----------------------------------------------------------------
c = bus.Client(path, timeout=5)
check("request/reply round trips", c.request({"cmd": "ping"}).get("echo") == "ping")
check("unicode survives the wire",
      c.request({"cmd": u"café — ü"}).get("echo") == u"café — ü")
c.close()

# -- stale socket file -----------------------------------------------------
# A crash leaves the socket file behind and bind() then fails with EADDRINUSE
# forever, so the service never comes back. It must clean up after itself.
srv.close()
open(path, "w").close()
try:
    srv2 = bus.Server(path, handler).start()
    check("a stale socket file does not stop the service starting", True)
except Exception as e:
    check("a stale socket file does not stop the service starting", False, e)
    srv2 = None

if srv2:
    # -- framing ------------------------------------------------------------
    # Two messages in one packet, and one message split across two packets.
    # Both happen under load, and getting either wrong looks like random
    # JSON corruption.
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.connect(path)
    raw.sendall(b'{"cmd":"a"}\n{"cmd":"b"}\n')
    time.sleep(0.4)
    check("two messages in one packet are both handled",
          [m.get("cmd") for m in seen[-2:]] == ["a", "b"], seen[-2:])
    raw.sendall(b'{"cmd":"spl')
    time.sleep(0.2)
    raw.sendall(b'it"}\n')
    time.sleep(0.4)
    check("a message split across packets is reassembled",
          seen[-1].get("cmd") == "split", seen[-1])
    raw.sendall(b'not json at all\n{"cmd":"after"}\n')
    time.sleep(0.4)
    check("malformed input is skipped without killing the stream",
          seen[-1].get("cmd") == "after", seen[-1])
    raw.close()

    # -- publish / subscribe ------------------------------------------------
    sub = bus.Client(path, timeout=5)
    sub.subscribe()
    got = []

    def drain():
        for m in sub:
            got.append(m)

    t = threading.Thread(target=drain)
    t.daemon = True
    t.start()
    time.sleep(0.3)
    for i in range(5):
        srv2.publish({"mouth": [i / 10.0, 0.0]})
    time.sleep(0.6)
    check("subscribers receive published frames", len(got) >= 5, len(got))
    check("frames arrive in order and intact",
          got[0]["mouth"][0] == 0.0 and got[4]["mouth"][0] == 0.4, got[:5])

    # -- a dead subscriber must not break publishing ------------------------
    sub.close()
    time.sleep(0.3)
    try:
        for i in range(20):
            srv2.publish({"mouth": [0.5, 0.0]})
        check("publishing to a dead subscriber does not raise", True)
    except Exception as e:
        check("publishing to a dead subscriber does not raise", False, e)
    time.sleep(0.3)
    check("the dead subscriber is dropped from the list",
          len(srv2.subscribers) == 0, len(srv2.subscribers))

    # -- a slow client must not block another one ---------------------------
    done = []

    def slow():
        cc = bus.Client(path, timeout=10)
        cc.request({"cmd": "slow"})
        done.append("slow")
        cc.close()

    ts = threading.Thread(target=slow)
    ts.daemon = True
    ts.start()
    time.sleep(0.05)
    t0 = time.time()
    cf = bus.Client(path, timeout=5)
    cf.request({"cmd": "fast"})
    fast_ms = (time.time() - t0) * 1000
    cf.close()
    check("a slow client does not block a fast one", fast_ms < 200,
          "%.0f ms" % fast_ms)

    srv2.close()

# -- server gone -----------------------------------------------------------
try:
    bus.Client(path, timeout=2)
    check("connecting to a stopped service fails cleanly", False, "connected?")
except (socket.error, OSError):
    check("connecting to a stopped service fails cleanly", True)

print("VOICE BUS " + ("OK" if not FAIL else "FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
