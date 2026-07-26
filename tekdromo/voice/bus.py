"""
Newline-delimited JSON over a Unix socket. Shared by both ends.

The display renders at 30 fps and must never stall, so speech - which is
bursty, CPU-hungry and occasionally blocks on hardware - runs in a separate
process. This is the seam between them, and it is deliberately the smallest
thing that works: no broker, no dependency, no serialisation format anyone has
to learn.

One server (the voice service) with two kinds of client:

  * `tek say` connects, sends one command, reads one reply, exits.
  * The display connects and subscribes, then receives mouth frames for as
    long as it stays connected.

A dropped subscriber must never affect speech, and a dead display must never
be able to block the voice service - so writes to subscribers are best-effort
and a failing one is simply removed.
"""
import errno
import json
import os
import socket
import threading

DEFAULT_PATH = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "tekdromo-voice.sock")


def _send_line(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


class _Lines(object):
    """Reassembles newline-delimited JSON from a stream.

    A socket read does not respect message boundaries: one recv can carry half
    a message or three of them. Getting this wrong produces intermittent JSON
    errors under load and looks like a protocol bug.
    """

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def __iter__(self):
        while True:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line.decode("utf-8"))
                    except ValueError:
                        pass            # ignore junk, keep the stream alive
            try:
                chunk = self.sock.recv(4096)
            except (socket.error, OSError):
                return
            if not chunk:
                return
            self.buf += chunk


class Server(object):
    """Accepts clients and dispatches messages to `handler(msg, conn)`.

    handler returns a dict to reply with, or None for no reply.
    """

    def __init__(self, path=DEFAULT_PATH, handler=None):
        self.path = path
        self.handler = handler or (lambda msg, conn: None)
        self.subscribers = []
        self._lock = threading.Lock()
        self.running = False
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        # A socket file left behind by a crash makes bind() fail with
        # EADDRINUSE forever. Removing a stale one is safe; a live server would
        # have failed to bind anyway.
        try:
            os.unlink(path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(8)
        os.chmod(path, 0o600)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._accept)
        t.daemon = True
        t.start()
        return self

    def _accept(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
            except (socket.error, OSError):
                if not self.running:
                    return
                continue
            t = threading.Thread(target=self._serve, args=(conn,))
            t.daemon = True
            t.start()

    def _serve(self, conn):
        try:
            for msg in _Lines(conn):
                if msg.get("cmd") == "subscribe":
                    with self._lock:
                        self.subscribers.append(conn)
                    _send_line(conn, {"ok": True, "subscribed": True})
                    continue
                reply = self.handler(msg, conn)
                if reply is not None:
                    _send_line(conn, reply)
        except Exception:
            pass
        finally:
            with self._lock:
                if conn in self.subscribers:
                    self.subscribers.remove(conn)
            try:
                conn.close()
            except Exception:
                pass

    def publish(self, obj):
        """Best-effort fan-out. Never raises, never blocks on a dead client."""
        with self._lock:
            targets = list(self.subscribers)
        for c in targets:
            try:
                _send_line(c, obj)
            except Exception:
                with self._lock:
                    if c in self.subscribers:
                        self.subscribers.remove(c)

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass


class Client(object):
    """Connects to the voice service. Used by the CLI and by the display."""

    def __init__(self, path=DEFAULT_PATH, timeout=None):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if timeout:
            self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.lines = _Lines(self.sock)

    def send(self, obj):
        _send_line(self.sock, obj)

    def recv(self):
        for msg in self.lines:
            return msg
        return None

    def request(self, obj):
        self.send(obj)
        return self.recv()

    def subscribe(self):
        self.send({"cmd": "subscribe"})
        return self.recv()

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
