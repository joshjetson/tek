"""
Display-side link to the voice service.

The display and the voice service are separate processes because speech is
bursty and CPU-hungry and the render loop must never stall. This is the display
end of that seam, and it is deliberately tiny: it subscribes, keeps the latest
mouth frame in an attribute, and nothing in the frame loop ever blocks on it.

The mouth values it receives are computed from the SAME PCM that is being
played, so the face cannot drift out of step with the audio - lip-sync is a
property of the topology rather than something kept in agreement by hand.

Connection is retried forever. The voice service may start later than the
display, be restarted, or not exist at all; none of those are allowed to affect
rendering, so a failure here costs exactly one unused thread.
"""
import threading
import time

from .voice import bus


class MouthLink(object):
    """Latest (openness, rounding) from the voice service, or (0, 0).

        link = MouthLink().start()
        openness, rounding = link.mouth()
    """

    # If the voice service dies mid-utterance the last frame would otherwise
    # stick, leaving the mouth hanging open. Anything older than this is
    # treated as silence.
    STALE = 0.35

    def __init__(self, path=bus.DEFAULT_PATH, retry=3.0):
        self.path = path
        self.retry = retry
        self.running = True
        self.connected = False
        self.speaking = False
        # Radio protocol: is the channel open. Drives the third eye.
        self.channel_open = False
        # The expression to wear while this utterance is being said. Cleared
        # when speech ends so the face returns to whatever the display's own
        # state machine wants, rather than holding the last mood forever.
        self.mood = None
        self._mouth = (0.0, 0.0)
        self._at = 0.0

    def start(self):
        t = threading.Thread(target=self._loop)
        t.daemon = True
        t.start()
        return self

    def _resync(self):
        """Ask for state that is published only when it CHANGES.

        Subscribing gets you every future message and nothing about the
        present, so a display that connects after the channel opened never
        learns it is open - and the third eye stays down through a whole
        conversation. The star not appearing was exactly this: the service
        said channel_open True and the display had simply never been told,
        because the one publish happened while it was restarting.

        A separate short-lived connection, because the subscribed one is a
        stream and cannot carry a request. Failure is ignored: the worst case
        is the state we already had.
        """
        try:
            c = bus.Client(self.path, timeout=5.0)
            st = c.request({"cmd": "status"}) or {}
            c.close()
            if "channel_open" in st:
                self.channel_open = bool(st["channel_open"])
            if "speaking" in st:
                self.speaking = bool(st["speaking"])
        except Exception:
            pass

    def _loop(self):
        while self.running:
            client = None
            try:
                client = bus.Client(self.path)
                client.subscribe()
                self.connected = True
                self._resync()
                for msg in client:
                    if not self.running:
                        break
                    if "mouth" in msg:
                        m = msg["mouth"]
                        self._mouth = (float(m[0]), float(m[1]))
                        self._at = time.time()
                    if "channel" in msg:
                        self.channel_open = bool(msg["channel"])
                    if "speaking" in msg:
                        self.speaking = bool(msg["speaking"])
                        if self.speaking:
                            self.mood = msg.get("mood")
                        else:
                            self._mouth = (0.0, 0.0)
                            self.mood = None
            except Exception:
                pass                       # service down: retry, never raise
            finally:
                self.connected = False
                self.speaking = False
                self.mood = None
                self.channel_open = False
                self._mouth = (0.0, 0.0)
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
            if self.running:
                time.sleep(self.retry)

    def mouth(self):
        """(openness, rounding). Never blocks, never raises."""
        if time.time() - self._at > self.STALE:
            return 0.0, 0.0
        return self._mouth

    def stop(self):
        self.running = False
