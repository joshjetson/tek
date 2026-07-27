"""
Panels drawn around the head: clock, date, and whatever comes later.

Everything here emits the SAME thing the head emits - an (N, 2, 2) int32 array
of line segments - so HUD vectors are simply concatenated onto the head's and
the whole frame goes through one `render_bgra` call. There is no second
renderer, no second bloom, no second phosphor LUT, and therefore no way for the
panels to drift out of look with the face. That is the same reason the neck and
the ears became fields instead of separate meshes.

Digits are seven-segment because that is what the period actually used, and
because a segment IS a stroke: nothing has to be approximated to draw it on a
vector display. Letters use a minimal stroke font covering only the handful of
characters the panels need.
"""
import time

import numpy as np

# Seven-segment geometry, in a 0..1 glyph cell.
#
#      a
#    f   b
#      g
#    e   c
#      d
_SEG = {
    "a": ((0.0, 0.0), (1.0, 0.0)),
    "b": ((1.0, 0.0), (1.0, 0.5)),
    "c": ((1.0, 0.5), (1.0, 1.0)),
    "d": ((0.0, 1.0), (1.0, 1.0)),
    "e": ((0.0, 0.5), (0.0, 1.0)),
    "f": ((0.0, 0.0), (0.0, 0.5)),
    "g": ((0.0, 0.5), (1.0, 0.5)),
}
_DIGITS = {
    "0": "abcdef", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}
# "1" is deliberately NOT segments b+c. On real hardware those sit at the right
# edge of the cell, which is authentic and unreadable here: "5:51" rendered as
# "5 5  1" with the stroke pushed hard right and a hole where the digit should
# be. Centring it keeps the cell monospaced - so the clock does not jitter as
# the time changes - while putting the stroke where the eye expects a digit.

# A stroke font for the few non-digits the panels use. Each entry is a list of
# polylines in the same 0..1 cell; kept deliberately tiny rather than pulling in
# a font renderer, which would not look right on a vector display anyway.
_CHARS = {
    ":": [[(0.5, 0.22), (0.5, 0.34)], [(0.5, 0.66), (0.5, 0.78)]],
    "/": [[(0.05, 1.0), (0.95, 0.0)]],
    "1": [[(0.5, 0.0), (0.5, 1.0)]],
    "-": [[(0.1, 0.5), (0.9, 0.5)]],
    ".": [[(0.45, 0.92), (0.55, 0.92)]],
    " ": [],          # narrow blank - stands in for the blinking colon
    "_": [],          # full-width blank - pads a single-digit hour
    "A": [[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)], [(0.18, 0.62), (0.82, 0.62)]],
    "P": [[(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)]],
    "M": [[(0.0, 1.0), (0.0, 0.0), (0.5, 0.55), (1.0, 0.0), (1.0, 1.0)]],
}


def _seg_lines(ch):
    """A character as a list of polylines in the 0..1 cell."""
    if ch in _CHARS:
        return _CHARS[ch]
    if ch in _DIGITS:
        return [[_SEG[s][0], _SEG[s][1]] for s in _DIGITS[ch]]
    return []


def _place(polys, x, y, w, h):
    """Scale 0..1 polylines into a cell and emit 2-point segments."""
    out = []
    for poly in polys:
        pts = [(x + px * w, y + py * h) for px, py in poly]
        for i in range(len(pts) - 1):
            out.append((pts[i], pts[i + 1]))
    return out


def text(s, x, y, cw, ch, gap=None, narrow=0.45):
    """Lay out a string. Returns segments and the width consumed.

    ':' '/' and '.' get a narrower cell than digits, because giving a colon the
    same advance as an 8 leaves a hole in the middle of a clock.
    """
    gap = cw * 0.28 if gap is None else gap
    out, cx = [], float(x)
    for chch in s:
        # TWO kinds of blank, and conflating them made the panel pulse once a
        # second:
        #   " " is a NARROW blank, the same width as the colon it replaces
        #       when the colon blinks off.
        #   "_" is a FULL-WIDTH blank, the pad in front of a single-digit hour,
        #       and must be exactly one digit cell.
        # Using a space for both meant the blink alternated between a narrow
        # cell and a full one, so the box grew and shrank every second.
        cell = cw * narrow if chch in ":./ " else cw
        out.extend(_place(_seg_lines(chch), cx, y, cell, ch))
        cx += cell + gap
    return out, (cx - gap) - x


def box(x, y, w, h, notch=0.0):
    """A rectangle. With notch>0 the corners are cut, which reads as a panel
    bezel on a vector display rather than a plain box."""
    if notch <= 0:
        return [((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))]
    n = notch
    p = [(x + n, y), (x + w - n, y), (x + w, y + n), (x + w, y + h - n),
         (x + w - n, y + h), (x + n, y + h), (x, y + h - n), (x, y + n)]
    return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]


def to_pts(segments):
    """Segments -> the (N, 2, 2) int32 array the renderer already takes."""
    if not segments:
        return np.zeros((0, 2, 2), np.int32)
    return np.rint(np.array(segments, dtype=np.float32)).astype(np.int32)


class Clock(object):
    """A retro clock/date panel.

    Rebuilt only when the displayed text changes - once a second because of the
    blinking colon, not once a frame. At 30 fps that is 29 rebuilds saved out of
    every 30, and the render loop is the one thing on this machine that must
    never be given avoidable work.
    """

    def __init__(self, w, h, margin=26, digit=(26, 44), small=(15, 24)):
        self.w, self.h = w, h
        self.margin = margin
        self.dw, self.dh = digit
        self.sw, self.sh = small
        self._key = None
        self._pts = np.zeros((0, 2, 2), np.int32)

    def strings(self, when=None):
        """(time, meridiem, date). Local time - the machine is on
        America/Chicago, so this is Central, and it follows CDT/CST correctly
        rather than being pinned to one of them."""
        lt = time.localtime(when)
        hour12 = lt.tm_hour % 12 or 12
        # Padded to a fixed width with "_", a full-width blank. Without it the
        # panel is one cell narrower from 1:00 to 9:59 and visibly resizes on
        # the hour, because the box is sized from its contents.
        return ("%s%d:%02d" % ("_" if hour12 < 10 else "", hour12, lt.tm_min),
                "PM" if lt.tm_hour >= 12 else "AM",
                time.strftime("%m/%d/%Y", lt))

    def build(self, when=None):
        hhmm, mer, date = self.strings(when)
        blink = int(when if when is not None else time.time()) % 2 == 0
        shown = hhmm if blink else hhmm.replace(":", " ")

        segs = []
        # Lay the contents out first, then size the box to fit them, so
        # changing the digit size does not require re-tuning the frame.
        tsegs, tw = text(shown, 0, 0, self.dw, self.dh)
        msegs, mw = text(mer, tw + self.dw * 0.95, self.dh * 0.44,
                         self.sw, self.sh)
        dsegs, dw = text(date, 0, self.dh + 20, self.sw, self.sh)

        inner_w = max(tw + self.dw * 0.95 + mw, dw)
        pad_x, pad_y = 18, 14
        bw = inner_w + pad_x * 2
        bh = self.dh + 20 + self.sh + pad_y * 2
        # Upper right, as seen by someone looking at the screen.
        bx = self.w - self.margin - bw
        by = self.margin

        segs.extend(box(bx, by, bw, bh, notch=7))
        # A rule between clock and date; makes it read as one instrument
        # instead of two numbers sharing a border.
        ry = by + pad_y + self.dh + 9
        segs.append(((bx + pad_x, ry), (bx + bw - pad_x, ry)))

        ox, oy = bx + pad_x, by + pad_y
        for group in (tsegs, msegs, dsegs):
            segs.extend((((ax + ox, ay + oy), (bx2 + ox, by2 + oy)))
                        for (ax, ay), (bx2, by2) in group)
        return to_pts(segs), (bx, by, bw, bh)

    def points(self, when=None):
        """Cached per displayed second."""
        now = when if when is not None else time.time()
        hhmm, mer, date = self.strings(now)
        key = (hhmm, mer, date, int(now) % 2)
        if key != self._key:
            self._key = key
            self._pts, self.rect = self.build(now)
        return self._pts


class Scope(object):
    """An oscilloscope trace of whatever is coming out of the speaker.

    Fed from PulseAudio's sink MONITOR rather than from the voice service, so
    it shows *everything the machine plays* - speech, music, anything else -
    with no cooperation from whatever produced it. One source, every case.

    A vector display genuinely is an oscilloscope, so this is about the most
    native thing the panel could show.
    """

    def __init__(self, w, h, margin=26, width=300, height=104, cols=128,
                 window=512):
        self.w, self.h = w, h
        self.bw, self.bh = width, height
        self.bx = w - margin - width
        self.by = h - margin - height
        self.cols = cols
        self.window = window
        self.buf = np.zeros(window * 4, np.float32)
        # Slow-decaying peak. Speech and music differ by tens of dB, so a fixed
        # gain shows either a flat line or a clipped mess; tracking the peak and
        # letting it fall slowly keeps quiet passages visible without the trace
        # jumping about on every transient.
        self.peak = 0.05
        self._frame = None

    def push(self, samples):
        s = np.asarray(samples, dtype=np.float32) / 32767.0
        if not len(s):
            return
        n = min(len(s), len(self.buf))
        self.buf = np.roll(self.buf, -n)
        self.buf[-n:] = s[-n:]
        self.peak = max(float(np.abs(s).max()), self.peak * 0.90)

    def _triggered(self):
        """A window starting at a rising zero crossing.

        Without a trigger the waveform slides sideways every frame and reads as
        noise; with one it stands still, which is what makes a scope legible.

        Vectorised. The obvious Python loop over candidate offsets, together
        with a per-column argmax below, cost 30 ms a frame - the entire budget
        at 30 fps, for a decoration.
        """
        w, half = self.window, self.window // 2
        seg = self.buf[-(w + half):]
        head = seg[:half]
        rising = np.nonzero((head[:-1] <= 0.0) & (head[1:] > 0.0))[0]
        i = int(rising[0]) if len(rising) else 0
        return seg[i:i + w]

    def points(self):
        pad_x, pad_y = 12, 10
        ix, iy = self.bx + pad_x, self.by + pad_y
        iw, ih = self.bw - pad_x * 2, self.bh - pad_y * 2
        mid = iy + ih * 0.5

        # The bezel and the axis never change; building them 30 times a second
        # is pure waste.
        if self._frame is None:
            segs = box(self.bx, self.by, self.bw, self.bh, notch=7)
            for k in range(0, iw, 14):
                segs.append(((ix + k, mid), (ix + min(k + 7, iw), mid)))
            self._frame = to_pts(segs)
        frame = self._frame

        wave = self._triggered()
        per = max(1, self.window // self.cols)
        n = self.cols * per
        # Decimate by bucket PEAK, not by sampling: taking every Nth sample
        # drops the transients that carry the shape, so a drum hit vanishes.
        # Reshape + argmax along an axis does all the buckets in one call.
        block = wave[:n].reshape(self.cols, per)
        idx = np.argmax(np.abs(block), axis=1)
        v = block[np.arange(self.cols), idx] * (1.0 / max(self.peak, 0.02))
        xs = ix + iw * (np.arange(self.cols, dtype=np.float32)
                        / float(self.cols - 1))
        ys = mid - np.clip(v, -1.0, 1.0) * (ih * 0.46)
        pts = np.stack([xs, ys], axis=1)
        trace = np.rint(np.stack([pts[:-1], pts[1:]], axis=1)).astype(np.int32)
        return np.concatenate([frame, trace]) if len(frame) else trace
