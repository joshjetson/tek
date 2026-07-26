"""Assemble the tekdromo package from the existing modules.

Extracts working code by line range rather than retyping it, so behaviour is
preserved exactly. Verified afterwards by the test suite and the live service.
"""
import os
import re

HOME = "/home/super"
PKG = os.path.join(HOME, "tekdromo")
os.makedirs(PKG, exist_ok=True)

src = {f: open(os.path.join(HOME, f)).read().split("\n")
       for f in ("tekvector.py", "tekfb.py", "tekhead.py", "tekfdl.py")}


def grab(f, start_pat, end_pat=None, inclusive_end=False):
    """Lines from the line matching start_pat up to (not incl.) end_pat."""
    lines = src[f]
    i = next(k for k, l in enumerate(lines) if re.match(start_pat, l))
    if end_pat is None:
        return "\n".join(lines[i:])
    j = next(k for k in range(i + 1, len(lines)) if re.match(end_pat, lines[k]))
    if inclusive_end:
        j += 1
    return "\n".join(lines[i:j]).rstrip() + "\n"


def write(name, header, *chunks):
    body = header.rstrip() + "\n\n\n" + "\n\n".join(c.rstrip() for c in chunks)
    open(os.path.join(PKG, name), "w").write(body.rstrip() + "\n")
    n = body.count("\n")
    print("  %-18s %4d lines" % (name, n))


# ---------------------------------------------------------------------------
print("writing package modules:")

write("phosphor.py", '''"""
Storage-tube look: phosphor colour, beam bloom, and the intensity->BGRA LUT.

ONE implementation. This maths previously existed twice - once in tekvector for
still renders and again in tekfb for the live path - which is precisely the kind
of duplication that lets two renderers drift apart.

Authenticity notes that constrain the code:
  * No scanlines. A DVST does not scan; adding them is the classic fake tell.
  * Constant beam intensity - stored charge is binary-ish.
  * Bloom is a CPU pyramid, NOT CUDA. Measured at 1024x600: CUDA 29ms,
    CPU pyramid 6.8ms. Kernel-launch overhead dominates at this size.
"""
import cv2
import numpy as np

# The 4010's storage tube read as a slightly yellowed green, not pure #00FF00.
PHOSPHOR_BGR = np.array([0.30, 1.00, 0.45], dtype=np.float32)
SCREEN_TINT = np.array([0.010, 0.028, 0.014], dtype=np.float32)
MAX_I = 2.0                      # intensity range packed into the 0..255 LUT
''',
      grab("tekfb.py", r"^def build_statics", r"^def render_bgra"),
      grab("tekfb.py", r"^def render_bgra", r"^def erase_bgra"),
      grab("tekfb.py", r"^def erase_bgra", r"^def main"))

write("framebuffer.py", '''"""
/dev/fb0 plumbing: geometry query, mmap, and keeping the panel awake.

The panel is BGRA with no row padding, which is byte-identical to OpenCV's
layout - frames go in with no conversion at all.
"""
import fcntl
import mmap
import os
import struct

import numpy as np

FBIOBLANK = 0x4611
FB_BLANK_UNBLANK = 0
''',
      grab("tekfb.py", r"^def unblank", r"^def _bye"),
      grab("tekfb.py", r"^def fb_info", r"^def build_statics"),
      '''def open_screen(dev="/dev/fb0"):
    """Returns (fd, mmap, ndarray view, w, h). The array writes straight to the
    panel."""
    w, h, bpp, stride = fb_info(dev)
    if bpp != 32 or stride != w * 4:
        raise RuntimeError("unexpected framebuffer: %dbpp stride=%d" % (bpp, stride))
    fd = os.open(dev, os.O_RDWR)
    mm = mmap.mmap(fd, h * stride, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    screen = np.frombuffer(mm, dtype=np.uint8).reshape(h, w, 4)
    return fd, mm, screen, w, h''')

write("geometry.py", '''"""
Transform, perspective projection, and back-face culling.

Extracted from the old tekhead.py, which was 338 lines of which the runtime
used 17 - the rest was face_1's head model and a mannequin, both superseded.
The name was wrong too: this is projection, not a head.
"""
import numpy as np
''',
      grab("tekvector.py", r"^def rotate", r"^def project"),
      grab("tekvector.py", r"^def project", r"^# ---"),
      grab("tekhead.py", r"^def build_pts_culled", r"^def _register"))

print("done")


# --- split the god file -----------------------------------------------------
print("\nsplitting tekfdl.py (685 lines, 9 responsibilities):")

write("anatomy.py", '''"""
The head's measured shape and the FDL region constants.

Pure data plus the field primitives that evaluate it. No contouring, no
rendering - those live in contour.py and phosphor.py.

Sources for the numbers (see docs/TEKDROMO.md):
  * Loomis: a head is a sphere with the sides cut FLAT. An ellipse cannot be a
    head, so the silhouette is an explicit landmark profile.
  * Widest point is the zygomatic arch, below the eye line - not the cranium.
  * Five eyes wide; mouth two eyes wide; vertical thirds.
  * The jaw turns at the gonial angle: a slope break, not a smooth taper.
  * Ears span brow line to nose base, tilted 18 degrees.
"""
import math

import numpy as np
''',
      grab("tekfdl.py", r"^_SIL_Y", r"^def sil_w"),
      grab("tekfdl.py", r"^def sil_w", r"^FOREHEAD"),
      grab("tekfdl.py", r"^FOREHEAD", r"^def _blob"),
      grab("tekfdl.py", r"^def _blob", r"^def _superblob"),
      grab("tekfdl.py", r"^_EAR_OUT|^# EAR canon", r"^def ear_field"),
      grab("tekfdl.py", r"^def ear_field", r"^def _add_ears"))

write("field.py", '''"""
The FDL surface equation: one scalar height field for the whole head.

    z(x,y) = skull + forehead + brow + nose + cheeks + lips + chin
             - eyes - philtrum - nostrils

The neck is unioned in with max(), which also gives correct jaw-over-neck
occlusion for free - whichever surface is nearer the viewer wins.

The ear is NOT here: it protrudes sideways, and a z(x,y) height field cannot
express that. It has its own lateral field in anatomy.ear_field.
"""
import numpy as np

from .anatomy import (BROW_POLY, CHEEK, CHIN, EYE, FOREHEAD, LIP_LINE,
                      LOWER_LIP, NOSE_CENTRE, NOSE_TIP, NOSE_WIDTH, NOSTRIL,
                      PHILTRUM, UPPER_LIP, _blob, _ridge, sil_w, skull_base)
''',
      grab("tekfdl.py", r"^MOUTH_BOX", r"^def _nk"),
      grab("tekfdl.py", r"^def _nk", r"^def zfield"),
      grab("tekfdl.py", r"^def zfield", r"^# ---"))

write("contour.py", '''"""
The contour generator - literally the loop from the FDL spec:

    for z = .95 down to -.25 step -.05:
        intersect the surface, extract closed contours, simplify, emit vectors

This is why the whole approach works: contours are LEVEL SETS of the real
surface, so they flow around every feature for free. face_1 drew feature curves
onto an undeformed mesh and they always read as decals.
"""
import math

import numpy as np

from .anatomy import EAR_BOT, EAR_TILT, EAR_TOP, ear_field, sil_w
from .field import MOUTH_BOX, head_mask, zfield
''',
      grab("tekfdl.py", r"^def _march", r"^GRID_RES"),
      grab("tekfdl.py", r"^GRID_RES", r"^def _resample"),
      grab("tekfdl.py", r"^def _resample", r"^def _add_back"),
      grab("tekfdl.py", r"^def _add_back", r"^# EAR canon|^_EAR_OUT"),
      grab("tekfdl.py", r"^def _add_ears", r"^def _add_neck"))
print("done")
