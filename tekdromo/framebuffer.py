"""
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


def unblank(fd):
    """The kernel console blanker (consoleblank=600 here) switches the display
    off after 10 min with no console keypress. We keep drawing happily into a
    dark panel. Re-assert unblank periodically so the display cannot die under
    us regardless of how the console is configured."""
    try:
        fcntl.ioctl(fd, FBIOBLANK, FB_BLANK_UNBLANK)
    except OSError:
        pass

def fb_info(dev="/dev/fb0"):
    with open(dev, "rb") as f:
        u = struct.unpack("<40I", fcntl.ioctl(f, 0x4600, b"\0" * 160)[:160])
    xres, yres, bpp = u[0], u[1], u[6]
    with open("/sys/class/graphics/fb0/stride") as f:
        stride = int(f.read().strip())
    return xres, yres, bpp, stride

def open_screen(dev="/dev/fb0"):
    """Returns (fd, mmap, ndarray view, w, h). The array writes straight to the
    panel."""
    w, h, bpp, stride = fb_info(dev)
    if bpp != 32 or stride != w * 4:
        raise RuntimeError("unexpected framebuffer: %dbpp stride=%d" % (bpp, stride))
    fd = os.open(dev, os.O_RDWR)
    mm = mmap.mmap(fd, h * stride, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE)
    screen = np.frombuffer(mm, dtype=np.uint8).reshape(h, w, 4)
    return fd, mm, screen, w, h
