#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Prove the CUDA build actually accelerates work, not just that it links."""
import time
import cv2
import numpy as np

print(f"OpenCV {cv2.__version__}   CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
print()

img = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)


def bench(fn, n=12):
    fn()                       # warm-up: first CUDA call pays context init (~seconds)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


gpu_src = cv2.cuda_GpuMat()
gpu_src.upload(img)

tests = []

# --- Gaussian blur -----------------------------------------------------------
gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (31, 31), 0)
tests.append((
    "GaussianBlur 31x31",
    lambda: cv2.GaussianBlur(img, (31, 31), 0),
    lambda: gauss.apply(gpu_src),
))

# --- Resize ------------------------------------------------------------------
tests.append((
    "Resize 2x cubic",
    lambda: cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
    lambda: cv2.cuda.resize(gpu_src, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
))

# --- Colour convert ----------------------------------------------------------
tests.append((
    "BGR->Gray",
    lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    lambda: cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGR2GRAY),
))

print(f"{'operation':<22} {'CPU (ms)':>10} {'GPU (ms)':>10} {'speedup':>9}")
print("-" * 54)
for name, cpu_fn, gpu_fn in tests:
    c = bench(cpu_fn)
    g = bench(gpu_fn)
    print(f"{name:<22} {c:>10.1f} {g:>10.1f} {c/g:>8.1f}x")

print()
print("Note: GPU timings exclude host<->device transfer (data already resident).")
print("For a one-shot op on a fresh frame, upload/download can dominate; the win")
print("is real when you chain several ops on-GPU before downloading once.")

# --- DNN CUDA backend --------------------------------------------------------
print()
backends = [b for b, _ in cv2.dnn.getAvailableBackends()] \
    if hasattr(cv2.dnn, "getAvailableBackends") else []
print("DNN CUDA backend available:",
      hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and cv2.cuda.getCudaEnabledDeviceCount() > 0)

# --- GStreamer (this is what reaches NVENC/NVDEC and the CSI camera) ---------
info = cv2.getBuildInformation()
for key in ("GStreamer", "v4l/v4l2", "NVIDIA CUDA", "cuDNN"):
    line = next((l.strip() for l in info.splitlines() if l.strip().startswith(key)), None)
    if line:
        print(line)
