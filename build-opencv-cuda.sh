#!/bin/bash
# Build OpenCV 4.5.5 + contrib with CUDA for Jetson Nano 2GB (L4T 32.5.2).
#
# Runs as 'super' (NOT root): cmake must be able to import the pip-installed
# numpy in ~/.local to build the Python bindings. Only 'make install' needs root.
#
# Installs to /usr/local, which precedes /usr/lib/python3.6/dist-packages in
# sys.path - so the new cv2 shadows NVIDIA's 4.1.1 without removing the
# libopencv debs. If this build is bad, delete /usr/local's cv2 and 4.1.1 is back.
set -e

VER=4.5.5
SRC=$HOME/opencv_src
JOBS=${JOBS:-2}          # 2GB RAM: nvcc units are ~1GB each. -j4 OOMs.

# numpy 1.19.5 on Cortex-A57 SIGILLs unless OpenBLAS core detection is pinned.
export OPENBLAS_CORETYPE=ARMV8

echo "=== OpenCV $VER + CUDA build | jobs=$JOBS | $(date) ==="

mkdir -p "$SRC" && cd "$SRC"
for repo in opencv opencv_contrib; do
  if [ ! -d "$repo" ]; then
    echo "--- cloning $repo $VER"
    git clone --depth 1 --branch "$VER" "https://github.com/opencv/$repo.git"
  else
    echo "--- $repo already present, reusing"
  fi
done

mkdir -p "$SRC/opencv/build" && cd "$SRC/opencv/build"

echo "=== configuring ==="
cmake \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH="$SRC/opencv_contrib/modules" \
  \
  `# --- CUDA. 5.3 = Maxwell GM20B, the Nano's GPU. Building only this arch` \
  `# --- keeps compile time and binary size down enormously.` \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=5.3 \
  -D CUDA_ARCH_PTX= \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  \
  `# --- cuDNN 8.0 is installed, so the dnn module gets a CUDA backend.` \
  `# --- Requires compute >= 5.3; the Nano is exactly 5.3, so it qualifies.` \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  \
  `# --- On Tegra, hardware video en/decode is reached through GStreamer` \
  `# --- (nvv4l2decoder / nvarguscamerasrc), NOT through NVCUVID. So GStreamer` \
  `# --- is what actually unlocks NVENC/NVDEC and the CSI camera.` \
  -D WITH_GSTREAMER=ON \
  -D WITH_NVCUVID=OFF \
  -D WITH_LIBV4L=ON \
  -D WITH_V4L=ON \
  \
  -D WITH_OPENGL=ON \
  -D WITH_GTK=ON \
  -D WITH_TBB=ON \
  -D WITH_EIGEN=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_python2=OFF \
  -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
  `# --- OpenCV defaults to .../site-packages, but Debian/Ubuntu only put` \
  `# --- .../dist-packages on sys.path. Without this the new cv2 installs to a` \
  `# --- directory Python never searches and 'import cv2' silently keeps` \
  `# --- returning NVIDIA's 4.1.1.` \
  -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.6/dist-packages \
  \
  `# --- Skipping tests/samples/java cuts hours off the build.` \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_JAVA=OFF \
  -D BUILD_DOCS=OFF \
  \
  -D CMAKE_VERBOSE_MAKEFILE=OFF \
  .. 2>&1 | tail -40

echo
echo "=== configure summary (the lines that matter) ==="
grep -iE 'NVIDIA CUDA|cuDNN|GStreamer|Python 3|install path|NVIDIA GPU arch' \
  CMakeCache.txt >/dev/null 2>&1 || true
sed -n '/NVIDIA CUDA/,/^$/p;/Python 3/,/^$/p;/Video I\/O/,/^$/p' CMakeVars.txt 2>/dev/null || true

if [ "${CONFIGURE_ONLY:-0}" = "1" ]; then
  echo
  echo "=== CONFIGURE_ONLY set - stopping before the long compile ==="
  exit 0
fi

echo
echo "=== compiling with -j$JOBS (expect 3-5 hours on this board) ==="
make -j"$JOBS"

echo
echo "=== build finished OK at $(date) ==="
echo "Now run:  sudo make install && sudo ldconfig"
echo "  (from $SRC/opencv/build)"
