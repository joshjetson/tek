#!/bin/bash
# Cleanup to bare essentials. Everything is tracked in git, so all of this is
# recoverable with `git checkout HEAD~1 -- <file>`.
set -e
cd /home/super

echo "=== DEAD MODULES (superseded, nothing imports them) ==="
# tekanat: face_1's model. Superseded by tekfdl; preserved in faces/face_1/.
# tekface: old runner, replaced by tekrun (which has the never-stop guarantees).
# tekscreen: X-based renderer, replaced by direct framebuffer output.
# tekctl.sh: manual start/stop, replaced by systemctl.
for f in tekanat.py tekface.py tekscreen.py tekctl.sh; do
    [ -e "$f" ] && git rm -q "$f" && echo "  removed $f"
done

echo "=== ONE-OFF DIAGNOSTICS (served their purpose) ==="
# Each of these answered one question during development and has no further
# use. The ones worth keeping are moved to tools/ and tests/ below.
for f in bc.py dbg.py rigdbg.py camtest.py camdiag.py camtrack_test.py \
         holediag.py livediff.py browtest.py rigtest.py killtest.sh \
         inventory.py build_check.py cleanup.sh; do
    [ -e "$f" ] && git rm -q "$f" 2>/dev/null && echo "  removed $f"
done

echo "=== STALE SNAPSHOT ==="
# faces/face_2/ was a copy of the live base face. tekfdl.py IS base v1, so the
# copy is duplication that drifts - it was already several fixes out of date.
# faces/face_1/ stays: it is a genuinely superseded approach worth keeping.
if [ -d faces/face_2 ]; then
    git rm -rq faces/face_2 && echo "  removed faces/face_2 (duplicate of tekfdl.py)"
fi

echo "=== ORGANISE KEEPERS ==="
mkdir -p tools tests docs
git mv -f holecheck.py   tests/  2>/dev/null && echo "  tests/holecheck.py"
git mv -f follow_unit.py tests/  2>/dev/null && echo "  tests/follow_unit.py"
git mv -f abtest.py      tools/  2>/dev/null && echo "  tools/abtest.py"
git mv -f grabfb.py      tools/  2>/dev/null && echo "  tools/grabfb.py"
git mv -f verify-opencv-cuda.py tools/ 2>/dev/null && echo "  tools/verify-opencv-cuda.py"
git mv -f build-opencv-cuda.sh  tools/ 2>/dev/null && echo "  tools/build-opencv-cuda.sh"
git mv -f nano-tune-REVERT.md   docs/  2>/dev/null && echo "  docs/nano-tune-REVERT.md"

echo
echo "=== REMAINING PROJECT FILES ==="
ls -1 *.py 2>/dev/null | sed 's/^/  runtime: /'
ls -1 tools tests docs 2>/dev/null >/dev/null && for d in tools tests docs; do
    ls -1 $d 2>/dev/null | sed "s|^|  $d/|"
done
