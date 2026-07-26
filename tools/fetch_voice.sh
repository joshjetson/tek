#!/bin/bash
# Fetch Piper voice models into voices/.
#
#   tools/fetch_voice.sh en_US-amy-medium en_GB-alan-medium
#   tools/fetch_voice.sh --list
#
# Models are ~61 MB each and deliberately NOT in git - they are refetchable
# binaries, and a repo that carries a few hundred MB of them is unpleasant to
# clone. This is how you get them back on a fresh checkout.
set -u
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main
DIR="$(cd "$(dirname "$0")/.." && pwd)/voices"
mkdir -p "$DIR"

if [ "${1:-}" = "--list" ]; then
    curl -sSL "$BASE/voices.json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for k,v in sorted(d.items()):
    if not v.get("language",{}).get("code","").startswith("en"): continue
    n=v.get("num_speakers",1)
    print("  %-34s %s" % (k, ("%d speakers"%n) if n>1 else ""))'
    exit 0
fi

for KEY in "$@"; do
    # en_US-amy-medium -> en/en_US/amy/medium/en_US-amy-medium.onnx
    LANG=${KEY%%-*}                     # en_US
    REST=${KEY#*-}                      # amy-medium
    NAME=${REST%-*}                     # amy
    QUAL=${REST##*-}                    # medium
    FAM=${LANG%%_*}                     # en
    URL="$BASE/$FAM/$LANG/$NAME/$QUAL/$KEY"
    if [ -f "$DIR/$KEY.onnx" ] && [ -f "$DIR/$KEY.onnx.json" ]; then
        echo "  have    $KEY"; continue
    fi
    printf "  fetch   %-34s " "$KEY"
    if curl -sSLf -o "$DIR/$KEY.onnx.json" "$URL.onnx.json" \
       && curl -sSLf -o "$DIR/$KEY.onnx" "$URL.onnx"; then
        echo "$(du -h "$DIR/$KEY.onnx" | cut -f1)"
    else
        rm -f "$DIR/$KEY.onnx" "$DIR/$KEY.onnx.json"
        echo "FAILED (no such voice?)"
    fi
done
