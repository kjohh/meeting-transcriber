#!/bin/bash
# Build icon.icns from a 1024x1024 RGB(A) PNG.
#
# Usage:
#   scripts/build-icon.sh                  # uses assets/icon.png by default
#   scripts/build-icon.sh path/to/src.png
#
# Output:
#   icon.icns (project root, picked up by setup.py)
set -euo pipefail

SRC="${1:-assets/icon.png}"
OUT="icon.icns"

if [ ! -f "$SRC" ]; then
  echo "Source PNG not found: $SRC" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap "rm -rf '$TMP'" EXIT
ICONSET="$TMP/icon.iconset"
mkdir -p "$ICONSET"

# Apple's required sizes for an .iconset:
#   16, 32, 64 (32@2x), 128, 256, 512, 1024 (512@2x)
# Each "logical" size also needs an @2x retina variant where applicable.
for size in 16 32 128 256 512; do
  sips -z $size $size "$SRC" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null
  dbl=$((size * 2))
  sips -z $dbl $dbl "$SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null
done

iconutil -c icns -o "$OUT" "$ICONSET"
echo "✓ Generated $OUT from $SRC"
