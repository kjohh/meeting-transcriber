#!/bin/bash
# Full build of Meeting Transcriber.app.
#
# Steps:
#   1. Generate icon.icns from assets/icon.png (if needed)
#   2. Fetch + checksum the bundled small whisper model (cached in build-cache/)
#   3. Build the Swift system-audio helper
#   4. py2app full bundle
#   5. Sign inside-out with hardened runtime (scripts/sign-and-package.sh).
#      With DEVELOPER_ID (+ NOTARY_PROFILE) set this also notarizes and makes
#      the .dmg; without it you get an ad-hoc signed dev build.
#
# Output:
#   dist/Meeting Transcriber.app   (+ dist/Meeting Transcriber <ver>.dmg when signed)

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/opt/homebrew/bin/python3.13
BUNDLE_ID=$("$PY" -c "import re;print(re.search(r'^BUNDLE_ID\s*=\s*\"([^\"]+)\"',open('setup.py').read(),re.M).group(1))")

# A release build without the update key would ship an app that can never
# update itself in-app. Refuse instead.
if [ -n "${DEVELOPER_ID:-}" ] && [ ! -s sparkle_public_key.txt ] && [ -z "${MT_SPARKLE_PUBLIC_KEY:-}" ]; then
  echo "✗ sparkle_public_key.txt missing — run scripts/sparkle-keys.sh once on this machine first" >&2
  exit 1
fi

# 1. Icon
if [ ! -f icon.icns ] || [ assets/icon.png -nt icon.icns ]; then
  if [ -f assets/icon.png ]; then
    ./scripts/build-icon.sh
  else
    echo "WARNING: no icon.icns and no assets/icon.png — bundle will use Python default icon"
  fi
fi

# 2. Bundled model (pinned revision + SHA256; see BUNDLED_MODEL in app.py)
MODEL=build-cache/ggml-small-q5_1.bin
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q5_1.bin"
MODEL_SHA=ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb
mkdir -p build-cache
if [ ! -f "$MODEL" ] || [ "$(shasum -a 256 "$MODEL" | cut -d' ' -f1)" != "$MODEL_SHA" ]; then
  echo "→ Fetching bundled model"
  curl -fL --retry 3 -C - -o "$MODEL" "$MODEL_URL"
  if [ "$(shasum -a 256 "$MODEL" | cut -d' ' -f1)" != "$MODEL_SHA" ]; then
    echo "✗ bundled model checksum mismatch" >&2
    rm -f "$MODEL"
    exit 1
  fi
fi

# 2b. Sparkle (in-app updates). Pinned release + SHA256.
SPARKLE_VERSION=2.10.0
SPARKLE_SHA=c2bf58aa8387266ac179357b1415d6f2635f044da8be41042af32425dae6da0c
SPARKLE_DIR="build-cache/Sparkle-$SPARKLE_VERSION"
if [ ! -d "$SPARKLE_DIR/Sparkle.framework" ]; then
  echo "→ Fetching Sparkle $SPARKLE_VERSION"
  TARBALL="build-cache/Sparkle-$SPARKLE_VERSION.tar.xz"
  curl -fL --retry 3 -o "$TARBALL" \
    "https://github.com/sparkle-project/Sparkle/releases/download/$SPARKLE_VERSION/Sparkle-$SPARKLE_VERSION.tar.xz"
  if [ "$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)" != "$SPARKLE_SHA" ]; then
    echo "✗ Sparkle checksum mismatch" >&2
    rm -f "$TARBALL"
    exit 1
  fi
  mkdir -p "$SPARKLE_DIR"
  tar -xf "$TARBALL" -C "$SPARKLE_DIR"
  rm -f "$TARBALL"
fi

# 3. Swift helper
(cd native && swift build -c release)

# 4. py2app
rm -rf build dist
"$PY" setup.py py2app

# 4b. Embed Sparkle.framework. XPCServices are only for sandboxed apps (this
#     app isn't), and Sparkle's docs say to drop them in that case.
FW="dist/Meeting Transcriber.app/Contents/Frameworks"
rm -rf "$FW/Sparkle.framework"
ditto "$SPARKLE_DIR/Sparkle.framework" "$FW/Sparkle.framework"
rm -rf "$FW/Sparkle.framework/Versions/B/XPCServices" "$FW/Sparkle.framework/XPCServices"

# 5. Sign (+ notarize + dmg when DEVELOPER_ID is set)
./scripts/sign-and-package.sh

APP="dist/Meeting Transcriber.app"
echo
echo "=== Bundle ready ==="
du -sh "$APP"
codesign -dvv "$APP" 2>&1 | grep -E "Identifier|Authority|Signature|flags|TeamIdentifier"

# Opt-in (RESET_TCC=1), ad-hoc dev builds only: an ad-hoc signature changes
# with every build and macOS keys permission grants to it, so a rebuilt dev
# app silently loses its grants while System Settings still shows them on.
# Not automatic, because the reset is per bundle ID: it would also wipe the
# grants of the signed release installed in /Applications.
if [ -z "${DEVELOPER_ID:-}" ] && [ "${RESET_TCC:-0}" = "1" ]; then
  tccutil reset AudioCapture "$BUNDLE_ID" > /dev/null 2>&1 || true
  tccutil reset Microphone   "$BUNDLE_ID" > /dev/null 2>&1 || true
  echo "✓ (ad-hoc dev build) permission entries reset — next launch will ask again"
fi

# Dev convenience: RESET_ONBOARDING=1 shows the first-run flow on next launch.
CONFIG="$HOME/Library/Application Support/Meeting Transcriber/.config.json"
if [ "${RESET_ONBOARDING:-0}" = "1" ] && [ -f "$CONFIG" ]; then
  "$PY" - "$CONFIG" <<'EOF'
import json, sys
p = sys.argv[1]
try:
    cfg = json.load(open(p))
except Exception:
    cfg = {}
cfg.pop("onboarding_completed", None)
json.dump(cfg, open(p, "w"))
print("✓ Onboarding flag reset — next launch shows first-run flow")
EOF
fi

echo
echo "→ Double-click $APP from Finder, or:"
echo "   open \"$APP\""
