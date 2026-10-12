#!/bin/bash
# Cut a GitHub release with the .app zipped up.
#
# Usage:
#   scripts/release.sh v1.0.0
#
# Prerequisites:
#   - `gh` CLI authenticated against this repo
#   - dist/Meeting Transcriber.app must already be built (run scripts/build-app.sh first)
#
#   - build must be a Developer ID build (DEVELOPER_ID + NOTARY_PROFILE set),
#     which produces a signed, notarized and stapled dmg
#
# What end users download:
#   Meeting.Transcriber.dmg  →  open  →  drag to Applications

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <tag>   (e.g. v1.0.0)" >&2
  exit 1
fi

TAG=$1
cd "$(dirname "$0")/.."

APP="dist/Meeting Transcriber.app"
if [ ! -d "$APP" ]; then
  echo "✗ $APP not found — run scripts/build-app.sh first" >&2
  exit 1
fi

VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist")
DMG="dist/Meeting Transcriber ${VERSION}.dmg"
if [ ! -f "$DMG" ]; then
  echo "✗ $DMG not found — run a Developer ID build first (DEVELOPER_ID + NOTARY_PROFILE)" >&2
  exit 1
fi

# Refuse to ship a dmg that Gatekeeper would block (ad-hoc / unnotarized).
echo "→ Checking Gatekeeper on $DMG…"
if ! spctl -a -t open --context context:primary-signature "$DMG" 2>/dev/null; then
  spctl -a -t open --context context:primary-signature -vv "$DMG" >&2 || true
  echo "✗ $DMG is not notarized — Gatekeeper would reject it" >&2
  exit 1
fi
ls -lh "$DMG"

# Upload under a fixed name so README's releases/latest/download link keeps
# working across versions (GitHub turns the space into a dot).
ASSET="dist/Meeting Transcriber.dmg"
cp "$DMG" "$ASSET"

echo
echo "→ Creating GitHub release ${TAG}…"
# --generate-notes auto-fills the body with the commit list since the
# previous tag, so users see what changed without us writing it by hand.
# --target pins the tag to the exact commit we built + pushed, not the repo
# default branch (releases are cut from the working branch, e.g. wip-machine-b).
gh release create "${TAG}" "$ASSET" \
  --title "${TAG}" \
  --target "$(git rev-parse HEAD)" \
  --generate-notes

echo
echo "✓ Released. End users download from:"
gh release view "${TAG}" --json url --jq .url
