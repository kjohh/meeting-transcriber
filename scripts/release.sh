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
# What end users download:
#   Meeting Transcriber.zip  →  unzip  →  drag to Applications
#
# We use `ditto` (not `zip`) because it preserves macOS metadata, code
# signatures, and resource forks. Plain `zip` strips the signature and
# Gatekeeper will reject the .app.

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

ZIP="dist/Meeting Transcriber.zip"
echo "→ Packaging .app with ditto (preserves signatures)…"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
ls -lh "$ZIP"

echo
echo "→ Creating GitHub release ${TAG}…"
# --generate-notes auto-fills the body with the commit list since the
# previous tag, so users see what changed without us writing it by hand.
gh release create "${TAG}" "$ZIP" \
  --title "${TAG}" \
  --generate-notes

echo
echo "✓ Released. End users download from:"
gh release view "${TAG}" --json url --jq .url
