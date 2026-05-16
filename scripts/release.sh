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
gh release create "${TAG}" "$ZIP" \
  --title "${TAG}" \
  --notes "Meeting Transcriber ${TAG}

Download \`Meeting Transcriber.zip\`, unzip, drag the app to your Applications folder.

First launch will walk you through screen-recording + microphone permissions and let you pick between cloud (Groq) or local (whisper.cpp) transcription."

echo
echo "✓ Released. End users download from:"
gh release view "${TAG}" --json url --jq .url
