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
#   - `npx wrangler login` done once on this machine (uploads to Cloudflare R2)
#
# What end users download:
#   Meeting.Transcriber.dmg  →  open  →  drag to Applications
#
# Where the dmg is downloaded from:
#   Cloudflare R2 (bucket kylehsia-downloads, https://downloads.kylehsia.com).
#   GitHub release assets measured 0.06–0.12 MB/s from Taiwan vs ~8 MB/s
#   from R2 (2026-10-14), i.e. ~40 minutes vs ~30 seconds for the 240 MB dmg.
#   The dmg is still attached to the GitHub release as a backup, and
#   appcast.xml stays on GitHub because already-installed apps read
#   SUFeedURL from there — only the enclosure URL inside it points to R2.

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
echo "→ Checking Gatekeeper on ${DMG}…"
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

# In-app update feed (Sparkle). sign_update reads the EdDSA private key from
# this machine's Keychain (created by scripts/sparkle-keys.sh); the app only
# installs an update whose signature matches the public key it shipped with.
SIGN_UPDATE="$(ls -d build-cache/Sparkle-*/bin/sign_update 2>/dev/null | tail -1)"
if [ -z "$SIGN_UPDATE" ]; then
  echo "✗ Sparkle tools not found — run scripts/build-app.sh first" >&2
  exit 1
fi
echo "→ Signing update for Sparkle…"
if ! SIG_ATTRS=$("$SIGN_UPDATE" "$ASSET"); then
  echo "✗ sign_update failed — is the update key on this machine? (scripts/sparkle-keys.sh)" >&2
  exit 1
fi
PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || true)
NOTES=$(git log --no-merges --format='<li>%s</li>' ${PREV_TAG:+$PREV_TAG..}HEAD | head -30 \
        | sed 's/&/\&amp;/g; s/<\([^l/]\)/\&lt;\1/g')
REPO_URL="https://github.com/kjohh/meeting-transcriber"

# Upload the dmg to R2 before anything points at it. Each version gets its
# own immutable path; latest/ is the stable link for README and the website,
# with a short cache so a new release shows up within minutes.
WRANGLER="npx --yes wrangler@4"
BUCKET="kylehsia-downloads"
PREFIX="meeting-transcriber"
DOWNLOADS="https://downloads.kylehsia.com"
DMG_URL="${DOWNLOADS}/${PREFIX}/${VERSION}/Meeting.Transcriber.dmg"
r2_put() {
  $WRANGLER r2 object put "${BUCKET}/$1" --file "$ASSET" --remote \
    --content-type application/x-apple-diskimage --cache-control "$2"
}
echo "→ Uploading to Cloudflare R2…"
r2_put "${PREFIX}/${VERSION}/Meeting.Transcriber.dmg" "public, max-age=31536000, immutable"
r2_put "${PREFIX}/latest/Meeting.Transcriber.dmg" "public, max-age=300"

# Never publish an appcast whose download is missing or truncated.
LOCAL_SIZE=$(stat -f%z "$ASSET")
for url in "$DMG_URL" "${DOWNLOADS}/${PREFIX}/latest/Meeting.Transcriber.dmg"; do
  # ?v= skips any edge-cached copy of the previous latest/
  REMOTE_SIZE=$(curl -sfI "${url}?v=${VERSION}" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r' || true)
  if [ "$REMOTE_SIZE" != "$LOCAL_SIZE" ]; then
    echo "✗ $url returned size '${REMOTE_SIZE}', expected ${LOCAL_SIZE}" >&2
    exit 1
  fi
done

MIN_OS=$(/usr/libexec/PlistBuddy -c "Print :LSMinimumSystemVersion" "$APP/Contents/Info.plist")
BUILD=$(/usr/libexec/PlistBuddy -c "Print :CFBundleVersion" "$APP/Contents/Info.plist")
APPCAST="dist/appcast.xml"
cat > "$APPCAST" <<XML
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Meeting Transcriber</title>
    <item>
      <title>${VERSION}</title>
      <pubDate>$(LC_ALL=C date -u "+%a, %d %b %Y %H:%M:%S +0000")</pubDate>
      <sparkle:version>${BUILD}</sparkle:version>
      <sparkle:shortVersionString>${VERSION}</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>${MIN_OS}</sparkle:minimumSystemVersion>
      <description><![CDATA[<ul>${NOTES}</ul><p><a href="${REPO_URL}/releases/tag/${TAG}">完整說明</a></p>]]></description>
      <enclosure url="${DMG_URL}"
                 type="application/octet-stream" ${SIG_ATTRS} />
    </item>
  </channel>
</rss>
XML
xmllint --noout "$APPCAST"

echo
echo "→ Creating GitHub release ${TAG}…"
# --generate-notes auto-fills the body with the commit list since the
# previous tag, so users see what changed without us writing it by hand.
# --target pins the tag to the exact commit we built + pushed, not the repo
# default branch (releases are cut from the working branch, e.g. wip-machine-b).
gh release create "${TAG}" "$ASSET" "$APPCAST" \
  --title "${TAG}" \
  --target "$(git rev-parse HEAD)" \
  --generate-notes

# Keep the 5 newest versions on R2 (free tier is 10 GB; each dmg is ~240 MB).
# Releases are cut one at a time, so dropping the 6th newest is enough.
# Installed apps only ever download the version in the current appcast.
OLD_TAG=$(gh release list --limit 10 --json tagName --jq '.[5].tagName // empty')
if [ -n "$OLD_TAG" ]; then
  echo "→ Removing ${OLD_TAG#v} from R2…"
  $WRANGLER r2 object delete "${BUCKET}/${PREFIX}/${OLD_TAG#v}/Meeting.Transcriber.dmg" --remote \
    || echo "  (nothing to remove)"
fi

echo
echo "✓ Released. End users download from:"
echo "  ${DOWNLOADS}/${PREFIX}/latest/Meeting.Transcriber.dmg"
gh release view "${TAG}" --json url --jq .url
