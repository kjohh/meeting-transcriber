#!/bin/bash
# Sign dist/<App>.app inside-out with hardened runtime, then (with a Developer
# ID) notarize + staple and wrap it in a signed .dmg.
#
# Modes (picked from the environment):
#   DEVELOPER_ID unset   → ad-hoc signature, hardened runtime still on so the
#                          entitlements get exercised locally. No notarization.
#   DEVELOPER_ID set     → e.g. "Developer ID Application: Kyle Hsia (TEAMID)".
#                          Adds a secure timestamp. With NOTARY_PROFILE also
#                          set (a `xcrun notarytool store-credentials` profile
#                          name), notarizes the app and the dmg and staples both.
#
# Why inside-out instead of `codesign --deep`: --deep signs nested code with
# the outer bundle's options, can't give helpers their own entitlements, and
# Apple's notary service rejects any nested Mach-O that lacks hardened
# runtime or a timestamp. Signing every binary explicitly avoids all three.

set -euo pipefail
cd "$(dirname "$0")/.."

APP_NAME="${APP_NAME:-Meeting Transcriber}"
APP="dist/${APP_NAME}.app"
IDENTITY="${DEVELOPER_ID:--}"
# Release: minimal set. Ad-hoc: + disable-library-validation (see the plist).
if [ "$IDENTITY" = "-" ]; then ENT="entitlements-dev.plist"; else ENT="entitlements.plist"; fi

[ -d "$APP" ] || { echo "✗ $APP not found" >&2; exit 1; }

sign() {
  # $1 = path, $2 = "ent" to attach entitlements
  local args=(--force --options runtime --sign "$IDENTITY")
  [ "$IDENTITY" != "-" ] && args+=(--timestamp)
  [ "${2:-}" = "ent" ] && args+=(--entitlements "$ENT")
  codesign "${args[@]}" "$1"
}

# py2app copies some packages read-only; codesign needs to rewrite them.
chmod -R u+w "$APP"
xattr -cr "$APP"

echo "→ Signing nested code (identity: $IDENTITY)"
MAIN_EXE="$APP/Contents/MacOS/$APP_NAME"
FRAMEWORK="$APP/Contents/Frameworks/Python.framework"

# 1. Every loose Mach-O (dylibs, .so extension modules, helper executables),
#    deepest first. Files inside Python.framework's own binary slot and the
#    main executable are signed as part of their bundles below.
count=0
while IFS= read -r f; do
  if file -b "$f" | grep -q "Mach-O"; then
    case "$f" in
      "$MAIN_EXE") continue ;;
    esac
    # Executables that run app code (the multiprocessing child re-launches
    # Contents/MacOS/python) need the same entitlements as the main binary.
    if [ "$f" = "$APP/Contents/MacOS/python" ]; then
      sign "$f" ent
    else
      sign "$f"
    fi
    count=$((count + 1))
  fi
done < <(find "$APP/Contents" -type f ! -path "*/_CodeSignature/*" \
          | awk -F/ '{ print NF "\t" $0 }' | sort -rn | cut -f2-)
echo "  signed $count nested binaries"

# 2. Python.framework as a bundle (its Versions/Current seal).
if [ -d "$FRAMEWORK" ]; then
  for v in "$FRAMEWORK"/Versions/*; do
    [ -L "$v" ] && continue
    sign "$v"
  done
fi

# 3. The app bundle itself, with entitlements.
sign "$APP" ent

echo "→ Verifying"
codesign --verify --strict --deep --verbose=2 "$APP" 2>&1 | tail -2
codesign -d --entitlements - --xml "$APP" 2>/dev/null | plutil -p - 2>/dev/null || true

if [ "$IDENTITY" = "-" ]; then
  echo "(ad-hoc build: no notarization, no dmg)"
  exit 0
fi

notarize() {
  # $1 = file to submit (zip or dmg)
  if [ -z "${NOTARY_PROFILE:-}" ]; then
    echo "  NOTARY_PROFILE not set — skipping notarization of $1"
    return 1
  fi
  xcrun notarytool submit "$1" --keychain-profile "$NOTARY_PROFILE" --wait
}

echo "→ Notarizing app"
ZIP="dist/${APP_NAME}-notarize.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
if notarize "$ZIP"; then
  xcrun stapler staple "$APP"
fi
rm -f "$ZIP"

echo "→ Building dmg"
VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist")
DMG="dist/${APP_NAME} ${VERSION}.dmg"
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"

echo "→ Notarizing dmg"
if notarize "$DMG"; then
  xcrun stapler staple "$DMG"
  spctl -a -t open --context context:primary-signature -v "$DMG" 2>&1 | tail -1
fi
echo "✓ $DMG"
