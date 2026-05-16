#!/bin/bash
# Full release build of Meeting Transcriber.app.
#
# Steps:
#   1. Generate icon.icns from assets/icon.png (if needed)
#   2. Clean previous build/dist
#   3. py2app full bundle
#   4. Deep ad-hoc sign so TCC matches the bundle and the embedded Swift
#      audio binary across launches. py2app only signs the main wrapper —
#      without this extra step the coreaudio_tap helper is treated as an
#      unsigned child, so macOS re-prompts for screen recording permission
#      on every launch.
#
# Output:
#   dist/Meeting Transcriber.app

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/opt/homebrew/bin/python3.13

# 1. Icon
if [ ! -f icon.icns ] || [ assets/icon.png -nt icon.icns ]; then
  if [ -f assets/icon.png ]; then
    ./scripts/build-icon.sh
  else
    echo "WARNING: no icon.icns and no assets/icon.png — bundle will use Python default icon"
  fi
fi

# 2. Clean
rm -rf build dist

# 3. py2app
"$PY" setup.py py2app

# 4. Deep ad-hoc sign
APP="dist/Meeting Transcriber.app"
xattr -cr "$APP"
codesign --force --deep --sign - "$APP"

echo
echo "=== Bundle ready ==="
du -sh "$APP"
codesign -dvv "$APP" 2>&1 | grep -E "Identifier|Signature|flags"

# Reset TCC permission entries for this bundle. Every py2app rebuild
# produces a new code-signature hash; macOS TCC treats it as a different
# app instance even though the bundle identifier is unchanged, so the old
# "approved" entry in System Settings is stale and CGPreflight returns
# false in the new bundle. Clearing it forces a clean re-grant flow.
tccutil reset ScreenCapture com.kylehsia.meeting-transcriber > /dev/null 2>&1 || true
tccutil reset Microphone   com.kylehsia.meeting-transcriber > /dev/null 2>&1 || true
echo "✓ TCC entries reset — next launch will re-prompt for permissions"

# Reset onboarding flag in the bundle's user-data dir so the next launch
# shows the first-run onboarding flow. Useful for iterating on the wizard.
# Other settings (API key, backend choice) are preserved.
CONFIG="$HOME/Library/Application Support/Meeting Transcriber/.config.json"
if [ -f "$CONFIG" ]; then
  "$PY" - <<EOF
import json
p = "$CONFIG"
try:
    cfg = json.load(open(p))
except Exception:
    cfg = {}
cfg.pop("onboarding_completed", None)
json.dump(cfg, open(p, "w"))
print("✓ Onboarding flag reset — next launch will show the first-run flow")
EOF
fi

echo
echo "→ Double-click $APP from Finder, or:"
echo "   open \"$APP\""
