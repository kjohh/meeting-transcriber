#!/bin/bash
# One-time setup on the RELEASE machine (the one with the Developer ID
# certificate): create the EdDSA key pair that signs in-app updates.
#
#   - The private key is stored in this Mac's login Keychain by Sparkle's
#     generate_keys tool and never leaves it. scripts/release.sh uses it.
#   - The public key is written to sparkle_public_key.txt. Commit that file:
#     it is built into every app and only verifies updates.
#
# Running it again is safe: an existing key is reused, not replaced.
# Back the private key up (generate_keys -x <file>) somewhere safe. Losing it
# means installed copies can no longer update in-app.

set -euo pipefail
cd "$(dirname "$0")/.."

GEN="$(ls -d build-cache/Sparkle-*/bin/generate_keys 2>/dev/null | tail -1)"
if [ -z "$GEN" ]; then
  echo "→ Sparkle tools not found; fetching them via a dev build step…"
  ./scripts/build-app.sh >/dev/null
  GEN="$(ls -d build-cache/Sparkle-*/bin/generate_keys | tail -1)"
fi

"$GEN" >/dev/null           # creates the key if this Mac has none yet
PUB="$("$GEN" -p)"          # prints only the public key
printf '%s\n' "$PUB" > sparkle_public_key.txt
echo "✓ Public key written to sparkle_public_key.txt:"
echo "  $PUB"
echo "  Commit it:  git add sparkle_public_key.txt && git commit -m 'Add Sparkle update key'"
