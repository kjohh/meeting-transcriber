#!/bin/bash
set -e

echo "=== Meeting Transcriber Setup ==="
echo ""

PYTHON="/opt/homebrew/bin/python3.13"

if [ ! -f "$PYTHON" ]; then
  echo "ERROR: python3.13 not found at $PYTHON"
  echo "Install it first:"
  echo "  brew install python@3.13"
  exit 1
fi

echo "1. Installing Python dependencies..."
$PYTHON -m pip install --break-system-packages -r requirements.txt
echo "   Done."
echo ""

echo "2. Building Swift audio binary..."
cd native && swift build -c release 2>&1
cd ..
echo "   Done. Binary at: native/.build/release/coreaudio_tap"
echo ""

echo "=== Setup complete ==="
echo ""
echo "Launch: open 'Meeting Transcriber.app'"
echo ""
echo "First launch — grant these permissions when prompted:"
echo "  System Settings → Privacy & Security → Screen & System Audio Recording"
echo "  System Settings → Privacy & Security → Microphone"
