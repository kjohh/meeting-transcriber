"""py2app build configuration for Meeting Transcriber.

Build:
    /opt/homebrew/bin/python3.13 setup.py py2app          # release build
    /opt/homebrew/bin/python3.13 setup.py py2app -A       # alias mode (dev iterate)

Output:
    dist/Meeting Transcriber.app

Notes:
- One small model (build-cache/ggml-small-q5_1.bin, fetched + checksummed by
  scripts/build-app.sh) is bundled under Contents/Resources/models/ so local
  transcription works with zero download. The larger models are downloaded
  in the background into ~/Library/Application Support/pywhispercpp/models/.
- Config/vocab live in ~/Library/Application Support/Meeting Transcriber/
  when running from the bundle.
- The Swift coreaudio_tap binary is bundled under Contents/Resources/native/.
"""
import re

from setuptools import setup

APP = ["app.py"]

# Single source of truth for the version is APP_VERSION in app.py. Read it via
# regex (not import — that would pull in flask/groq/etc.) so the bundle's
# CFBundleVersion can never drift from the runtime version-change gate.
_m = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', open("app.py", encoding="utf-8").read(), re.M)
if not _m:
    raise SystemExit("setup.py: could not find APP_VERSION in app.py")
APP_VERSION = _m.group(1)

# The bundle identifier is what macOS TCC keys permissions on. Once the first
# notarized build ships it must never change (users would have to re-grant
# microphone + system audio). Settle the product name first.
BUNDLE_ID = "com.kylehsia.meeting-transcriber"
APP_NAME = "Meeting Transcriber"

DATA_FILES = [
    ("static", ["static/index.html"]),
    ("native/.build/release", ["native/.build/release/coreaudio_tap"]),
    ("models", ["build-cache/ggml-small-q5_1.bin"]),
    ("", ["THIRD_PARTY_NOTICES.md"]),
]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "icon.icns",
    # Hidden imports py2app's static analyser misses. Most of these are
    # discovered lazily (entry points, importlib, dynamic factories).
    "includes": [
        "webview",
        "webview.platforms.cocoa",
        "pywhispercpp",
        "pywhispercpp.model",
        "pywhispercpp.constants",
        "sounddevice",
        "numpy",
        "flask",
        "groq",
        "AVFoundation",
        "Security",
    ],
    # Packages listed here are extracted as plain directories instead of being
    # zipped into python313.zip. Required for any package shipping dylibs
    # (sounddevice → libportaudio.dylib; pywhispercpp → whisper.cpp ggml libs),
    # because dlopen can't load from inside a zip.
    "packages": [
        "pywhispercpp",
        "sounddevice",
        "_sounddevice_data",
        "groq",
        "flask",
        "werkzeug",
        "jinja2",
        "click",
        "blinker",
        "itsdangerous",
        "markupsafe",
        "certifi",
        "charset_normalizer",
        "idna",
        "urllib3",
        "requests",
    ],
    "excludes": [
        "tkinter",
        "matplotlib",
        "pandas",
        "scipy",
        "pytest",
        "PIL",
        # Build tooling, never imported at runtime. Excluding it also keeps
        # its vendored packages (one of them LGPL-3.0) out of the bundle.
        "setuptools",
        "pkg_resources",
        "wheel",
    ],
    "plist": {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleDevelopmentRegion": "zh_TW",
        "NSMicrophoneUsageDescription":
            "用來把你說的話轉成逐字稿。錄音只在你按下開始後進行。",
        # Core Audio process tap (macOS 14.4+): "System Audio Recording Only".
        # Replaces the old Screen Recording permission; the screen is never read.
        "NSAudioCaptureUsageDescription":
            "用來擷取電腦播放的聲音（例如線上會議中對方的聲音），轉成逐字稿。不會讀取你的畫面。",
        "LSMinimumSystemVersion": "14.4",
        # Apple Silicon only: local transcription is too slow on Intel to be usable.
        "LSArchitecturePriority": ["arm64"],
        "LSUIElement": False,
        "NSHighResolutionCapable": True,
    },
}

setup(
    app=APP,
    name="Meeting Transcriber",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
