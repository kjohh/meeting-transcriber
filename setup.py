"""py2app build configuration for Meeting Transcriber.

Build:
    /opt/homebrew/bin/python3.13 setup.py py2app          # release build
    /opt/homebrew/bin/python3.13 setup.py py2app -A       # alias mode (dev iterate)

Output:
    dist/Meeting Transcriber.app

Notes:
- Models are NOT bundled. First-launch onboarding downloads them via
  pywhispercpp into ~/Library/Application Support/pywhispercpp/models/.
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

DATA_FILES = [
    ("static", ["static/index.html"]),
    ("native/.build/release", ["native/.build/release/coreaudio_tap"]),
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
        "huggingface_hub",
        "sounddevice",
        "numpy",
        "flask",
        "groq",
        "Quartz",
        "AVFoundation",
    ],
    # Packages listed here are extracted as plain directories instead of being
    # zipped into python313.zip. Required for any package shipping dylibs
    # (sounddevice → libportaudio.dylib; pywhispercpp → whisper.cpp ggml libs),
    # because dlopen can't load from inside a zip.
    "packages": [
        "pywhispercpp",
        "sounddevice",
        "_sounddevice_data",
        "huggingface_hub",
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
    ],
    "plist": {
        "CFBundleName": "Meeting Transcriber",
        "CFBundleDisplayName": "Meeting Transcriber",
        "CFBundleIdentifier": "com.kylehsia.meeting-transcriber",
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "NSMicrophoneUsageDescription":
            "Meeting Transcriber needs microphone access to transcribe your voice during meetings.",
        "NSScreenCaptureDescription":
            "Meeting Transcriber needs screen recording permission to capture system audio (the remote side of online meetings).",
        "LSMinimumSystemVersion": "13.0",
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
