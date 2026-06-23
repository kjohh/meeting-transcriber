from __future__ import annotations

import atexit
import io
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from multiprocessing.connection import Connection
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from flask import Flask, Response, abort, jsonify, request, send_file
from groq import Groq

# ─── Config ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000

# Silence-aware chunking parameters (ported from lazy-take-notes).
# A chunk is triggered when EITHER the buffer hits CHUNK_DURATION (hard cap)
# OR the tail goes silent for PAUSE_DURATION while the body had speech (natural
# sentence boundary). OVERLAP samples are retained between chunks so the next
# transcription gets context.
CHUNK_DURATION = 25.0           # seconds — hard cap when speech is continuous
OVERLAP = 1.0                   # seconds — retained tail for context bleed
SILENCE_THRESHOLD = 0.005       # Per-frame voice-activity threshold. Low so
                                # individual frames of soft speech still
                                # register as active.
TRANSCRIBE_MIN_RMS = 0.012      # Whole-chunk RMS gate. Below this the chunk
                                # is mostly ambient noise — don't send to
                                # Whisper (it would hallucinate "字幕視聴",
                                # "ご視聴ありがとうございました", etc).
PAUSE_BODY_THRESHOLD = 0.012    # For pause-boundary detection: body must
                                # have real speech (not just ambient noise).
PAUSE_TAIL_THRESHOLD = 0.015    # Tail counts as silence when RMS is below
                                # this — tolerates ambient noise.
PAUSE_DURATION = 1.5            # seconds of silence required to trigger
MIN_SPEECH = 2.0                # don't trigger before this much speech buffered
VOICE_ACTIVITY_RATIO = 0.15     # min fraction of 100ms frames that must be
                                # "active" (above SILENCE_THRESHOLD) before we
                                # send a chunk to Whisper
MIN_ACTIVE_SPEECH = 0.6         # seconds — absolute floor of real speech. A
                                # short interjection ("對", "OK") sitting in a
                                # long quiet 25s cap chunk has a low active
                                # ratio but enough real speech to keep; the
                                # ratio gate alone would discard it.

PORT = 8765

# Consecutive failed system-audio (re)connects before we stop retrying and
# fall back to mic-only for the rest of the session.
MAX_SYS_RECONNECT = 5

# Groq chat model used for EN→ZH live translation. Swap if Groq retires it.
TRANSLATE_MODEL = "llama-3.3-70b-versatile"


def _is_frozen_bundle() -> bool:
    """True when running inside a py2app-built .app bundle (read-only)."""
    return getattr(sys, "frozen", False) or "RESOURCEPATH" in os.environ


def _resource_dir() -> str:
    """Where bundled read-only assets live (Swift binary, static files).
    In source mode this is the project root; in a py2app bundle it's
    Contents/Resources/."""
    if _is_frozen_bundle():
        return os.environ.get("RESOURCEPATH") or os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.abspath(__file__))


def _user_data_dir() -> str:
    """Where mutable per-user state goes (config, vocab).
    Source mode: project root (gitignored). Bundle: ~/Library/Application Support/Meeting Transcriber/."""
    if _is_frozen_bundle():
        d = os.path.expanduser("~/Library/Application Support/Meeting Transcriber")
        os.makedirs(d, exist_ok=True)
        return d
    return os.path.dirname(os.path.abspath(__file__))


BINARY = os.path.join(_resource_dir(), "native/.build/release/coreaudio_tap")
CONFIG_FILE = os.path.join(_user_data_dir(), ".config.json")
VOCAB_FILE = os.path.join(_user_data_dir(), ".vocab.local")

# Hugging Face model registry — borrowed from lazy-take-notes/hf_model_resolver.
# Models are cached in pywhispercpp's MODELS_DIR so this app shares the cache
# with lazy-take-notes (no double-download on machines that have both).
BREEZE_REPO = "alan314159/Breeze-ASR-25-whispercpp"
WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # alias: (hf_repo, filename)
    "large-v3-turbo-q8_0": (WHISPER_CPP_REPO, "ggml-large-v3-turbo-q8_0.bin"),
    "breeze-q8":           (BREEZE_REPO, "ggml-model-q8_0.bin"),
}

app = Flask(__name__, static_folder=os.path.join(_resource_dir(), "static"))


_ALLOWED_ORIGINS = frozenset([
    "",  # no-Origin requests come from pywebview / curl localhost / direct browser bar
    f"http://localhost:{PORT}",
    f"http://127.0.0.1:{PORT}",
])


@app.before_request
def _enforce_origin():
    """Block cross-origin requests from arbitrary websites.

    Flask binds localhost, so external attackers can't reach this — but any
    browser tab the user opens to a malicious page could `fetch('http://
    localhost:8765/start')` and silently drive the transcriber. The browser
    always sends an `Origin` header on cross-origin fetches, so checking it
    is sufficient to block that class of attack. Same-origin requests from
    the pywebview UI have an Origin of `http://localhost:8765`."""
    origin = request.headers.get("Origin", "")
    if origin not in _ALLOWED_ORIGINS:
        abort(403)

# ─── Global state ─────────────────────────────────────────────────────────────

_recording = False
_paused = False
_language = "auto"   # default: let Whisper detect per chunk
_backend = "local"   # "cloud" (Groq) or "local" (whisper.cpp). Default local
                     # to match onboarding's privacy-first preselection.
_chunk_worker_thread: Optional[threading.Thread] = None
_transcribe_consumer_thread: Optional[threading.Thread] = None
_mic_test_stream = None  # separate stream used by onboarding mic preview

# Recording-lifecycle mutations (start/stop/pause/clear, swift_proc, mic_stream,
# worker threads) all serialise through this lock so a double-click or a
# Flask threadpool race can't half-flip state.
_lifecycle_lock = threading.Lock()

# Bounded queue of audio chunks waiting to be transcribed. One consumer
# thread drains it serially — prevents Groq slow / network glitch from
# piling up transcribe threads, and prevents two local-whisper inferences
# from contending for CPU at the same time.
_transcribe_queue: "queue.Queue" = queue.Queue(maxsize=8)

# Set by /stop AFTER the chunk worker is joined (its final flush already
# enqueued). The consumer only checks this on an empty queue, so it always
# drains the last flushed chunk before exiting — fixes a dropped-tail race.
_consumer_should_exit = threading.Event()

# Download state (local backend only). Lock-protected so SSE clients can poll.
_download_state: dict = {"active": False, "percent": 0, "model": "", "error": ""}
_download_lock = threading.Lock()

# Whisper inference runs in a spawn subprocess so heavy compute doesn't pin
# P-cores via the parent GUI process's user-interactive QoS. One worker
# per recording session, started at /start and reaped at /stop.
_local_worker: Optional["LocalWhisperWorker"] = None
_local_worker_lock = threading.Lock()

# Prompt chain — last N transcript segments fed back as conditioning. Whisper's
# prompt window is ~224 tokens, so we cap by char count and keep only recent.
_prompt_chain: list[str] = []

# Whisper's `prompt` parameter is conditioning context (not instruction).
# Best practice for code-switched zh/en meetings: force language="zh" so the
# decoder stays in Chinese mode (which natively interleaves Latin tokens),
# and provide example sentences that demonstrate the expected style. The
# decoder mimics the style of the prompt, not its semantic content.
_BILINGUAL_PROMPT = (
    "以下是一段繁體中文與英文混合的工作會議逐字稿。"
    "我覺得這個方案的 timeline 有點趕,我們先 sync 一下。"
    "這個 feature 的 spec 還沒 finalize,等等 review 完再 follow up。"
    "OK,那我們 align 一下 priority,下週 update 進度。"
    "麻煩照之前的 format 處理,有問題隨時 ping 我。"
)
_sys_buf: list[np.ndarray] = []
_mic_buf: list[np.ndarray] = []
_buf_lock = threading.Lock()
# Transcript lines. Each entry: {"ts", "text", "tr", "tag"} — text is the
# original transcription, tr is the (optional) translation, tag is an optional
# source label (e.g. an uploaded filename).
_lines: list[dict] = []

# EN→ZH live translation toggle. Off by default — the user opts in per session.
# Mirrors the config "translate" flag; set on launch and via /translate.
_translate_enabled = False

# Live Groq rate-limit snapshot for translation, read from response headers.
# x-ratelimit-remaining-requests is the per-DAY window, so this is the
# authoritative "how many translations left today" — accurate across app
# restarts and other clients on the same key (it's the server's own count).
_translate_usage: dict = {"limit": 0, "used": 0, "remaining": 0, "reset": "", "exhausted": False}
_swift_proc: Optional[subprocess.Popen] = None
_sys_capture_thread: Optional[threading.Thread] = None
_mic_stream = None
_sse_clients: list[queue.Queue] = []
_transcribing = False  # True while waiting for Groq response
_mic_level: float = 0.0
_sys_level: float = 0.0
_level_tick = 0


def _broadcast(event_type: str, data):
    for q in list(_sse_clients):
        try:
            q.put_nowait({"type": event_type, "data": data})
        except queue.Full:
            pass


def _set_status(msg: str):
    _broadcast("status", msg)


def _append_line(text: str, tr: str = "", tag: str = "", ts: Optional[str] = None):
    line = {
        "ts": ts or datetime.now().strftime("%H:%M:%S"),
        "text": text,
        "tr": tr,
        "tag": tag,
    }
    _lines.append(line)
    _broadcast("transcript", line)


def _format_line(line: dict) -> str:
    """Flatten a transcript line to plain text for save / download. When a
    translation is present, the original and translation go on two lines."""
    head = f"[{line['ts']}]"
    if line.get("tag"):
        head += f" [{line['tag']}]"
    text = line.get("text", "")
    tr = line.get("tr", "")
    if tr:
        return f"{head} {text}\n    ↳ {tr}"
    return f"{head} {text}"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _read_config() -> dict:
    try:
        return json.loads(open(CONFIG_FILE).read())
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"WARN: config read failed: {e}", file=sys.stderr)
        return {}


def _write_config(cfg: dict):
    # Open with explicit 0o600 — config holds the Groq API key, must not
    # leak to other users on shared / misconfigured-umask machines.
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)


def load_api_key() -> str:
    if k := os.environ.get("GROQ_API_KEY", ""):
        return k
    return _read_config().get("groq_api_key", "")


def save_api_key(key: str):
    cfg = _read_config()
    cfg["groq_api_key"] = key
    _write_config(cfg)


def load_backend() -> str:
    return _read_config().get("backend", "local")


def save_backend(backend: str):
    cfg = _read_config()
    cfg["backend"] = backend
    _write_config(cfg)


def load_translate() -> bool:
    return bool(_read_config().get("translate", False))


def save_translate(value: bool):
    cfg = _read_config()
    cfg["translate"] = bool(value)
    _write_config(cfg)


def load_onboarding_completed() -> bool:
    return bool(_read_config().get("onboarding_completed", False))


def is_translocated() -> bool:
    """True if running from macOS App Translocation path.

    macOS sandboxes downloaded apps still in their original Downloads
    location by running them from a random read-only path under
    /private/var/folders/.../AppTranslocation/. The path can change on
    every launch, so TCC treats it as a different app each time and
    grants never stick — manifests as an endless re-authorisation loop.

    Fix is for the user: drag the .app into /Applications, which strips
    the quarantine bit and escapes translocation."""
    try:
        path = os.path.realpath(sys.executable)
        return ("AppTranslocation" in path) or (".translocation/" in path.lower())
    except Exception:
        return False


APP_VERSION = "0.1.10"  # Bumped on each release. Used to gate one-time
                       # `tccutil reset` of stale entries across upgrades.


def handle_version_change():
    """On version change, allow one fresh permission reset.

    Stale TCC entries (granted to the previous build's hash) make the new
    build silently fail capture even though System Settings shows toggle ON.
    `tccutil reset` clears them once per version — gated by config flag so
    repeated clicks of "立即重新授權" don't keep wiping fresh grants."""
    cfg = _read_config()
    if cfg.get("last_seen_version") != APP_VERSION:
        cfg["last_seen_version"] = APP_VERSION
        cfg["first_perm_trigger_done"] = False
        _write_config(cfg)


# NOTE: there is intentionally no startup "needs revalidation" check. It
# proved unreliable on ad-hoc-signed apps (CGPreflight + ground-truth spawn
# both false-positive). Permission problems are surfaced reactively instead:
# the sys-audio-warning banner fires off the Swift binary's own stderr (the
# only reliable signal), and its "立即重新授權" button opens the revalidation
# modal on demand. handle_version_change() still arms a one-shot tccutil reset
# for the first revalidation click after an upgrade.


def save_onboarding_completed(value: bool):
    cfg = _read_config()
    cfg["onboarding_completed"] = bool(value)
    _write_config(cfg)


# ─── Local whisper backend ────────────────────────────────────────────────────

def pick_local_model(language: str) -> str:
    """Choose best local model for a given language.

    - Force-Chinese → Breeze ASR 25 (繁中 fine-tuned)
    - Auto / English → large-v3-turbo-q8_0 (general, handles every language
      via Whisper's auto-detect)
    """
    if language == "zh":
        return "breeze-q8"
    return "large-v3-turbo-q8_0"


def model_local_path(alias: str) -> Optional[str]:
    """Return the cached on-disk path for *alias*, or None if not downloaded."""
    from pywhispercpp.constants import MODELS_DIR

    if alias not in MODEL_REGISTRY:
        return None
    repo, fname = MODEL_REGISTRY[alias]
    owner, repo_name = repo.split("/")
    # Match lazy-take-notes' layout so caches are shared.
    if alias.startswith("breeze"):
        candidate = os.path.join(MODELS_DIR, "breeze", fname)
    elif alias.startswith("large-v3-turbo"):
        # lazy-take-notes uses 'whisper-cpp', but pywhispercpp uses 'hf/owner__repo'
        # We check both for compatibility.
        candidates = [
            os.path.join(MODELS_DIR, "whisper-cpp", fname),
            os.path.join(MODELS_DIR, "hf", f"{owner}__{repo_name}", fname),
        ]
        return next((p for p in candidates if os.path.exists(p)), None)
    else:
        candidate = os.path.join(MODELS_DIR, "hf", f"{owner}__{repo_name}", fname)
    return candidate if os.path.exists(candidate) else None


def download_model(alias: str, on_progress=None) -> str:
    """Download *alias* from HF Hub into MODELS_DIR. Returns local path."""
    from huggingface_hub import hf_hub_download
    from pywhispercpp.constants import MODELS_DIR

    if alias not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model alias: {alias}")
    repo, fname = MODEL_REGISTRY[alias]
    owner, repo_name = repo.split("/")
    if alias.startswith("breeze"):
        cache_dir = os.path.join(MODELS_DIR, "breeze")
    else:
        cache_dir = os.path.join(MODELS_DIR, "hf", f"{owner}__{repo_name}")
    os.makedirs(cache_dir, exist_ok=True)

    kwargs = dict(repo_id=repo, filename=fname, local_dir=cache_dir)
    if on_progress:
        kwargs["tqdm_class"] = _make_progress_tqdm(on_progress)
    return hf_hub_download(**kwargs)


def _make_progress_tqdm(callback):
    """Build a tqdm-compatible class that pipes progress to *callback*(percent)."""
    class _Progress:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0) or 0
            self.n = 0
            if self.total > 0:
                callback(0)
        def update(self, n=1):
            self.n += n
            if self.total > 0:
                callback(min(int(self.n / self.total * 100), 100))
        def close(self): pass
        def set_description(self, *a, **k): pass
        def set_description_str(self, *a, **k): pass
        def refresh(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    return _Progress


def _whisper_subprocess_main(model_path: str, conn: Any) -> None:
    """Subprocess entry: load model once, then loop on transcription requests.

    Runs inside a `multiprocessing.spawn` child so whisper.cpp inference
    can't compete with the parent's WKWebView + Flask for the GIL, and so
    macOS's QoS / thermal scheduler isn't forced to keep inference on a
    P-core just because the parent is a user-interactive GUI app. Result:
    sustained transcription stops pinning P-cores and the Mac stops
    getting hot.

    Permanently redirects C-level stdout/stderr to /dev/null so whisper.cpp's
    fprintf() calls don't escape to the parent's Flask log.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        from pywhispercpp.model import Model
        model = Model(model_path, print_progress=False, print_realtime=False)
    except Exception as e:
        try:
            conn.send({"status": "error", "error": f"model load failed: {e}"})
        finally:
            conn.close()
        return

    conn.send({"status": "ready"})

    while True:
        try:
            req = conn.recv()
        except EOFError:
            break
        if req is None:
            break
        try:
            kw: dict = {}
            if req.get("language") and req["language"] != "auto":
                kw["language"] = req["language"]
            if req.get("prompt"):
                kw["initial_prompt"] = req["prompt"]
            audio = req["audio"].astype(np.float32)
            segments = model.transcribe(audio, **kw)
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
            conn.send({"status": "ok", "text": text})
        except Exception as e:
            conn.send({"status": "error", "error": str(e)})

    conn.close()


class LocalWhisperWorker:
    """Owns the whisper inference subprocess.

    Lifecycle: ``start()`` spawns a child, loads the model, blocks until ready.
    ``transcribe()`` is the synchronous request/response over the pipe.
    ``close()`` signals the child to exit and reaps it.

    Worker is **app-scoped, not session-scoped** — first /start with a
    given model path spawns and pays the ~1-3s model load; subsequent
    /start calls (same language → same model path) reuse the still-alive
    worker so the user doesn't wait for model load between sessions. A
    language change picks a different model alias, which we detect via
    model_path mismatch and respawn.
    """

    # Hard caps. Model load over Pipe handshake is fast (a few seconds at
    # most); transcription of a 25s chunk on M-series with q8 is under 10s
    # in the bad case. 120s leaves margin without hanging forever if the
    # child wedges.
    _LOAD_TIMEOUT = 120.0
    _TRANSCRIBE_TIMEOUT = 180.0

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path  # public — caller compares for reuse
        self._process: Optional[Any] = None
        self._conn: Optional[Connection] = None
        self._lock = threading.Lock()

    def is_alive(self) -> bool:
        """True if the subprocess is up and the pipe is healthy."""
        return (
            self._process is not None
            and self._process.is_alive()
            and self._conn is not None
        )

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._process = ctx.Process(
            target=_whisper_subprocess_main,
            args=(self.model_path, child_conn),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        self._conn = parent_conn

        if not self._conn.poll(timeout=self._LOAD_TIMEOUT):
            self.close()
            raise RuntimeError("whisper subprocess: timeout loading model")
        msg = self._conn.recv()
        if msg.get("status") != "ready":
            err = msg.get("error", "unknown")
            self.close()
            raise RuntimeError(f"whisper subprocess: {err}")

    def transcribe(self, audio: np.ndarray, language: str, prompt: str) -> str:
        if self._conn is None:
            raise RuntimeError("Worker not started")
        # Pipe is duplex but single send/recv pair — serialise so two
        # _transcribe calls (shouldn't happen with single consumer, but
        # defensive) can't interleave bytes on the same Connection.
        with self._lock:
            self._conn.send({"audio": audio, "language": language, "prompt": prompt})
            if not self._conn.poll(timeout=self._TRANSCRIBE_TIMEOUT):
                raise RuntimeError("whisper subprocess: transcribe timeout")
            result = self._conn.recv()
        if result.get("status") == "error":
            raise RuntimeError(result["error"])
        return result.get("text", "")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            self._process = None


def save_vocab(text: str):
    """Persist user vocabulary. Empty text deletes the file."""
    if not text.strip():
        try:
            os.unlink(VOCAB_FILE)
        except FileNotFoundError:
            pass
        return
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        f.write(text)


def read_vocab_raw() -> str:
    """Return the raw vocab file contents (for the editor UI)."""
    try:
        with open(VOCAB_FILE, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def cleanup_orphan_tempfiles():
    """Remove any mt_*.wav left in TMPDIR by previous crashed runs."""
    tmp_dir = tempfile.gettempdir()
    for name in os.listdir(tmp_dir):
        if name.startswith("mt_") and name.endswith(".wav"):
            try:
                os.unlink(os.path.join(tmp_dir, name))
            except OSError:
                pass


def _terminate_process(proc: subprocess.Popen, name: str, soft_timeout: float = 2.0):
    """Send SIGTERM, wait briefly, escalate to SIGKILL if still alive.

    `Popen.terminate()` is non-blocking and macOS lets a child trap or stall
    on SIGTERM — without a kill fallback the child becomes an orphan, which
    is exactly how the Swift coreaudio_tap binary kept holding a
    ScreenCaptureKit audio tap after the app appeared to stop recording."""
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=soft_timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"WARN: {name} ignored SIGTERM after {soft_timeout}s — sending SIGKILL", file=sys.stderr)
    try:
        proc.kill()
        proc.wait(timeout=1.0)
    except Exception:
        pass


# Keeps the Mac awake while recording. macOS idle display-sleep stops the
# ScreenCaptureKit stream and throttles the app, so a meeting left unattended
# stalls (UI still says recording, but capture/transcription has died). We hold
# a `caffeinate` assertion for the duration of a recording session instead.
_caffeinate_proc: Optional[subprocess.Popen] = None


def _start_caffeinate():
    """Prevent display + system idle sleep while recording."""
    global _caffeinate_proc
    if _caffeinate_proc is not None and _caffeinate_proc.poll() is None:
        return
    try:
        # -d: no display sleep, -i: no system idle sleep, -s: no system sleep
        # (AC only), -w <pid>: auto-exit if we crash without cleaning up.
        _caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-d", "-i", "-s", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"NOTE: caffeinate failed to start: {e}", file=sys.stderr)
        _caffeinate_proc = None


def _stop_caffeinate():
    """Release the keep-awake assertion."""
    global _caffeinate_proc
    if _caffeinate_proc is not None:
        _terminate_process(_caffeinate_proc, "caffeinate", soft_timeout=1.0)
        _caffeinate_proc = None


def reap_orphan_audio_taps():
    """Kill any leftover coreaudio_tap processes from prior crashed runs.

    macOS does NOT auto-clean the Swift audio-capture binary when the
    parent Python app crashes or gets force-quit before /stop runs. Every
    leftover process keeps ScreenCaptureKit + a CoreAudio tap alive and
    contributes to sustained CPU + heat. Run this at app startup.

    Matches by absolute path so we only kill our own binary, never a
    similarly-named user process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", BINARY],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return
    if result.returncode != 0:
        return  # pgrep returns 1 when no matches
    my_pid = os.getpid()
    killed = 0
    for line in result.stdout.strip().splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid == my_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
            try:
                os.kill(pid, 0)  # still alive?
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        print(f"INFO: reaped {killed} orphan coreaudio_tap process(es)", file=sys.stderr)


def load_vocab() -> str:
    """Read user vocabulary from .vocab.local. Returns a comma-joined hint string
    to be appended to Whisper's prompt, improving recognition of proper nouns
    that aren't in the model's training distribution (brand names, internal
    jargon, people)."""
    try:
        with open(VOCAB_FILE, encoding="utf-8") as f:
            words = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except FileNotFoundError:
        return ""
    if not words:
        return ""
    return "專有名詞:" + "、".join(words) + "。"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(_resource_dir(), "static/index.html"))


@app.route("/events")
def events():
    q: queue.Queue = queue.Queue(maxsize=200)
    _sse_clients.append(q)

    def generate():
        try:
            # send initial state on connect
            yield f"data: {json.dumps({'type':'init','key':load_api_key(),'lines':_lines,'recording':_recording,'paused':_paused,'language':_language,'backend':load_backend(),'translate':load_translate(),'translate_usage':dict(_translate_usage),'models':_model_status_payload(),'onboarding_completed':load_onboarding_completed(),'translocated':is_translocated()})}\n\n"
            while True:
                try:
                    event = q.get(timeout=25)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/key", methods=["POST"])
def route_key():
    """Validate against Groq before persisting — saving an invalid key
    silently is the worst UX failure mode here."""
    key = (request.json or {}).get("key", "").strip()
    if not key:
        return jsonify({"ok": False, "error": "金鑰是空的"})

    try:
        Groq(api_key=key).models.list()
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "401" in msg or "invalid" in low or "auth" in low:
            return jsonify({"ok": False, "error": "金鑰無效,請確認複製完整。"})
        if "connection" in low or "network" in low or "timeout" in low:
            return jsonify({"ok": False, "error": "無法連線到 Groq,請檢查網路。"})
        return jsonify({"ok": False, "error": f"驗證失敗: {msg[:120]}"})

    save_api_key(key)
    return jsonify({"ok": True})


def _ensure_local_worker(model_path: str) -> "LocalWhisperWorker":
    """Return a live whisper worker bound to *model_path*, (re)spawning as
    needed. App-scoped and reused across recording sessions AND uploads, so
    the ~1-3s model load is paid at most once per model. The caller handles
    the spawn exception (model-load failure)."""
    global _local_worker
    with _local_worker_lock:
        if (_local_worker is None
                or not _local_worker.is_alive()
                or _local_worker.model_path != model_path):
            if _local_worker is not None:
                _local_worker.close()
            _local_worker = LocalWhisperWorker(model_path)
            _local_worker.start()
        return _local_worker


@app.route("/start", methods=["POST"])
def route_start():
    global _recording, _paused, _swift_proc, _mic_stream, _language, _backend
    global _chunk_worker_thread, _transcribe_consumer_thread, _local_worker
    global _sys_capture_thread

    data = request.json or {}
    key = data.get("key", "").strip()
    language = data.get("language", "auto")
    backend = data.get("backend", "cloud")

    if backend == "cloud" and not key:
        return jsonify({"ok": False, "error": "No API key (required for cloud backend)"})
    if backend == "local":
        alias = pick_local_model(language)
        model_path = model_local_path(alias)
        if model_path is None:
            return jsonify({"ok": False, "error": f"Local model not downloaded: {alias}"})
    if not os.path.exists(BINARY):
        return jsonify({"ok": False, "error": "Binary missing — run: cd native && swift build -c release"})

    with _lifecycle_lock:
        if _recording:
            return jsonify({"ok": False, "error": "Already recording"})

        _language = language
        _backend = backend
        _recording = True
        _paused = False
        with _buf_lock:
            _sys_buf.clear()
            _mic_buf.clear()

        # Drain any stale items left in the transcribe queue from a previous
        # session (shouldn't happen if /stop joined properly, but defensive).
        while not _transcribe_queue.empty():
            try: _transcribe_queue.get_nowait()
            except queue.Empty: break
        _consumer_should_exit.clear()

        # Local backend: ensure a whisper subprocess is up with the right
        # model (app-scoped — reused across sessions so back-to-back
        # recordings don't re-pay the ~1-3s model load).
        if backend == "local":
            try:
                _set_status("Loading local whisper model…")
                _ensure_local_worker(model_path)
            except Exception as e:
                _recording = False
                _broadcast("state", {"recording": False, "paused": False})
                _set_status(f"⚠ 模型啟動失敗: {e}")
                return jsonify({"ok": False, "error": f"Whisper subprocess failed to start: {e}"})

        # Mic stream — its failure (denied mic permission) is what should abort
        # /start. Roll the half-started state back so _recording doesn't stay
        # True and wedge every future /start on "Already recording".
        try:
            _mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
                callback=_mic_cb, blocksize=int(SAMPLE_RATE * 0.1),
            )
            _mic_stream.start()
        except Exception as e:
            if _mic_stream:
                try: _mic_stream.close()
                except Exception: pass
                _mic_stream = None
            _recording = False
            _broadcast("state", {"recording": False, "paused": False})
            _set_status(f"啟動失敗:{e}")
            return jsonify({"ok": False, "error": f"無法啟動錄音:{e}"})

        # System audio runs under a supervisor thread that re-spawns the Swift
        # capture if macOS stops the ScreenCaptureKit stream mid-recording
        # (the monthly re-confirm / system stop). A transient drop self-heals
        # in ~1s instead of silently losing the rest of the meeting; only a
        # persistent failure (e.g. permission revoked) falls back to mic-only.
        _sys_capture_thread = threading.Thread(target=_sys_capture_supervisor, daemon=True)
        _sys_capture_thread.start()

        _broadcast("state", {"recording": True, "paused": False})
        _set_status("Starting system audio capture…")

        _transcribe_consumer_thread = threading.Thread(
            target=_transcribe_consumer, args=(key,), daemon=True,
        )
        _transcribe_consumer_thread.start()

        _chunk_worker_thread = threading.Thread(target=_chunk_worker, args=(key,), daemon=True)
        _chunk_worker_thread.start()

        # Keep the Mac awake for the whole session so an unattended meeting
        # doesn't stall when the screen sleeps.
        _start_caffeinate()

    return jsonify({"ok": True})


@app.route("/pause", methods=["POST"])
def route_pause():
    """Toggle pause. The actual buffer flush is handled by `_chunk_worker`
    which watches `_paused` — no separate flush thread, which was the source
    of an earlier race against the chunker.
    """
    global _paused
    with _lifecycle_lock:
        _paused = not _paused
        paused_now = _paused
    _broadcast("state", {"recording": _recording, "paused": paused_now})
    _set_status("暫停中" if paused_now else "錄音中…")
    return jsonify({"ok": True, "paused": paused_now})


def _flush_pending_audio():
    """Drain whatever's in the merged buffer into the transcribe queue.

    Called by `_chunk_worker` on pause entry and on session shutdown so
    audio that accumulated below the trigger threshold isn't lost. Voice
    activity gate in `_transcribe` filters out pure-silence flushes."""
    with _buf_lock:
        sa = np.concatenate(_sys_buf) if _sys_buf else np.array([], np.float32)
        ma = np.concatenate(_mic_buf) if _mic_buf else np.array([], np.float32)
        _sys_buf.clear()
        _mic_buf.clear()
    audio = _mix_buffers(sa, ma)
    if len(audio) > SAMPLE_RATE // 2:
        try:
            _transcribe_queue.put(audio, timeout=2)
        except queue.Full:
            print("WARN: transcribe queue full, dropping flush chunk", file=sys.stderr)


@app.route("/stop", methods=["POST"])
def route_stop():
    global _recording, _paused, _swift_proc, _mic_stream
    global _chunk_worker_thread, _transcribe_consumer_thread

    with _lifecycle_lock:
        if not _recording:
            return jsonify({"ok": True})  # idempotent
        _recording = False
        _paused = False

        _stop_caffeinate()  # let the Mac sleep again once recording ends

        if _mic_stream:
            _mic_stream.stop()
            _mic_stream.close()
            _mic_stream = None
        if _swift_proc:
            _terminate_process(_swift_proc, "coreaudio_tap")
            _swift_proc = None

        chunk_worker = _chunk_worker_thread
        consumer = _transcribe_consumer_thread
        _chunk_worker_thread = None
        _transcribe_consumer_thread = None

    # Join outside the lifecycle lock — chunk_worker needs to acquire _buf_lock
    # and the consumer needs to drain the queue, both can take seconds.
    # `newSession` (frontend) chains /stop → /clear, so /stop must return only
    # after every late _append_line has landed, or /clear will race them.
    if chunk_worker and chunk_worker.is_alive():
        chunk_worker.join(timeout=15)
    # Signal the consumer to exit only NOW — the chunk worker has flushed its
    # final chunk into the queue, so the consumer drains that (and anything
    # else queued) before it sees an empty queue + this flag and quits.
    _consumer_should_exit.set()
    if consumer and consumer.is_alive():
        consumer.join(timeout=30)

    # NOTE: do NOT close the whisper subprocess here. Worker is now
    # app-scoped — keeping it alive across sessions skips the 2-3s model
    # reload between recordings. atexit + the next /start's mismatch
    # check still handle cleanup on app exit / language switch.

    _broadcast("state", {"recording": False, "paused": False})
    _set_status(f"Stopped — {len(_lines)} segment(s) transcribed")
    return jsonify({"ok": True})


@app.route("/upload", methods=["POST"])
def route_upload():
    # Honour the current backend instead of always going to Groq — a
    # local-only / no-key user must be able to transcribe an uploaded file too,
    # which is the whole point of the privacy-preserving local mode.
    backend = load_backend()
    key = request.form.get("key", "").strip()
    language = request.form.get("language", "auto")
    if language not in ("auto", "zh", "en"):
        language = "auto"

    if backend == "cloud" and not key:
        return jsonify({"ok": False, "error": "雲端模式需要 Groq 金鑰(或切到本機模式)"})

    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "No file"})

    suffix = os.path.splitext(f.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(prefix="mt_", suffix=suffix, delete=False)
    f.save(tmp.name)
    fname = f.filename

    def _do():
        ts = datetime.now().strftime("%H:%M:%S")
        _set_status(f"Transcribing {fname}…")
        try:
            vocab = load_vocab()
            prompt = vocab
            if language == "zh":
                prompt = (vocab + " " + _BILINGUAL_PROMPT).strip()
            if backend == "local":
                text = _transcribe_file_local(tmp.name, language, prompt)
            else:
                text = _transcribe_file_cloud(tmp.name, fname, key, language, prompt)
            # Same post-processing as the live path: strip non-speech markers /
            # stock fillers, then trim Whisper repetition loops (a long file can
            # loop just like a live chunk).
            text = _drop_hallucinations((text or "").strip(), language)
            if text:
                text = _trim_repetition(text)
            tr = _maybe_translate(text, key) if text else ""
            _append_line(text, tr=tr, tag=fname, ts=ts)
            _set_status("Upload transcribed.")
        except Exception as e:
            _append_line(f"Upload error: {e}", tag=fname, ts=ts)
            _set_status("Upload failed.")
        finally:
            os.unlink(tmp.name)

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/clear", methods=["POST"])
def route_clear():
    # Serialise with /stop so a late _append_line from a still-draining
    # transcribe doesn't land into the cleared list.
    with _lifecycle_lock:
        _lines.clear()
    return jsonify({"ok": True})


@app.route("/vocab", methods=["GET"])
def route_vocab_get():
    return jsonify({"ok": True, "text": read_vocab_raw()})


@app.route("/vocab", methods=["POST"])
def route_vocab_post():
    text = (request.json or {}).get("text", "")
    save_vocab(text)
    return jsonify({"ok": True})


def _model_status_payload() -> dict:
    """Return per-model {alias: {downloaded: bool, path: str|None}}."""
    out = {}
    for alias in MODEL_REGISTRY:
        path = model_local_path(alias)
        out[alias] = {
            "downloaded": path is not None,
            "path": path,
        }
    return out


@app.route("/backend", methods=["GET"])
def route_backend_get():
    return jsonify({
        "ok": True,
        "backend": load_backend(),
        "models": _model_status_payload(),
        "download": dict(_download_state),
    })


@app.route("/backend", methods=["POST"])
def route_backend_post():
    backend = (request.json or {}).get("backend", "cloud")
    if backend not in ("cloud", "local"):
        return jsonify({"ok": False, "error": "Invalid backend"})
    save_backend(backend)
    return jsonify({"ok": True, "backend": backend})


@app.route("/translate", methods=["POST"])
def route_translate():
    """Toggle EN→ZH live translation. Takes effect from the next transcribed
    segment — safe to flip mid-session (it's post-transcription, doesn't touch
    the ASR pipeline)."""
    global _translate_enabled
    enabled = bool((request.json or {}).get("enabled", False))
    _translate_enabled = enabled
    save_translate(enabled)
    return jsonify({"ok": True, "enabled": enabled})


@app.route("/onboarding/complete", methods=["POST"])
def route_onboarding_complete():
    save_onboarding_completed(True)
    return jsonify({"ok": True})


@app.route("/model/download", methods=["POST"])
def route_model_download():
    """Kick off a background HF Hub download. Progress is exposed via /backend
    (the SSE stream also broadcasts 'download' events)."""
    alias = (request.json or {}).get("model", "")
    if alias not in MODEL_REGISTRY:
        return jsonify({"ok": False, "error": f"Unknown model: {alias}"})
    with _download_lock:
        if _download_state["active"]:
            return jsonify({"ok": False, "error": "Another download in progress"})
        _download_state.update({"active": True, "percent": 0, "model": alias, "error": ""})

    def _on_progress(percent: int):
        with _download_lock:
            _download_state["percent"] = percent
        _broadcast("download", dict(_download_state))

    def _do():
        try:
            download_model(alias, on_progress=_on_progress)
            with _download_lock:
                _download_state.update({"active": False, "percent": 100})
            _broadcast("download", dict(_download_state))
            _broadcast("models", _model_status_payload())
        except Exception as e:
            with _download_lock:
                _download_state.update({"active": False, "error": str(e)})
            _broadcast("download", dict(_download_state))

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/debug")
def route_debug():
    with _buf_lock:
        mic_samples = sum(len(x) for x in _mic_buf)
        sys_samples = sum(len(x) for x in _sys_buf)
    devices = []
    try:
        import sounddevice as _sd
        devices = [str(d) for d in _sd.query_devices()]
    except Exception as e:
        devices = [str(e)]
    return jsonify({
        "recording": _recording,
        "paused": _paused,
        "mic_buf_seconds": round(mic_samples / SAMPLE_RATE, 2),
        "sys_buf_seconds": round(sys_samples / SAMPLE_RATE, 2),
        "mic_level": round(_mic_level, 4),
        "sys_level": round(_sys_level, 4),
        "sse_clients": len(_sse_clients),
        "lines": len(_lines),
        "swift_running": _swift_proc is not None and _swift_proc.poll() is None,
        "input_devices": devices,
    })


@app.route("/transcript")
def route_transcript():
    content = "\n".join(_format_line(l) for l in _lines)
    fname = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    buf = io.BytesIO(content.encode("utf-8"))
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="text/plain")


# ─── Audio threads ────────────────────────────────────────────────────────────

def _read_sys_stdout(proc):
    """Pump *proc*'s stdout (float32 PCM from the Swift capture) into _sys_buf
    until EOF (proc died / was killed), we stop recording, OR we enter pause.
    Returns either way — the supervisor decides whether that was a stop, a
    pause (release the tap), or a drop to reconnect.

    The pipe is opened unbuffered (bufsize=0), so a single read can return a
    byte count that isn't a multiple of 4 — splitting a float32 sample across
    two reads. We carry the trailing odd bytes into the next read so
    np.frombuffer always gets a 4-byte-aligned buffer (an unaligned buffer
    raises ValueError, which previously killed this thread silently and
    stranded the Swift process holding the audio tap)."""
    global _sys_level
    chunk = int(SAMPLE_RATE * 0.1) * 4  # 100ms of float32
    carry = b""
    tick = 0
    while _recording and not _paused and proc.poll() is None:
        try:
            data = proc.stdout.read(chunk)
        except Exception:
            return
        if not data:
            return
        buf = carry + data
        usable = len(buf) - (len(buf) % 4)
        carry = buf[usable:]
        if usable <= 0:
            continue
        samples = np.frombuffer(buf[:usable], dtype=np.float32).copy()
        with _buf_lock:
            _sys_buf.append(samples)
        tick += 1
        if tick % 2 == 0:
            _sys_level = float(min(1.0, np.sqrt(np.mean(samples ** 2)) * 12))


def _watch_sys_stderr(proc, ready: threading.Event):
    """Watch *proc*'s stderr. READY → flag success + recording status. ERROR
    (ScreenCaptureKit's didStopWithError, or a start failure) → kill the proc
    so the stdout reader unblocks and the supervisor can reconnect (the Swift
    side leaves the process alive but the stream dead after didStopWithError)."""
    while proc.poll() is None:
        try:
            line = proc.stderr.readline().decode(errors="replace").strip()
        except Exception:
            return
        if not line:
            continue
        if line == "READY":
            ready.set()
            _set_status("錄音中…")
            _broadcast("sys_audio", {"ok": True})
        elif line.startswith("ERROR"):
            try:
                proc.terminate()
            except Exception:
                pass
            return


def _sys_capture_supervisor():
    """Own system-audio capture for the whole session and auto-reconnect on a
    mid-recording stream stop.

    macOS can stop an in-flight ScreenCaptureKit stream (the monthly
    screen-recording re-confirm, display sleep, or a system stop). Previously
    that ended capture for the rest of the session — the remote side of the
    meeting was silently lost. Here we re-spawn the Swift binary and resume,
    with backoff, so a transient stop self-heals. A run that never reaches
    READY is treated as a hard failure (permission likely missing) and we give
    up after MAX_SYS_RECONNECT tries rather than retry-storming."""
    global _swift_proc
    fails = 0  # consecutive (re)connects that never started capturing
    while _recording:
        # Paused: don't hold a capture process. ScreenCaptureKit + replayd
        # burn CPU for as long as the tap is open, so a "pause and walk away"
        # shouldn't keep them spinning. We re-spawn on resume.
        if _paused:
            time.sleep(0.2)
            continue
        try:
            proc = subprocess.Popen(
                [BINARY], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            )
        except Exception as e:
            _broadcast("sys_audio", {"ok": False, "msg": f"spawn failed: {e}"})
            _set_status("⚠ 系統音擷取程式啟動失敗,僅麥克風錄音中")
            return
        _swift_proc = proc
        ready = threading.Event()
        threading.Thread(target=_watch_sys_stderr, args=(proc, ready), daemon=True).start()

        _read_sys_stdout(proc)  # blocks until the stream ends, we stop, or we pause

        if not _recording:
            return  # normal /stop teardown — /stop already terminated the proc
        _terminate_process(proc, "coreaudio_tap")
        _swift_proc = None

        if _paused:
            # Released for a pause, not a failure. Loop back; the pause branch
            # above idles until resume, then re-spawns. Don't touch `fails`.
            continue

        fails = 0 if ready.is_set() else fails + 1
        if fails > MAX_SYS_RECONNECT:
            _broadcast("sys_audio", {"ok": False, "msg": "system audio stopped"})
            _set_status("⚠ 系統音抓不到 — 系統設定 → 隱私權 → 螢幕錄製 找到 Meeting Transcriber 並開啟")
            return
        # A transient stream stop self-heals on the next spawn (usually < 1s).
        # Don't raise the red "擷取失敗" banner for that — it flashes and
        # vanishes, which only alarms the user. Use the low-key status line;
        # the banner is reserved for the genuine give-up case (fails > MAX).
        _set_status("系統音短暫中斷,重新連線中…")
        time.sleep(min(2 ** fails, 8) if fails else 1)


def _mic_test_cb(indata, frames, time_info, status):
    """Mic callback used by onboarding's live-preview mode (not recording)."""
    samples = indata[:, 0]
    rms = float(min(1.0, np.sqrt(np.mean(samples ** 2)) * 12))
    _broadcast("mic_test_level", rms)


def _mic_cb(indata, frames, time_info, status):
    global _mic_level, _level_tick
    if not _paused:
        samples = indata[:, 0].copy()
        with _buf_lock:
            _mic_buf.append(samples)
        # broadcast level every ~200ms (2 × 100ms blocks)
        _level_tick += 1
        if _level_tick % 2 == 0:
            _mic_level = float(min(1.0, np.sqrt(np.mean(samples ** 2)) * 12))
            _broadcast("level", {"mic": _mic_level, "sys": _sys_level})


def _mix_buffers(sa: np.ndarray, ma: np.ndarray) -> np.ndarray:
    """Mix sys + mic, falling back to whichever is non-empty."""
    if len(sa) == 0 and len(ma) == 0:
        return np.array([], np.float32)
    if len(sa) == 0:
        return ma
    if len(ma) == 0:
        return sa
    n = min(len(sa), len(ma))
    return np.clip(sa[:n] + ma[:n], -1.0, 1.0)


def _should_trigger(buf_len: int) -> tuple[bool, str]:
    """Decide whether to fire a transcription based on the merged buffer length.

    Returns (trigger, reason). Reason is 'cap', 'pause', or '' (no trigger).
    """
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
    min_speech_samples = int(SAMPLE_RATE * MIN_SPEECH)

    if buf_len >= chunk_samples:
        return True, "cap"
    if buf_len >= min_speech_samples + pause_samples:
        return True, "pause-check"  # actual silence check needs the audio array
    return False, ""


def _is_pause_boundary(audio: np.ndarray) -> bool:
    """Check whether the tail of *audio* is silent and the body had speech —
    indicating a natural sentence boundary (lazy-take-notes' VAD heuristic)."""
    pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
    if len(audio) < pause_samples + int(SAMPLE_RATE * MIN_SPEECH):
        return False
    tail = audio[-pause_samples:]
    body = audio[:-pause_samples]
    tail_rms = float(np.sqrt(np.mean(tail ** 2)))
    body_rms = float(np.sqrt(np.mean(body ** 2)))
    return tail_rms < PAUSE_TAIL_THRESHOLD and body_rms >= PAUSE_BODY_THRESHOLD


def _build_prompt(vocab: str) -> str:
    """Compose the Whisper conditioning prompt: vocab + (zh-only) bilingual
    style demo + last segment from the prompt chain.

    The bilingual prime only goes in when language is forced zh — under auto
    it would bias the decoder toward Chinese tokens and turn pure-English
    chunks into garbled CJK. Under forced en it's irrelevant."""
    parts: list[str] = []
    if vocab:
        parts.append(vocab)
    if _language == "zh":
        parts.append(_BILINGUAL_PROMPT)
    if _prompt_chain:
        # Cap aggressively — long prompts make Whisper much more likely to
        # enter a repetition loop on tokens that appear in the prompt.
        parts.append(_prompt_chain[-1][-80:])
    return " ".join(parts).strip()


def _update_prompt_chain(text: str):
    """Append the latest transcript to the prompt chain. Keep just 1 entry —
    feeding more risks Whisper entering a repetition loop (it treats the prompt
    as a continuation context and can fixate on tokens it sees there)."""
    if not text:
        return
    _prompt_chain.clear()
    _prompt_chain.append(text)


_SENT_SPLIT_RE = re.compile(r'(?<=[。\.!?！？])\s*')

# Hallucinated dialogue labels Whisper emits during fast turn-taking
# ("余婷:", "Brad:", "OK，余婷：先講…"). Matches 1-4 CJK chars OR
# 1-12 latin chars followed by full-width / half-width colon, at sentence
# boundaries only (lookbehind on whitespace / punctuation / start).
_SPEAKER_LABEL_RE = re.compile(
    r'(?:^|(?<=[\s。\.\?\!,，、;；]))'
    r'(?:[一-鿿]{1,4}|[A-Za-z][A-Za-z\s]{0,11})[：:]\s*'
)
# Whisper non-speech markers: "*Piano*", "*Music*", "[Music]", "[Applause]".
# These never come from real meeting audio in our use case.
_NONSPEECH_MARK_RE = re.compile(r'\*[A-Za-z][A-Za-z\s]*\*|\[[A-Za-z][A-Za-z\s]*\]')
_CJK_RE = re.compile(r'[一-鿿]')

# CJK ideographs + kana + CJK/full-width punctuation. A forced-English session
# should never contain these — Whisper drifts into them on low-confidence audio
# (silence → 字幕/subtitle boilerplate), and once such a chunk leaks into the
# prompt chain it snowballs the rest of the meeting into Chinese gibberish.
_CJK_KANA_PUNCT_RE = re.compile(
    "[　-〿"   # CJK symbols & punctuation（、。「」etc.）
    "぀-ヿ"    # hiragana + katakana
    "一-鿿"    # CJK ideographs
    "＀-￯]+"  # full-width forms
)

# Stock Whisper "silence filler" phrases — what it emits on quiet/ambiguous
# audio under a zh lock instead of nothing. Normalised (lowercase, letters +
# spaces only). Used to drop ONLY these, so a real English sentence in a
# code-switching meeting ("let me share my screen") survives the zh lock.
_EN_HALLUCINATION = frozenset({
    "thank you", "thank you very much", "thank you so much", "thanks",
    "thanks for watching", "thank you for watching", "thanks for watching everyone",
    "please subscribe", "subscribe to my channel", "see you next time",
    "see you in the next video", "bye", "bye bye", "you", "the end",
    "im not sure", "i dont know", "okay", "ok", "mm", "mmm", "hmm", "yeah",
})


def _normalize_en(text: str) -> str:
    """Lowercase, strip to letters + single spaces — for matching against the
    boilerplate set regardless of punctuation/casing."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', ' ', text.lower())).strip()


def _strip_speaker_labels(text: str) -> str:
    """Strip hallucinated speaker labels before feeding to prompt chain.

    Whisper occasionally prepends dialogue labels for fast turn-taking
    sections. Once the format leaks into the chain, the decoder copies it
    forward and attributes everything to the same name. Stripping at the
    chain boundary breaks the propagation without altering what the user
    sees in the transcript."""
    return _SPEAKER_LABEL_RE.sub('', text).strip()


def _drop_hallucinations(text: str, language: str) -> str:
    """Filter Whisper hallucinations that survived the audio gates.

    Two patterns covered:
      1. Non-speech markers (``*Piano*``, ``[Music]``) — always dropped.
      2. Under ``language="zh"`` lock, a chunk with NO Chinese characters is
         suspicious — but only dropped when every sentence is a known stock
         filler phrase ("thank you", "thanks for watching", …). A genuine
         English sentence in a code-switching meeting is kept. Auto / en
         modes are left alone entirely."""
    text = _NONSPEECH_MARK_RE.sub('', text).strip()
    if not text:
        return ""
    if language == "zh" and not _CJK_RE.search(text):
        sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()] or [text]
        if all(_normalize_en(s) in _EN_HALLUCINATION for s in sents):
            return ""
    if language == "en" and _CJK_KANA_PUNCT_RE.search(text):
        # Forced-English but CJK/kana appeared → Whisper drift / hallucination.
        # Strip those runs; keep any real English, drop an all-CJK segment.
        # This also keeps the prompt chain clean, which is what stops a single
        # hallucinated chunk from snowballing the rest of the session.
        stripped = re.sub(r'\s+', ' ', _CJK_KANA_PUNCT_RE.sub(' ', text)).strip()
        return stripped if re.search(r'[A-Za-z]', stripped) else ""
    return text


def _trim_repetition(text: str, max_repeat: int = 2) -> str:
    """Trim consecutive sentence-level repetitions in *text*.

    Whisper's repetition-loop failure mode emits the same phrase N times in a
    row when it loses confidence (often primed by a prompt token). This keeps
    at most ``max_repeat`` consecutive copies of each sentence.
    """
    parts = [p for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return text
    out: list[str] = []
    prev = None
    count = 0
    for p in parts:
        norm = p.strip().lower()
        if norm == prev:
            count += 1
            if count > max_repeat:
                continue
        else:
            prev = norm
            count = 1
        out.append(p)
    return ' '.join(out)


def _is_repetition_loop(text: str) -> bool:
    """True if *text* contains 3+ consecutive identical sentences (the
    signature of a Whisper repetition loop). Used to suppress prompt-chain
    propagation so the next chunk isn't primed with poisonous context."""
    parts = [p.strip().lower() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < 3:
        return False
    prev, count = None, 0
    for p in parts:
        if p == prev:
            count += 1
            if count >= 3:
                return True
        else:
            prev, count = p, 1
    return False


def _dedup_boundary(text: str) -> str:
    """Drop the leading slice of *text* that repeats the tail of the previous
    transcript line.

    A hard-cap ('cap') cut keeps a 1s audio OVERLAP for context, so that
    second of speech is transcribed twice and the seam echoes a phrase. We
    find the longest suffix of the previous line that is a prefix of this one
    (char-level, so it works for spaceless Chinese too) and strip it. Requires
    a 5-char match so we don't clip incidental shared openers like "我覺得"."""
    if not _lines or not text.strip():
        return text
    prev_body = (_lines[-1].get("text") or "").strip()
    if not prev_body:
        return text
    cur = text.lstrip()
    tail = prev_body[-60:]                 # bounded search window
    maxk = min(len(tail), len(cur))
    for k in range(maxk, 4, -1):           # require >= 5 overlapping chars
        if tail[-k:].lower() == cur[:k].lower():
            return cur[k:].lstrip()
    return text


def _chunk_worker(api_key: str):
    """Silence-aware chunk loop (ported from lazy-take-notes).

    Triggers on either CHUNK_DURATION (hard cap) or PAUSE_DURATION of tail
    silence (natural sentence boundary). Pushes chunks to `_transcribe_queue`
    rather than spawning per-chunk threads — the consumer thread drains the
    queue serially.
    """
    overlap_samples = int(SAMPLE_RATE * OVERLAP)
    _prompt_chain.clear()
    last_pause_state = False

    while _recording:
        time.sleep(0.3)
        if _paused:
            # Edge: just entered pause → flush whatever's in the buffer so
            # audio below the trigger threshold isn't lost. Done by the
            # worker (not a separate thread) so it can't race the trigger.
            if not last_pause_state:
                _flush_pending_audio()
            last_pause_state = True
            continue
        last_pause_state = False

        mixed = None
        with _buf_lock:
            sys_len = sum(len(x) for x in _sys_buf)
            mic_len = sum(len(x) for x in _mic_buf)
            buf_len = sys_len if sys_len else mic_len

            trigger, reason = _should_trigger(buf_len)
            if not trigger:
                continue

            sa = np.concatenate(_sys_buf) if _sys_buf else np.array([], np.float32)
            ma = np.concatenate(_mic_buf) if _mic_buf else np.array([], np.float32)
            mixed = _mix_buffers(sa, ma)

            if reason == "pause-check" and not _is_pause_boundary(mixed):
                continue

            # 'cap' keeps an overlap tail for context across the cut.
            # 'pause-check' clears everything — the sentence already ended,
            # and a stale-speech tail would prime a phantom silent chunk
            # and Whisper would hallucinate.
            if reason == "cap":
                _sys_buf[:] = [sa[-overlap_samples:]] if len(sa) > overlap_samples else []
                _mic_buf[:] = [ma[-overlap_samples:]] if len(ma) > overlap_samples else []
            else:
                _sys_buf.clear()
                _mic_buf.clear()

        try:
            _transcribe_queue.put(mixed, timeout=2)
        except queue.Full:
            print("WARN: transcribe queue full, dropping chunk", file=sys.stderr)
            _set_status("⚠ Transcribe 跟不上速度,跳過一段")

    # Final flush after /stop sets _recording=False.
    _flush_pending_audio()


def _transcribe_consumer(api_key: str):
    """Single consumer that drains `_transcribe_queue` serially. Lives for
    the full recording session — bounded so Groq slowness / local CPU
    contention can't spawn unbounded threads.

    Exits only when /stop sets `_consumer_should_exit` AND the queue has been
    fully drained. /stop sets that flag after joining the chunk worker, so the
    worker's final flush is already enqueued and gets transcribed here before
    we quit — the trailing segment of a meeting is never dropped."""
    while True:
        try:
            audio = _transcribe_queue.get(timeout=0.5)
        except queue.Empty:
            if _consumer_should_exit.is_set():
                break
            continue
        try:
            _transcribe(audio, api_key)
        except Exception as e:
            print(f"transcribe failed: {e}", file=sys.stderr)
        finally:
            _transcribe_queue.task_done()


def _translate_text(text: str, api_key: str) -> str:
    """Translate *text* (English) to Traditional Chinese via a Groq chat model.

    Conditioned with the user's vocab so proper nouns survive. Kept to a single
    sentence/segment per call — no chat history — to stay fast and stateless."""
    sys_prompt = (
        "你是專業的會議口譯。把使用者輸入的英文逐字稿翻成自然、口語的台灣繁體中文。"
        "規則:只輸出譯文本身,不要任何解釋、引號或前綴;"
        "保留專有名詞、產品名、人名、英文縮寫(如 ETA、HBL、AMS)原樣不譯;"
        "若輸入本來就是中文,原樣輸出。"
    )
    vocab = load_vocab()
    if vocab:
        sys_prompt += f" 參考{vocab}"
    # with_raw_response so we can read the rate-limit headers (remaining quota)
    # alongside the parsed body.
    raw = Groq(api_key=api_key).chat.completions.with_raw_response.create(
        model=TRANSLATE_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    _update_translate_usage(raw.headers)
    chat = raw.parse()
    return (chat.choices[0].message.content or "").strip()


def _update_translate_usage(headers, exhausted: bool = False):
    """Snapshot Groq's per-day request quota from response headers and push it
    to the UI. x-ratelimit-remaining-requests is the per-day window."""
    try:
        limit = int(headers.get("x-ratelimit-limit-requests") or 0)
        remaining = int(headers.get("x-ratelimit-remaining-requests") or 0)
    except (TypeError, ValueError):
        limit = remaining = 0
    if limit:
        _translate_usage.update({
            "limit": limit,
            "remaining": remaining,
            "used": max(0, limit - remaining),
            "reset": headers.get("x-ratelimit-reset-requests") or "",
        })
    _translate_usage["exhausted"] = exhausted
    if exhausted:
        _translate_usage["remaining"] = 0
    _broadcast("translate_usage", dict(_translate_usage))


def _maybe_translate(text: str, api_key: str) -> str:
    """Return the EN→ZH translation of *text* when the toggle is on, else "".

    Translation always uses Groq (needs the API key) regardless of the ASR
    backend. Skips text that is already Chinese with no Latin letters. On
    failure returns a visible marker rather than dropping the line."""
    if not _translate_enabled or not api_key:
        return ""
    if _CJK_RE.search(text) and not re.search(r'[A-Za-z]', text):
        return ""
    try:
        return _translate_text(text, api_key)
    except Exception as e:
        # 429 = quota hit. Flag it so the UI shows "額度用盡 · reset in …".
        if getattr(e, "status_code", None) == 429:
            try:
                _update_translate_usage(e.response.headers, exhausted=True)
            except Exception:
                _translate_usage["exhausted"] = True
                _broadcast("translate_usage", dict(_translate_usage))
        print(f"translate failed: {e}", file=sys.stderr)
        return "⚠ 翻譯失敗"


def _transcribe(audio: np.ndarray, api_key: str):
    """Transcribe *audio*, dispatching to cloud (Groq) or local (whisper.cpp)
    based on `_backend`."""
    global _transcribing

    # Two-layer silence gate against Whisper hallucination on near-silent input:
    #   (1) Overall RMS too low → entire chunk is quiet.
    #   (2) Voice-activity ratio: fraction of 100ms frames that exceed the
    #       speech threshold. Whisper hallucinates on brief-speech-then-silence.
    # Thresholds tuned permissive (catch soft speech) — repetition_trim +
    # loop detection still handle the false-positive case.
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < TRANSCRIBE_MIN_RMS:
        _restore_idle_status()
        return

    frame = int(SAMPLE_RATE * 0.1)  # 100 ms frames
    if len(audio) >= frame * 4:
        frame_count = len(audio) // frame
        frames = audio[: frame_count * frame].reshape(frame_count, frame)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        active_frames = int(np.count_nonzero(frame_rms > SILENCE_THRESHOLD))
        active_ratio = active_frames / frame_count
        active_seconds = active_frames * 0.1
        # Drop only when the ratio is low AND there's little absolute speech.
        # A short interjection in a long quiet cap chunk has a low ratio but
        # enough real speech (>= MIN_ACTIVE_SPEECH) to keep.
        if active_ratio < VOICE_ACTIVITY_RATIO and active_seconds < MIN_ACTIVE_SPEECH:
            _restore_idle_status()
            return

    ts = datetime.now().strftime("%H:%M:%S")
    _transcribing = True
    _broadcast("transcribing", True)
    _set_status(f"Transcribing [{ts}]…")

    vocab = load_vocab()
    prompt = _build_prompt(vocab)

    try:
        if _backend == "local":
            text = _transcribe_local(audio, prompt)
        else:
            text = _transcribe_cloud(audio, api_key, prompt)

        text = _drop_hallucinations((text or "").strip(), _language)
        if text:
            cleaned = _dedup_boundary(_trim_repetition(text))
            if cleaned.strip():
                tr = _maybe_translate(cleaned, api_key)
                _append_line(cleaned, tr=tr, ts=ts)
                # If the result still shows a repetition loop after trimming,
                # the chunk was unreliable — don't poison the next chunk's
                # prompt chain. Also strip speaker labels before chaining so a
                # hallucinated "余婷:" prefix doesn't prime the next chunk.
                if not _is_repetition_loop(text):
                    _update_prompt_chain(_strip_speaker_labels(cleaned))
    except Exception as e:
        _append_line(f"Error: {e}", ts=ts)
    finally:
        _transcribing = False
        _broadcast("transcribing", False)

    _restore_idle_status()


def _restore_idle_status():
    """Set the status bar back to the right ambient state — depends on
    whether we're recording, paused, or idle. Called whenever a transient
    "Transcribing…" needs to clear."""
    if not _recording:
        return
    _set_status("暫停中" if _paused else "錄音中…")


def _transcribe_cloud(audio: np.ndarray, api_key: str, prompt: str) -> str:
    """Cloud backend — Groq Whisper API."""
    tmp = tempfile.NamedTemporaryFile(prefix="mt_", suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        kwargs: dict = dict(model="whisper-large-v3-turbo")
        if _language != "auto":
            kwargs["language"] = _language
        if prompt:
            kwargs["prompt"] = prompt

        with open(tmp.name, "rb") as f:
            kwargs["file"] = ("chunk.wav", f, "audio/wav")
            result = Groq(api_key=api_key).audio.transcriptions.create(**kwargs)
        return result.text
    finally:
        os.unlink(tmp.name)


def _transcribe_local(audio: np.ndarray, prompt: str) -> str:
    """Local backend — inference runs in the LocalWhisperWorker subprocess.

    See module-level comments on _local_worker for why this isn't in-process."""
    worker = _local_worker
    if worker is None:
        raise RuntimeError("Local worker not started — was /start called with backend=local?")
    return worker.transcribe(audio, _language, prompt)


# ─── Upload (whole-file) transcription ──────────────────────────────────────────
# Distinct from the streaming `_transcribe_*` helpers above: those take a numpy
# chunk from the live recording loop, these take an uploaded file on disk.

def _transcribe_file_cloud(path: str, fname: str, api_key: str,
                           language: str, prompt: str) -> str:
    """Cloud upload — hand the original file straight to Groq, which accepts
    common audio/video containers, so no local decode is needed."""
    kw: dict = dict(model="whisper-large-v3-turbo")
    if language != "auto":
        kw["language"] = language
    if prompt:
        kw["prompt"] = prompt
    with open(path, "rb") as af:
        kw["file"] = (fname, af)
        result = Groq(api_key=api_key).audio.transcriptions.create(**kw)
    return result.text


def _transcribe_file_local(path: str, language: str, prompt: str) -> str:
    """Local upload — decode the file to mono-16k numpy (pywhispercpp's static
    loader: WAV natively, other formats via ffmpeg) then run it through the
    same app-scoped whisper subprocess used for live recording."""
    alias = pick_local_model(language)
    model_path = model_local_path(alias)
    if model_path is None:
        raise RuntimeError(f"本機模型尚未下載:{alias}")
    from pywhispercpp.model import Model
    audio = Model._load_audio(path)
    worker = _ensure_local_worker(model_path)
    return worker.transcribe(audio, language, prompt)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # MUST be first — under py2app's frozen bundle, spawn re-enters the entry
    # point with a sentinel argv. freeze_support() detects that and runs the
    # child target then exits, instead of falling through to launch the full
    # app again. Required even though our worker target is a module-level
    # function, because spawn always re-executes the main module.
    mp.freeze_support()

    import webview

    cleanup_orphan_tempfiles()
    reap_orphan_audio_taps()
    handle_version_change()
    _translate_enabled = load_translate()  # restore the toggle from config

    # Belt-and-braces shutdown hook: if the user force-quits, closes the
    # webview window, or we hit an unhandled exception, atexit runs and
    # both the Swift audio binary and the whisper subprocess get reaped.
    # /stop already cleans these up on the happy path — this catches the
    # paths where /stop never fires.
    def _shutdown_cleanup():
        _stop_caffeinate()
        if _swift_proc is not None:
            _terminate_process(_swift_proc, "coreaudio_tap", soft_timeout=1.0)
        if _local_worker is not None:
            try:
                _local_worker.close()
            except Exception:
                pass
    atexit.register(_shutdown_cleanup)

    class JSAPI:
        """Bridge exposed to the webview JS as `window.pywebview.api`."""

        def trigger_screen_capture_permission(self):
            """Spawn the audio binary so macOS surfaces the screen-recording
            dialog, and keep it alive long enough for the user to read +
            click "Open System Settings" / "Allow".

            Earlier 1.5s timeout was too short — the binary would terminate
            before the user could respond, and the entry never got added to
            System Settings. We spawn in a background thread and let it run
            up to 20s, returning immediately so the JS API call doesn't
            block the UI. Polling picks up the grant state separately."""
            if not os.path.exists(BINARY):
                return {"ok": False, "error": "音訊擷取程式找不到 — bundle 可能損壞"}

            def _probe():
                try:
                    proc = subprocess.Popen(
                        [BINARY], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
                    # Live up to 20s. Exit early if the binary self-terminates
                    # (e.g. ScreenCaptureKit threw immediately because TCC said
                    # no after user declined).
                    for _ in range(200):
                        time.sleep(0.1)
                        if proc.poll() is not None:
                            break
                    try:
                        proc.terminate()
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception as e:
                    print(f"NOTE: permission probe error: {e}", file=sys.stderr)

            threading.Thread(target=_probe, daemon=True).start()
            return {"ok": True}

        def open_screen_recording_settings(self):
            """Open System Settings → Privacy → Screen Recording directly."""
            try:
                subprocess.Popen([
                    "open",
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
                ])
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        def reset_and_request_permission(self):
            """Trigger fresh macOS permission prompts.

            On the first click after an app version change, also `tccutil
            reset` stale entries from the previous build's signature —
            without that step macOS may silently apply the stale grant to
            our new hash and ScreenCaptureKit will still fail. The version
            gate (`first_perm_trigger_done`) ensures we only reset once per
            version, so repeat clicks don't wipe a grant the user just
            earned."""
            cfg = _read_config()
            if not cfg.get("first_perm_trigger_done", False):
                for service in ("ScreenCapture", "Microphone"):
                    try:
                        subprocess.run(
                            ["tccutil", "reset", service, "com.kylehsia.meeting-transcriber"],
                            check=False, timeout=5, capture_output=True,
                        )
                    except Exception as e:
                        print(f"NOTE: tccutil reset {service} failed: {e}", file=sys.stderr)
                cfg["first_perm_trigger_done"] = True
                _write_config(cfg)

            screen_result = self.trigger_screen_capture_permission()

            # Mic prompt — use AVFoundation's dedicated permission-request API
            # rather than implicitly via sd.InputStream. More reliable
            # because we don't have to keep a mic stream open + the API
            # is purpose-built for this prompt.
            def _trigger_mic_prompt():
                try:
                    from AVFoundation import AVCaptureDevice
                    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                        "soun", lambda granted: None,
                    )
                except Exception as e:
                    print(f"NOTE: mic prompt trigger failed: {e}", file=sys.stderr)

            threading.Thread(target=_trigger_mic_prompt, daemon=True).start()
            return screen_result

        def dismiss_revalidation(self):
            """Persistent escape hatch — user knows they have permissions even
            if our detection is reporting a false-negative. Persists a config
            flag (kept for forward-compat; the frontend also stops surfacing
            the sys-audio warning for the rest of the session)."""
            cfg = _read_config()
            cfg["revalidation_dismissed"] = True
            _write_config(cfg)
            return {"ok": True}

        def check_microphone_permission(self):
            """Return current microphone authorisation status without
            triggering the system prompt. Uses AVFoundation —
            `AVCaptureDevice.authorizationStatusForMediaType_('soun')`.

            Status codes:
              0 = NotDetermined (never asked)
              1 = Restricted (parental controls etc)
              2 = Denied
              3 = Authorized
            """
            try:
                from AVFoundation import AVCaptureDevice
                status = int(AVCaptureDevice.authorizationStatusForMediaType_("soun"))
                return {"ok": True, "granted": status == 3, "status": status}
            except Exception as e:
                return {"ok": False, "granted": False, "error": str(e)}

        def start_mic_test(self):
            """Open a mic stream so the user can see live waveform feedback.
            Triggers macOS' mic permission prompt the first time. Audio is
            NOT recorded — the callback only broadcasts level events."""
            global _mic_test_stream
            if _mic_test_stream is not None:
                return {"ok": True, "already_running": True}
            try:
                _mic_test_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype=np.float32,
                    callback=_mic_test_cb,
                    blocksize=int(SAMPLE_RATE * 0.1),
                )
                _mic_test_stream.start()
                return {"ok": True}
            except Exception as e:
                _mic_test_stream = None
                return {"ok": False, "error": str(e)}

        def stop_mic_test(self):
            """Close the onboarding mic-preview stream."""
            global _mic_test_stream
            if _mic_test_stream is not None:
                try:
                    _mic_test_stream.stop()
                    _mic_test_stream.close()
                except Exception:
                    pass
                _mic_test_stream = None
            return {"ok": True}

        def check_screen_capture_permission(self):
            """Query the current screen-capture permission state without
            triggering the permission dialog. Used by the onboarding modal to
            poll for completion after the user grants access in System Settings.

            Uses Quartz's `CGPreflightScreenCaptureAccess` — a documented
            preflight API that returns the current grant state without
            requesting it. Screen recording + system audio share the same
            TCC service (`kTCCServiceScreenCapture`), so this is accurate
            for our case.
            """
            try:
                from Quartz import CGPreflightScreenCaptureAccess
                granted = bool(CGPreflightScreenCaptureAccess())
                return {"ok": True, "granted": granted}
            except Exception as e:
                return {"ok": False, "granted": False, "error": str(e)}

        def save_transcript(self):
            """Show native macOS save dialog and write transcript to chosen path."""
            if not _lines:
                return {"ok": False, "error": "Nothing to save"}
            default_name = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            win = webview.windows[0] if webview.windows else None
            if not win:
                return {"ok": False, "error": "Window not ready"}
            result = win.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=default_name,
            )
            if not result:
                return {"ok": False, "cancelled": True}
            path = result if isinstance(result, str) else result[0]
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(_format_line(l) for l in _lines))
                return {"ok": True, "path": path}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    # Flask runs in a daemon thread; dies automatically when the window closes
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False, debug=False),
        daemon=True,
    ).start()
    time.sleep(0.6)  # let Flask start before opening the window

    window = webview.create_window(
        "Meeting Transcriber",
        f"http://localhost:{PORT}",
        width=1000,
        height=680,
        min_size=(720, 440),
        js_api=JSAPI(),
    )
    # NOTE: no `events.closing` handler. Opening a native confirmation dialog
    # from inside pywebview's closing callback re-enters the GUI event loop and
    # deadlocks the app on ⌘Q ("not responding"). atexit (_shutdown_cleanup)
    # still reaps the Swift binary + whisper subprocess on quit. If a
    # close-confirmation is wanted later, use create_window(confirm_close=True)
    # (pywebview's built-in, which handles this safely) rather than a custom
    # closing handler that opens a dialog.

    webview.start()
    # webview.start() blocks until window is closed — process exits cleanly
