from __future__ import annotations

import io
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from typing import Optional

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
SILENCE_THRESHOLD = 0.005       # Voice-activity gate — RMS below this in a
                                # given frame doesn't count as "speech". Low
                                # threshold catches soft-spoken input.
PAUSE_SILENCE_THRESHOLD = 0.015 # Pause-boundary detection — tail-RMS below
                                # this counts as "silence" for chunk cutting.
                                # Higher than SILENCE_THRESHOLD so ambient
                                # noise (fan / keyboard / room) reliably
                                # qualifies as a pause; otherwise the chunker
                                # only ever fires on the 25s hard cap.
PAUSE_DURATION = 1.5            # seconds of silence required to trigger
MIN_SPEECH = 2.0                # don't trigger before this much speech buffered
VOICE_ACTIVITY_RATIO = 0.15     # min fraction of 100ms frames that must be
                                # "active" (above SILENCE_THRESHOLD) before we
                                # send a chunk to Whisper (was 0.25)

PORT = 8765


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
_backend = "cloud"   # "cloud" (Groq) or "local" (whisper.cpp)
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

# Download state (local backend only). Lock-protected so SSE clients can poll.
_download_state: dict = {"active": False, "percent": 0, "model": "", "error": ""}
_download_lock = threading.Lock()

# Lazy-loaded local whisper model cache: {model_alias: pywhispercpp.Model}
_local_models: dict = {}
_local_models_lock = threading.Lock()

# Prompt chain — last N transcript segments fed back as conditioning. Whisper's
# prompt window is ~224 tokens, so we cap by char count and keep only recent.
_prompt_chain: list[str] = []

# Whisper's `prompt` parameter is conditioning context (not instruction).
# Best practice for code-switched zh/en meetings: force language="zh" so the
# decoder stays in Chinese mode (which natively interleaves Latin tokens),
# and provide example sentences that demonstrate the expected style. The
# decoder mimics the style of the prompt, not its semantic content.
_BILINGUAL_PROMPT = (
    "以下是一段繁體中文與英文混合的設計討論會議逐字稿。"
    "我覺得這個 component 的 hover state 太 subtle 了。"
    "我們等等 review 一下 design system 的 token。"
    "Loki 的 Modal 要用 ModalHeader 包標題,不要直接放 ModalBody。"
    "請 follow 既有的 pattern,不要自己造輪子。"
)
_sys_buf: list[np.ndarray] = []
_mic_buf: list[np.ndarray] = []
_buf_lock = threading.Lock()
_lines: list[str] = []
_swift_proc: Optional[subprocess.Popen] = None
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


def _append_line(line: str):
    _lines.append(line)
    _broadcast("transcript", line)


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
    return _read_config().get("backend", "cloud")


def save_backend(backend: str):
    cfg = _read_config()
    cfg["backend"] = backend
    _write_config(cfg)


def load_onboarding_completed() -> bool:
    return bool(_read_config().get("onboarding_completed", False))


def needs_revalidation() -> bool:
    """True iff onboarding was completed but at least one permission is now
    missing. This is the classic 'user updated the app' signature — macOS
    TCC binds grants to code signature hash, so a new build looks like an
    unauthorised app even though the bundle id is unchanged.

    For first-time users (onboarding incomplete) we return False — onboarding
    will handle permissions itself."""
    if not load_onboarding_completed():
        return False
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        screen_ok = bool(CGPreflightScreenCaptureAccess())
    except Exception:
        return False
    try:
        from AVFoundation import AVCaptureDevice
        mic_ok = int(AVCaptureDevice.authorizationStatusForMediaType_("soun")) == 3
    except Exception:
        mic_ok = True
    return not (screen_ok and mic_ok)


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


def get_local_model(alias: str):
    """Return a loaded pywhispercpp.Model for *alias*. Loads on first use."""
    with _local_models_lock:
        if alias in _local_models:
            return _local_models[alias]

    path = model_local_path(alias)
    if path is None:
        raise FileNotFoundError(f"Model {alias} not downloaded")

    from pywhispercpp.model import Model
    # Suppress whisper.cpp's C-level stdout so it doesn't pollute Flask logs.
    import contextlib

    @contextlib.contextmanager
    def _quiet():
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull, 1); os.dup2(devnull, 2); yield
        finally:
            os.dup2(old_out, 1); os.dup2(old_err, 2)
            os.close(devnull); os.close(old_out); os.close(old_err)

    with _quiet():
        m = Model(path, print_progress=False, print_realtime=False)

    with _local_models_lock:
        _local_models[alias] = m
    return m


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
            yield f"data: {json.dumps({'type':'init','key':load_api_key(),'lines':_lines,'recording':_recording,'paused':_paused,'language':_language,'backend':load_backend(),'models':_model_status_payload(),'onboarding_completed':load_onboarding_completed(),'needs_revalidation':needs_revalidation()})}\n\n"
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


@app.route("/start", methods=["POST"])
def route_start():
    global _recording, _paused, _swift_proc, _mic_stream, _language, _backend
    global _chunk_worker_thread, _transcribe_consumer_thread

    data = request.json or {}
    key = data.get("key", "").strip()
    language = data.get("language", "auto")
    backend = data.get("backend", "cloud")

    if backend == "cloud" and not key:
        return jsonify({"ok": False, "error": "No API key (required for cloud backend)"})
    if backend == "local":
        alias = pick_local_model(language)
        if model_local_path(alias) is None:
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

        _broadcast("state", {"recording": True, "paused": False})
        _set_status("Starting system audio capture…")

        _swift_proc = subprocess.Popen([BINARY], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=_read_sys_audio, daemon=True).start()
        threading.Thread(target=_watch_stderr, daemon=True).start()

        _mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            callback=_mic_cb, blocksize=int(SAMPLE_RATE * 0.1),
        )
        _mic_stream.start()

        _transcribe_consumer_thread = threading.Thread(
            target=_transcribe_consumer, args=(key,), daemon=True,
        )
        _transcribe_consumer_thread.start()

        _chunk_worker_thread = threading.Thread(target=_chunk_worker, args=(key,), daemon=True)
        _chunk_worker_thread.start()

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

        if _mic_stream:
            _mic_stream.stop()
            _mic_stream.close()
            _mic_stream = None
        if _swift_proc:
            _swift_proc.terminate()
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
    if consumer and consumer.is_alive():
        # Sentinel wakes the consumer; it then exits the loop.
        try: _transcribe_queue.put_nowait(None)
        except queue.Full: pass
        consumer.join(timeout=30)

    _broadcast("state", {"recording": False, "paused": False})
    _set_status(f"Stopped — {len(_lines)} segment(s) transcribed")
    return jsonify({"ok": True})


@app.route("/upload", methods=["POST"])
def route_upload():
    key = request.form.get("key", "").strip()
    if not key:
        return jsonify({"ok": False, "error": "No API key"})

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
            client = Groq(api_key=key)
            vocab = load_vocab()
            with open(tmp.name, "rb") as af:
                kw = dict(model="whisper-large-v3-turbo", file=(fname, af))
                if _language != "auto":
                    kw["language"] = _language
                prompt_parts = []
                if vocab:
                    prompt_parts.append(vocab)
                if _language == "zh":
                    prompt_parts.append(_BILINGUAL_PROMPT)
                if prompt_parts:
                    kw["prompt"] = " ".join(prompt_parts).strip()
                result = client.audio.transcriptions.create(**kw)
            _append_line(f"[{ts}] [{fname}]\n{result.text.strip()}")
            _set_status("Upload transcribed.")
        except Exception as e:
            _append_line(f"[{ts}] Upload error: {e}")
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
    """Return per-model {alias: {downloaded: bool, size_mb: int|None, path: str|None}}."""
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
    content = "\n".join(_lines)
    fname = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    buf = io.BytesIO(content.encode("utf-8"))
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="text/plain")


# ─── Audio threads ────────────────────────────────────────────────────────────

def _read_sys_audio():
    global _sys_level
    chunk = int(SAMPLE_RATE * 0.1) * 4  # 100ms of float32
    tick = 0
    while _recording and _swift_proc and _swift_proc.poll() is None:
        data = _swift_proc.stdout.read(chunk)
        if not data:
            # Swift binary EOF'd mid-recording — stream collapsed without
            # process exit. Tell the user and stop the busy-loop.
            _broadcast("sys_audio", {"ok": False, "msg": "system audio stream stopped"})
            _set_status("⚠ 系統音擷取中斷,僅麥克風錄音中")
            return
        if not _paused:
            samples = np.frombuffer(data, dtype=np.float32).copy()
            with _buf_lock:
                _sys_buf.append(samples)
            tick += 1
            if tick % 2 == 0:
                _sys_level = float(min(1.0, np.sqrt(np.mean(samples ** 2)) * 12))


def _watch_stderr():
    while _swift_proc and _swift_proc.poll() is None:
        line = _swift_proc.stderr.readline().decode().strip()
        if line == "READY":
            _set_status("錄音中…")
            _broadcast("sys_audio", {"ok": True})
        elif line.startswith("ERROR"):
            # ScreenCaptureKit failed — usually means the user denied or
            # never granted screen recording permission. Mic is independent
            # and may still be working, but the user must know we can't
            # capture the remote side of meetings until they grant it.
            _set_status(
                "⚠ 系統音抓不到 — 系統設定 → 隱私權 → 螢幕錄製 找到 Meeting Transcriber 並開啟"
            )
            _broadcast("sys_audio", {"ok": False, "msg": line})


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
    # PAUSE_SILENCE_THRESHOLD (looser) for "is the tail silent" — ambient
    # noise still counts as silence. SILENCE_THRESHOLD (stricter) for "does
    # the body have actual speech" — soft speech still counts.
    return tail_rms < PAUSE_SILENCE_THRESHOLD and body_rms >= SILENCE_THRESHOLD


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
    contention can't spawn unbounded threads."""
    while True:
        try:
            audio = _transcribe_queue.get(timeout=0.5)
        except queue.Empty:
            # Drain done + recording stopped → exit. Otherwise keep waiting.
            if not _recording:
                break
            continue
        if audio is None:  # /stop sentinel
            break
        try:
            _transcribe(audio, api_key)
        except Exception as e:
            print(f"transcribe failed: {e}", file=sys.stderr)
        finally:
            _transcribe_queue.task_done()


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
    if rms < SILENCE_THRESHOLD:
        _restore_idle_status()
        return

    frame = int(SAMPLE_RATE * 0.1)  # 100 ms frames
    if len(audio) >= frame * 4:
        frame_count = len(audio) // frame
        frames = audio[: frame_count * frame].reshape(frame_count, frame)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        active_ratio = float(np.mean(frame_rms > SILENCE_THRESHOLD))
        if active_ratio < VOICE_ACTIVITY_RATIO:
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

        text = (text or "").strip()
        if text:
            cleaned = _trim_repetition(text)
            _append_line(f"[{ts}] {cleaned}")
            # If the result still shows a repetition loop after trimming, the
            # chunk was unreliable — don't poison the next chunk's prompt chain.
            if not _is_repetition_loop(text):
                _update_prompt_chain(cleaned)
    except Exception as e:
        _append_line(f"[{ts}] Error: {e}")
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
    """Local backend — pywhispercpp on-device inference."""
    alias = pick_local_model(_language)
    model = get_local_model(alias)

    kwargs: dict = {}
    if _language != "auto":
        kwargs["language"] = _language
    if prompt:
        kwargs["initial_prompt"] = prompt

    # pywhispercpp expects float32 numpy at 16kHz mono — which is exactly our
    # internal format, so no resampling needed.
    segments = model.transcribe(audio.astype(np.float32), **kwargs)
    return " ".join(s.text.strip() for s in segments if s.text.strip())


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webview

    cleanup_orphan_tempfiles()

    class JSAPI:
        """Bridge exposed to the webview JS as `window.pywebview.api`."""

        def trigger_screen_capture_permission(self):
            """Briefly spawn the audio binary to invoke macOS' permission prompt.

            macOS only shows the screen recording dialog when an app actually
            attempts to use ScreenCaptureKit — so we trigger a real capture.

            We don't treat an early exit as failure: ScreenCaptureKit also
            exits immediately when the user hasn't granted permission yet
            (which is the normal state during onboarding, before they click
            through the System Settings flow). The polling loop is what
            actually confirms grant — this trigger just kicks the dialog.
            """
            if not os.path.exists(BINARY):
                return {"ok": False, "error": "音訊擷取程式找不到 — bundle 可能損壞"}
            try:
                proc = subprocess.Popen(
                    [BINARY],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                time.sleep(1.5)
                # Log stderr for dev debugging (Intel Mac / arch mismatch /
                # binary corruption) but don't surface it to the user — they
                # may legitimately still be working through System Settings.
                if proc.poll() is not None:
                    err = (proc.stderr.read() or b"").decode(errors="replace").strip()
                    if err:
                        print(f"NOTE: coreaudio_tap exited during permission probe: {err[:200]}", file=sys.stderr)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

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
            """Trigger fresh macOS permission prompts for the new build's
            signature. We deliberately DO NOT `tccutil reset` here — doing
            so would wipe a grant the user just earned the first time they
            went through this flow, causing a loop where the System Settings
            toggle silently flips off on every click. macOS itself prompts
            when the new code-signature hash has no TCC entry, so we just
            need to attempt the protected operations.

            Both screen and mic prompts are triggered together so the user
            can resolve everything in one System Settings trip rather than
            seeing the modal twice."""
            # Screen: spawn the audio binary briefly
            screen_result = self.trigger_screen_capture_permission()
            # Mic: opening an sd.InputStream is what triggers macOS' mic
            # permission dialog. The stream auto-closes a moment later.
            self.start_mic_test()
            threading.Timer(1.0, self.stop_mic_test).start()
            return screen_result

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
                    f.write("\n".join(_lines))
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
    webview.start()
    # webview.start() blocks until window is closed — process exits cleanly
