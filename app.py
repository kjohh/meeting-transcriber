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
from collections import Counter, deque
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

# ── Mixing / signal conditioning ──
MIX_TARGET_RMS = 0.09     # per-stream loudness target before summing mic+sys
MIX_GAIN_MAX = 4.0        # cap on per-stream boost (don't over-amplify a stream)
MIX_GAIN_EMA = 0.3        # smoothing on the per-stream gain across chunks (0..1,
                          # lower = smoother) so levels don't pump between chunks
MIC_GATE_GRACE = 3.0      # seconds. Mic samples are withheld until system audio
                          # is READY (so mic[0] and sys[0] share a wall-clock
                          # start and aren't summed with a ~1s skew); after this
                          # grace we record mic anyway (mic-only fallback / slow
                          # ScreenCaptureKit warmup).

# ── Chunk worker ──
CHUNK_POLL = 0.1          # chunk-loop poll interval (matches the 100ms block
                          # cadence; tighter pause-boundary detection than 0.3s)
PAUSE_REL_DROP = 0.45     # tail counts as a pause when its RMS drops below this
                          # fraction of the body RMS — relative test so a raised
                          # noise floor (typing/fan) doesn't defeat boundary cuts

# ── Prompt conditioning ──
PROMPT_CHAIN_CHARS = 160  # chars of prior transcript carried as cross-chunk
                          # context (was 80 — too short to keep continuity)
PROMPT_CHAIN_KEEP = 2     # number of recent segments kept in the chain
PROMPT_MAX_CHARS = 330    # hard cap on the assembled prompt (~224 Whisper tokens
                          # for CJK-heavy text); the style prime is placed LAST so
                          # Whisper's tail-keep truncation never drops it

# ── Local whisper.cpp decoding ──
LOCAL_BEAM_SIZE = 5       # beam search (default greedy beam=-1 is lower accuracy);
                          # needs params_sampling_strategy=1 on the Model to engage

# ── Model-confidence post-filter (layered AFTER the energy gate) ──
# Conservative: drop a segment only on a STRONG combined low-confidence signal;
# flag (amber, non-destructive) on a softer signal. No eval harness yet, so these
# err toward keeping real speech — tune against real meeting clips before raising.
LOCAL_PROB_DROP = 0.25       # local: drop a segment whose geo-mean prob is below this
LOCAL_PROB_LOWCONF = 0.55    # local: flag the line when min kept prob is below this
CLOUD_NSP_DROP = 0.85        # cloud: drop a segment when no_speech_prob > this AND…
CLOUD_LOGPROB_DROP = -1.0    # …avg_logprob < this (both must hold)
CLOUD_NSP_LOWCONF = 0.6      # cloud: flag the line when max kept no_speech_prob > this…
CLOUD_LOGPROB_LOWCONF = -0.9 # …or min kept avg_logprob < this

# ── Cloud (Groq) model selection ──
CLOUD_MODEL_LIVE = "whisper-large-v3-turbo"  # live default: speed helps serial display
CLOUD_MODEL_BATCH = "whisper-large-v3"       # upload/batch: no latency cost → use accuracy


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
_language = "zh"     # default: force Chinese. This user's meetings are zh/en
                     # code-switched; the Chinese decoder natively interleaves
                     # Latin tokens, and forcing zh turns on the bilingual style
                     # prime + the zh hallucination/script-lock cleanup. "auto"
                     # gave no prime and flip-flopped language per chunk.
_backend = "local"   # "cloud" (Groq) or "local" (whisper.cpp). Default local
                     # to match onboarding's privacy-first preselection.

# Set when system audio reaches READY; gates mic-buffer appends so the two
# streams start at the same wall-clock instant (see _mic_cb / MIC_GATE_GRACE).
_sys_ready = threading.Event()
_mic_gate_deadline = 0.0  # monotonic time after which mic records even if sys isn't ready

# Per-stream mix gains, EMA-smoothed across chunks so loudness matching doesn't
# pump between chunks. Reset at session start.
_mix_gain_sys = 1.0
_mix_gain_mic = 1.0
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
# Kept deliberately short: it competes with vocab + the prompt chain for
# Whisper's ~224-token prompt window, and a bloated prime crowds the chain out
# (or gets truncated itself). Two representative code-switched sentences are
# enough to set the style.
_BILINGUAL_PROMPT = (
    "以下是繁體中文與英文混合的工作會議逐字稿。"
    "這個 feature 的 spec 還沒 finalize,等 review 完再 follow up,我們先 sync 一下 priority。"
)
# Rolling source-line context for translation (separate from the ASR prompt
# chain). Gives the translator cross-sentence context so pronouns / continuations
# resolve across chunk boundaries.
_translate_ctx: "deque[str]" = deque(maxlen=2)
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


def _append_line(text: str, tr: str = "", tag: str = "", ts: Optional[str] = None,
                 conf: str = ""):
    line = {
        "ts": ts or datetime.now().strftime("%H:%M:%S"),
        "text": text,
        "tr": tr,
        "tag": tag,
        "conf": conf,  # "" normal, "low" = model was unsure (rendered de-emphasized)
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


def load_cloud_model() -> str:
    """Groq model for the LIVE cloud path. Default = turbo (its speed helps the
    serial chunk display); a user on hard bilingual meetings can opt into the
    higher-accuracy non-turbo large-v3 in Settings. Upload/batch always uses
    large-v3 (no latency cost) regardless of this setting."""
    m = _read_config().get("cloud_model", CLOUD_MODEL_LIVE)
    return m if m in (CLOUD_MODEL_LIVE, CLOUD_MODEL_BATCH) else CLOUD_MODEL_LIVE


def save_cloud_model(model: str):
    cfg = _read_config()
    cfg["cloud_model"] = model if model in (CLOUD_MODEL_LIVE, CLOUD_MODEL_BATCH) else CLOUD_MODEL_LIVE
    _write_config(cfg)


# Local model used when language is forced "zh". Breeze is Traditional-Chinese
# fine-tuned (best for pure Chinese); large-v3-turbo is general multilingual and
# tends to handle dense zh/en code-switching + English proper nouns better.
# Default Breeze (unchanged); exposed so the user can A/B for their meetings.
_ZH_MODELS = ("breeze-q8", "large-v3-turbo-q8_0")


def load_zh_model() -> str:
    m = _read_config().get("zh_model", "breeze-q8")
    return m if m in _ZH_MODELS else "breeze-q8"


def save_zh_model(model: str):
    cfg = _read_config()
    cfg["zh_model"] = model if model in _ZH_MODELS else "breeze-q8"
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


APP_VERSION = "0.1.12"  # Bumped on each release. Used to gate one-time
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

    - Force-Chinese → user-selectable (default Breeze ASR 25, 繁中 fine-tuned;
      can switch to large-v3-turbo for heavy zh/en code-switching)
    - Auto / English → large-v3-turbo-q8_0 (general, handles every language
      via Whisper's auto-detect)
    """
    if language == "zh":
        return load_zh_model()
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
        # params_sampling_strategy=1 = beam search (greedy=0 is the default and
        # lower-accuracy); the beam_search dict per-call tunes beam_size.
        model = Model(model_path, params_sampling_strategy=1,
                      print_progress=False, print_realtime=False)
    except Exception as e:
        try:
            conn.send({"status": "error", "error": f"model load failed: {e}"})
        finally:
            conn.close()
        return

    conn.send({"status": "ready"})

    def _run(audio, kw, with_extras):
        """Transcribe; with_extras adds beam/prob/NST params that some builds may
        reject — caller retries without them on failure."""
        k = dict(kw)
        if with_extras:
            # NOTE: the whisper.cpp beam_search param REQUIRES both keys —
            # {"beam_size": N} alone raises KeyError 'patience' on every call,
            # which would make this whole branch fall back to greedy and also
            # silently disable suppress_nst + extract_probability.
            k["beam_search"] = {"beam_size": LOCAL_BEAM_SIZE, "patience": -1.0}
            k["suppress_nst"] = True   # suppress non-speech tokens ([Music], 字幕…) at decode
            k["extract_probability"] = True
        return model.transcribe(audio, **k)

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
            try:
                segments = _run(audio, kw, with_extras=True)
            except Exception:
                # A whisper.cpp build that rejects the extra kwargs — degrade
                # gracefully to a plain decode rather than failing the chunk.
                segments = _run(audio, kw, with_extras=False)

            kept, probs = [], []
            for s in segments:
                t = s.text.strip()
                if not t:
                    continue
                p = getattr(s, "probability", float("nan"))
                # Drop only a segment with a VALID, strongly-low probability —
                # NaN (prob not computed) is always kept.
                if isinstance(p, float) and not np.isnan(p) and p < LOCAL_PROB_DROP:
                    continue
                kept.append(t)
                if isinstance(p, float) and not np.isnan(p):
                    probs.append(p)
            text = " ".join(kept)
            conf = "low" if probs and min(probs) < LOCAL_PROB_LOWCONF else ""
            conn.send({"status": "ok", "text": text, "conf": conf})
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

    def transcribe(self, audio: np.ndarray, language: str, prompt: str) -> tuple[str, str]:
        """Returns (text, conf) where conf is "low" when the model was unsure."""
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
        return result.get("text", ""), result.get("conf", "")

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


def _vocab_lines() -> list[str]:
    try:
        with open(VOCAB_FILE, encoding="utf-8") as f:
            return [ln.strip() for ln in f
                    if ln.strip() and not ln.strip().startswith("#")]
    except FileNotFoundError:
        return []


def load_vocab() -> str:
    """Read user vocabulary from .vocab.local. Returns a comma-joined hint string
    appended to Whisper's prompt, improving recognition of proper nouns that
    aren't in the model's training distribution (brand names, internal jargon,
    people).

    Lines of the form ``wrong=>right`` are correction pairs, NOT prompt hints —
    they're applied as a post-hoc find/replace on transcript output instead of
    eating the limited prompt window (see apply_vocab_corrections). Only the
    plain terms go into the prompt."""
    words = [ln for ln in _vocab_lines() if "=>" not in ln]
    if not words:
        return ""
    return "專有名詞:" + "、".join(words) + "。"


def load_vocab_corrections() -> list[tuple[str, str]]:
    """Parse ``wrong=>right`` lines from .vocab.local into (wrong, right) pairs.
    Deterministic post-transcription correction — fixes a recurring mishear for
    BOTH backends without priming Whisper into a repetition loop."""
    pairs = []
    for ln in _vocab_lines():
        if "=>" in ln:
            wrong, _, right = ln.partition("=>")
            wrong, right = wrong.strip(), right.strip()
            if wrong:
                pairs.append((wrong, right))
    return pairs


def apply_vocab_corrections(text: str) -> str:
    """Apply user-defined wrong=>right replacements to *text* (case-insensitive
    for Latin runs; exact for CJK). Intentionally simple literal replace — no
    fuzzy auto-matching, which would risk silently over-correcting real speech
    when there's no eval harness to catch regressions."""
    if not text:
        return text
    for wrong, right in load_vocab_corrections():
        if re.search(r'[A-Za-z]', wrong):
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
        else:
            text = text.replace(wrong, right)
    return text


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
            yield f"data: {json.dumps({'type':'init','key':load_api_key(),'lines':_lines,'recording':_recording,'paused':_paused,'language':_language,'backend':load_backend(),'cloud_model':load_cloud_model(),'zh_model':load_zh_model(),'translate':load_translate(),'translate_usage':dict(_translate_usage),'models':_model_status_payload(),'onboarding_completed':load_onboarding_completed(),'translocated':is_translocated()})}\n\n"
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
    global _sys_capture_thread, _mic_gate_deadline, _mix_gain_sys, _mix_gain_mic

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
        # Arm the mic-buffer gate: withhold mic until system audio is READY (so
        # the two streams start aligned), with a grace fallback. If screen
        # recording isn't granted, system audio will never arrive — open the
        # gate now so a mic-only session records from t=0 (no 3s dead start).
        _sys_ready.clear()
        _mic_gate_deadline = time.monotonic() + MIC_GATE_GRACE
        if not _screen_capture_granted():
            _sys_ready.set()
        _mix_gain_sys = 1.0
        _mix_gain_mic = 1.0
        _translate_ctx.clear()
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
        # NOTE: we deliberately do NOT re-arm the mic gate on resume. _sys_ready
        # stays set from the first READY, so mic records immediately on resume.
        # Re-arming would discard ~MIC_GATE_GRACE of the user's own speech every
        # resume just to re-align against the system-audio warmup — not worth it
        # (the brief re-skew is absorbed by _mix_buffers' zero-pad-to-longest).
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
        _queue_chunk(audio, "flush", "flush chunk")


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
        _set_status(f"轉錄中:{fname} — 長檔可能需要幾分鐘…")
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
            # stock fillers, collapse + trim repetition loops (a long file can
            # loop just like a live chunk), then apply vocab corrections.
            text = _drop_hallucinations((text or "").strip(), language)
            if text:
                text = apply_vocab_corrections(_trim_repetition(_collapse_runs(text)))
            tr = _maybe_translate(text, key) if text else ""
            _append_line(text, tr=tr, tag=fname, ts=ts)
            _set_status("檔案轉錄完成。")
        except Exception as e:
            _append_line(f"Upload error: {e}", tag=fname, ts=ts)
            _set_status("檔案轉錄失敗。")
        finally:
            os.unlink(tmp.name)
            _broadcast("upload_done", True)

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/clear", methods=["POST"])
def route_clear():
    # Serialise with /stop so a late _append_line from a still-draining
    # transcribe doesn't land into the cleared list. Also hold _buf_lock so the
    # clear can't interleave between /line's bounds-check and its indexed write.
    with _lifecycle_lock:
        with _buf_lock:
            _lines.clear()
    _translate_ctx.clear()
    return jsonify({"ok": True})


@app.route("/line", methods=["POST"])
def route_line():
    """Edit a transcript line in place. The DOM edit alone wouldn't survive into
    the saved file (save serializes from _lines), so we write the correction
    back into _lines under the buffer lock."""
    data = request.json or {}
    try:
        idx = int(data.get("index", -1))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad index"})
    with _buf_lock:
        if not (0 <= idx < len(_lines)):
            return jsonify({"ok": False, "error": "index out of range"})
        if "text" in data:
            _lines[idx]["text"] = str(data["text"])
        if "tr" in data:
            _lines[idx]["tr"] = str(data["tr"])
    return jsonify({"ok": True})


@app.route("/cloud_model", methods=["GET", "POST"])
def route_cloud_model():
    """Get/set the LIVE cloud model (turbo vs the higher-accuracy large-v3)."""
    if request.method == "POST":
        model = (request.json or {}).get("model", CLOUD_MODEL_LIVE)
        save_cloud_model(model)
        return jsonify({"ok": True, "cloud_model": load_cloud_model()})
    return jsonify({"ok": True, "cloud_model": load_cloud_model()})


@app.route("/zh_model", methods=["GET", "POST"])
def route_zh_model():
    """Get/set the local model used when language is forced 'zh' (Breeze vs
    large-v3-turbo). Takes effect on the next /start (the worker respawns for
    the new model path)."""
    if request.method == "POST":
        save_zh_model((request.json or {}).get("model", "breeze-q8"))
    return jsonify({"ok": True, "zh_model": load_zh_model(),
                    "models": _model_status_payload()})


@app.route("/language", methods=["POST"])
def route_language():
    """Set the ASR language live. Cloud reads _language per chunk so a
    mid-session switch takes effect on the next chunk; local would need a worker
    respawn across the breeze-q8 ↔ large-v3 boundary, so a mid-session local
    switch is rejected (the UI keeps the selector locked for local while
    recording). When idle, this just stages the value for the next /start."""
    global _language
    lang = (request.json or {}).get("language", "auto")
    if lang not in ("auto", "zh", "en"):
        return jsonify({"ok": False, "error": "bad language"})
    if _recording and _backend == "local" and lang != _language:
        return jsonify({"ok": False, "error": "本機模式錄音中無法切換語言(需重新開始)"})
    _language = lang
    return jsonify({"ok": True, "language": lang})


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
            break
        if not data:
            break
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

    # Recover the in-flight pipe tail — the blocking 100ms reads leave up to
    # ~100ms of remote speech buffered when we exit on /stop. Non-blocking drain.
    try:
        os.set_blocking(proc.stdout.fileno(), False)
        rest = proc.stdout.read() or b""
        buf = carry + rest
        usable = len(buf) - (len(buf) % 4)
        if usable > 0:
            samples = np.frombuffer(buf[:usable], dtype=np.float32).copy()
            with _buf_lock:
                _sys_buf.append(samples)
    except Exception:
        pass


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
            _sys_ready.set()  # opens the mic-buffer gate (aligned start)
            _set_status("錄音中…")
            _broadcast("sys_audio", {"ok": True})
        elif line.startswith("ERROR"):
            try:
                proc.terminate()
            except Exception:
                pass
            return


def _screen_capture_granted() -> bool:
    """Preflight the screen-recording grant WITHOUT triggering a prompt. Used to
    decide whether to even arm the mic-alignment gate — if system audio can't be
    captured, mic must record from t=0 (mic-only fallback)."""
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:
        return True  # can't tell → assume granted; the grace deadline still backstops


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
            _sys_ready.set()  # mic-only fallback: open the mic gate immediately
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
            _sys_ready.set()  # give up on system audio → don't keep withholding mic
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
    # Withhold mic until system audio is READY so mic[0] and sys[0] share a
    # wall-clock start (the ScreenCaptureKit warmup made the mic lead system by
    # ~1s, and _mix_buffers overlays by index — a constant skew that smeared
    # overlapping speech). After MIC_GATE_GRACE we record anyway: mic-only
    # fallback (sys permission denied) or an unusually slow warmup.
    gate_open = _sys_ready.is_set() or time.monotonic() >= _mic_gate_deadline
    if not _paused and gate_open:
        samples = indata[:, 0].copy()
        with _buf_lock:
            _mic_buf.append(samples)
        # broadcast level every ~200ms (2 × 100ms blocks)
        _level_tick += 1
        if _level_tick % 2 == 0:
            _mic_level = float(min(1.0, np.sqrt(np.mean(samples ** 2)) * 12))
            _broadcast("level", {"mic": _mic_level, "sys": _sys_level})


def _mix_buffers(sa: np.ndarray, ma: np.ndarray, update_gain: bool = True) -> np.ndarray:
    """Mix sys + mic, falling back to whichever is non-empty.

    update_gain=False computes the loudness-match gains for THIS mix without
    advancing the EMA state — used by the chunk worker's pause-boundary probe,
    which runs every poll. Advancing the EMA there (10×/s) would defeat the
    cross-chunk smoothing; the EMA is updated exactly once per emitted chunk."""
    if len(sa) == 0 and len(ma) == 0:
        return np.array([], np.float32)
    if len(sa) == 0:
        return ma
    if len(ma) == 0:
        return sa
    global _mix_gain_sys, _mix_gain_mic
    # Per-stream, noise-gated loudness match so the quieter party (close-mic user
    # vs post-gain remote, or vice-versa) isn't summed near the noise floor and
    # gated out. Gain is EMA-smoothed across emitted chunks so levels don't pump.
    gs = _stream_gain(sa, _mix_gain_sys)
    gm = _stream_gain(ma, _mix_gain_mic)
    if update_gain:
        _mix_gain_sys, _mix_gain_mic = gs, gm
    # Zero-pad to the LONGER stream (don't truncate to the shorter — that
    # silently dropped the tail of whichever stream was ahead).
    n = max(len(sa), len(ma))
    out = np.zeros(n, np.float32)
    if len(sa):
        out[: len(sa)] += sa * gs
    if len(ma):
        out[: len(ma)] += ma * gm
    # Peak-limit instead of hard clip — removes the dual-talk clipping distortion
    # np.clip introduced exactly when both parties spoke at once.
    peak = float(np.max(np.abs(out))) if n else 0.0
    if peak > 1.0:
        out /= peak
    return out


def _stream_gain(stream: np.ndarray, prev_gain: float) -> float:
    """EMA-smoothed gain bringing *stream* toward MIX_TARGET_RMS, but only when
    the stream actually contains SPEECH. An idle-but-noisy channel (typing/fan
    above the RMS floor but no voice) is left at unity so its noise isn't boosted
    up to target and smeared into the active speaker."""
    if len(stream) == 0:
        return prev_gain
    rms = float(np.sqrt(np.mean(stream ** 2)))
    # Voice-activity check: fraction of 100ms frames above the speech threshold.
    frame = int(SAMPLE_RATE * 0.1)
    active_ratio = 0.0
    if len(stream) >= frame:
        nf = len(stream) // frame
        fr = np.sqrt(np.mean(stream[: nf * frame].reshape(nf, frame) ** 2, axis=1))
        active_ratio = float(np.count_nonzero(fr > SILENCE_THRESHOLD)) / nf
    if rms > TRANSCRIBE_MIN_RMS and active_ratio >= VOICE_ACTIVITY_RATIO:
        target = min(MIX_TARGET_RMS / rms, MIX_GAIN_MAX)
        target = max(target, 0.25)  # don't over-attenuate a loud stream
    else:
        target = 1.0
    return prev_gain * (1 - MIX_GAIN_EMA) + target * MIX_GAIN_EMA


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
    if body_rms < PAUSE_BODY_THRESHOLD:
        return False  # no real speech in the body — not a sentence boundary
    # Boundary = the tail went quiet RELATIVE to the body. A purely absolute
    # threshold fails under steady background noise (typing/fan) that keeps the
    # floor elevated, so the speaker's pause never registers and the chunk runs
    # to the 25s hard cap. The relative drop catches "the speaker stopped" even
    # when the noise floor is high; the absolute threshold still short-circuits
    # genuinely quiet tails.
    return tail_rms < PAUSE_TAIL_THRESHOLD or tail_rms < body_rms * PAUSE_REL_DROP


def _build_prompt(vocab: str) -> str:
    """Compose the Whisper conditioning prompt from chain + vocab + (zh-only)
    bilingual style demo.

    Whisper keeps the LAST ~224 tokens of the prompt and truncates the front, so
    parts are ordered lowest-value-first: chain (droppable continuity) → vocab →
    style prime LAST, so the prime — the thing that makes code-switching work —
    always survives. The whole prompt is then capped at PROMPT_MAX_CHARS.

    The bilingual prime only goes in when language is forced zh — under auto it
    would bias the decoder toward Chinese tokens and turn pure-English chunks
    into garbled CJK. Under forced en it's irrelevant."""
    parts: list[str] = []
    if _prompt_chain:
        # Carry recent transcript for cross-chunk continuity (recurring proper
        # nouns, ongoing topic). _is_repetition_loop / _has_runaway_repeat gate
        # the chain, so a longer window doesn't reintroduce loops.
        chain = " ".join(_prompt_chain)[-PROMPT_CHAIN_CHARS:]
        parts.append(chain)
    if vocab:
        parts.append(vocab)
    if _language == "zh":
        parts.append(_BILINGUAL_PROMPT)
    prompt = " ".join(parts).strip()
    if len(prompt) > PROMPT_MAX_CHARS:
        # Keep the TAIL (prime + vocab survive; oldest chain drops first).
        prompt = prompt[-PROMPT_MAX_CHARS:]
    if os.environ.get("MT_DEBUG_PROMPT"):
        print(f"DEBUG prompt len={len(prompt)} chars", file=sys.stderr)
    return prompt


def _update_prompt_chain(text: str):
    """Append the latest transcript to the prompt chain, keeping the most recent
    PROMPT_CHAIN_KEEP segments. _build_prompt char-caps the assembled chain, and
    the repetition-loop guards suppress chaining of looped chunks, so a couple of
    segments is safe and preserves more cross-chunk continuity than just one."""
    if not text:
        return
    _prompt_chain.append(text)
    del _prompt_chain[:-PROMPT_CHAIN_KEEP]


_SENT_SPLIT_RE = re.compile(r'(?<=[。\.!?！？])\s*')
# Clause-level split — ALSO breaks on commas. Whisper's zh repetition loops are
# usually comma-separated run-ons with no sentence-ending punctuation
# ("…用途,…用途,…用途,"), so the sentence splitter saw them as ONE sentence and
# the repetition guards missed them entirely. Used by the repetition detectors.
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[。\.!?！？,，、;；])\s*')
_CLAUSE_PUNCT = "。.!?！？,，、;；:：…「」\"' \t"


def _norm_clause(c: str) -> str:
    """Normalize a clause for repetition comparison: lowercase + strip
    surrounding punctuation/space, so "…用途," and "…用途" (Whisper punctuates
    inconsistently) compare equal."""
    return c.strip().strip(_CLAUSE_PUNCT).lower()

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
# Unambiguous YouTube/video boilerplate — never a real meeting utterance, so
# safe to drop in ANY mode even as a lone chunk. The rest of _EN_HALLUCINATION
# ("thank you", "ok", "i dont know", "yeah"…) are also REAL meeting utterances,
# so they're only dropped under the zh lock (bare English is itself suspicious
# there) or when the whole chunk is multiple filler sentences (a hallucinated
# run). A deliberate standalone "OK." / "Yeah." in en/auto survives.
_EN_FILLER_PHRASES = frozenset({
    "thanks for watching", "thank you for watching", "thanks for watching everyone",
    "please subscribe", "subscribe to my channel", "see you next time",
    "see you in the next video", "the end",
})


def _normalize_en(text: str) -> str:
    """Lowercase, strip to letters + single spaces — for matching against the
    boilerplate set regardless of punctuation/casing."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', ' ', text.lower())).strip()


# Chinese / Japanese "silence filler" hallucinations Whisper emits on quiet or
# off-script audio — the zh/jp analogue of _EN_HALLUCINATION. These are almost
# always run-ons with NO sentence punctuation, so they're matched by normalized
# substring (not per-sentence). Both Traditional and Simplified variants, since
# Whisper may emit either. Seeded with near-zero-collision boilerplate.
_ZH_HALLUCINATION = frozenset({
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目", "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "明鏡與點點欄目", "明镜与点点栏目", "明鏡新聞", "明镜新闻",
    "請訂閱我的頻道", "请订阅我的频道", "記得訂閱", "记得订阅",
    "請訂閱按贊", "请订阅按赞", "點贊訂閱", "点赞订阅",
    "謝謝大家收看", "谢谢大家收看", "謝謝觀看", "谢谢观看", "謝謝你的觀看", "谢谢你的观看",
    "字幕by", "字幕志願者", "字幕志愿者", "由社群提供的字幕", "由社区提供的字幕",
    "ご視聴ありがとうございました", "本字幕由",
})
# Drop a chunk as a zh/jp hallucination only when the boilerplate DOMINATES the
# normalized text by at least this fraction — a meeting that merely mentions
# "字幕" or "訂閱" in a longer sentence is kept.
_ZH_HALLUCINATION_COVERAGE = 0.6


def _normalize_zh(text: str) -> str:
    """Strip to CJK ideographs + kana + latin + digits (drop all whitespace and
    punctuation) so the run-on boilerplate phrases match regardless of how
    Whisper punctuated them."""
    return re.sub(r'[^぀-ヿ一-鿿A-Za-z0-9]', '', text)


def _strip_speaker_labels(text: str) -> str:
    """Strip hallucinated speaker labels before feeding to prompt chain.

    Whisper occasionally prepends dialogue labels for fast turn-taking
    sections. Once the format leaks into the chain, the decoder copies it
    forward and attributes everything to the same name. Stripping at the
    chain boundary breaks the propagation without altering what the user
    sees in the transcript."""
    return _SPEAKER_LABEL_RE.sub('', text).strip()


# Display-safe speaker-label stripper. The chain stripper above is aggressive
# (fine — the chain is never shown). For the VISIBLE transcript we must not eat
# legitimate openers ("ETA:", "結論:", "11:30"), so this is tighter: only a
# name-like token (single Capitalized Latin word, or 2-4 CJK chars) followed by a
# colon and real text, and never an allowlisted meeting opener.
_LABEL_ALLOWLIST = frozenset({
    "eta", "etd", "ata", "atd", "hbl", "mbl", "ok", "note", "action", "actions",
    "decision", "todo", "re", "ps", "fyi", "q", "a", "ref", "po", "so", "inv",
    "結論", "重點", "行動", "決議", "決定", "摘要", "總結", "待辦", "問題",
    "回覆", "補充", "提醒", "註", "例", "附註", "備註", "結語",
})
_DISPLAY_LABEL_RE = re.compile(
    r'(?:^|(?<=[\s。\.\?\!,，、;；]))'
    r'(?P<lbl>[A-Z][A-Za-z]{1,11}|[一-鿿]{2,4})[：:](?=\s*\S)'
)


def _strip_display_labels(text: str) -> str:
    """Strip hallucinated speaker-name prefixes from the VISIBLE transcript,
    while preserving real openers via _LABEL_ALLOWLIST."""
    def _repl(m):
        lbl = m.group("lbl")
        if lbl.lower() in _LABEL_ALLOWLIST or lbl in _LABEL_ALLOWLIST:
            return m.group(0)
        return ""
    return _DISPLAY_LABEL_RE.sub(_repl, text).strip()


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

    # (A) English stock-filler drop — runs in EVERY mode now (was zh-only, hence
    # a no-op under the default). Under the zh lock, or when the whole chunk is a
    # run of filler sentences, the full ambiguous set drops; in en/auto a single
    # sentence only drops if it's unambiguous video boilerplate — so a real lone
    # "OK." / "I don't know." survives.
    sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()] or [text]
    drop_set = _EN_HALLUCINATION if (language == "zh" or len(sents) > 1) else _EN_FILLER_PHRASES
    if all(_normalize_en(s) in drop_set for s in sents):
        return ""

    # (B) Chinese/Japanese boilerplate — match the normalized WHOLE text (these
    # are punctuation-free run-ons). Drop only when the boilerplate dominates the
    # chunk (>= coverage), so a passing mention of 字幕/訂閱 in a real sentence
    # is kept. Runs in every mode for the same anti-snowball reason as (C).
    norm = _normalize_zh(text)
    if norm:
        for h in _ZH_HALLUCINATION:
            if h in norm and len(h) >= _ZH_HALLUCINATION_COVERAGE * len(norm):
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
    at most ``max_repeat`` consecutive copies of each clause. Splits on clauses
    (commas too), since zh loops are comma-separated with no sentence enders.
    """
    parts = [p for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return text
    out: list[str] = []
    prev = None
    count = 0
    trimmed = False
    for p in parts:
        norm = _norm_clause(p)
        if norm == prev:
            count += 1
            if count > max_repeat:
                trimmed = True
                continue
        else:
            prev = norm
            count = 1
        out.append(p)
    if not trimmed:
        return text  # nothing repeated → leave original formatting untouched
    return ' '.join(out)


def _is_repetition_loop(text: str) -> bool:
    """True if *text* contains 3+ consecutive identical clauses (the signature
    of a Whisper repetition loop). Used to suppress prompt-chain propagation so
    the next chunk isn't primed with poisonous context."""
    parts = [_norm_clause(p) for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]
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


def _is_loop_hallucination(text: str) -> bool:
    """True when ONE substantial clause dominates the chunk and repeats 3+ times
    — the signature of a pure repetition-loop hallucination (e.g. the classic
    "…用途,…用途,…用途,…用途,…用途" on near-silent audio). Such a chunk is dropped
    wholesale, unlike an emphatic real repeat ("對對對"), which is short and is
    handled by _collapse_runs instead."""
    clauses = [_norm_clause(c) for c in _CLAUSE_SPLIT_RE.split(text) if _norm_clause(c)]
    if len(clauses) < 3:
        return False
    top, n = Counter(clauses).most_common(1)[0]
    return len(top) >= 6 and n >= 3 and n / len(clauses) >= 0.5


# A unit (≤30 chars) repeated 3+ times in a row, tolerating zh/en separators
# between copies. Catches no-separator loops and longer Chinese phrase loops
# (the 12-char cap missed "我們可以更快地了解它們的用途"-length units).
_RUN_REPEAT_RE = re.compile(r'(.{1,30}?)(?:[，、,。.!?！？\s]*\1){2,}')


def _collapse_runs(text: str) -> str:
    """Collapse a consecutively-repeated unit down to 2 copies (matching the
    sentence-level max_repeat=2 semantics). Latin units are rejoined with a
    space (so "you know you know you know" → "you know you know", not glued);
    CJK units join with no separator ("好的好的好的好的" → "好的好的"). A lone
    repeated single digit/letter ("5 5 5") is left alone — too likely real."""
    def _repl(m):
        unit = m.group(1)
        bare = unit.strip()
        if len(bare) <= 1 and re.match(r'[A-Za-z0-9]$', bare):
            return m.group(0)  # don't collapse "5 5 5" / "a a a"
        sep = ' ' if (re.search(r'[A-Za-z]', unit) and not _CJK_RE.search(unit)) else ''
        return unit + sep + unit
    return _RUN_REPEAT_RE.sub(_repl, text)


def _has_runaway_repeat(text: str) -> bool:
    """True if *text* contains a unit repeated 3+ times in a row (the
    punctuation-independent analogue of _is_repetition_loop). Used to suppress
    prompt-chain propagation on a looped chunk."""
    return bool(_RUN_REPEAT_RE.search(text))


def _dedup_boundary(text: str, from_cap: bool = True) -> str:
    """Drop the leading slice of *text* that repeats the tail of the previous
    transcript line.

    Only meaningful after a hard-'cap' cut, which keeps a 1s audio OVERLAP that
    gets transcribed twice and echoes a phrase at the seam. A silence ('pause')
    cut keeps no overlap, so any boundary match there is a genuine restatement —
    skip dedup entirely (from_cap=False) to avoid clipping legitimately repeated
    phrasing ("好的好的", "對對對"). We find the longest suffix of the previous
    line that is a prefix of this one (char-level, so it works for spaceless
    Chinese too) and strip it. Requires a 5-char match (8 for all-CJK runs, which
    collide more easily) so we don't clip incidental shared openers like "我覺得"."""
    if not from_cap or not _lines or not text.strip():
        return text
    prev_body = (_lines[-1].get("text") or "").strip()
    if not prev_body:
        return text
    cur = text.lstrip()
    tail = prev_body[-60:]                 # bounded search window
    maxk = min(len(tail), len(cur))
    for k in range(maxk, 4, -1):           # require >= 5 overlapping chars
        match = tail[-k:]
        if match.lower() == cur[:k].lower():
            # A pure-CJK overlap collides more easily (no word boundaries); demand
            # a longer match before trusting it's a real seam echo.
            if k < 8 and _CJK_RE.search(match) and not re.search(r'[A-Za-z0-9]', match):
                continue
            return cur[k:].lstrip()
    return text


def _best_cut_index(audio: np.ndarray) -> Optional[int]:
    """For a hard-cap cut, find a low-energy 100ms frame within the last ~2s to
    cut at (an inter-word gap) rather than slicing mid-word at the raw 25s mark.
    Returns a sample index, or None to fall back to the overlap-retain cut."""
    frame = int(SAMPLE_RATE * 0.1)
    window = int(SAMPLE_RATE * 2.0)
    if len(audio) < window + frame:
        return None
    region = audio[-window:]
    nf = len(region) // frame
    if nf < 2:
        return None
    frames = region[: nf * frame].reshape(nf, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    qi = int(np.argmin(rms))
    cut = (len(audio) - window) + qi * frame
    if cut <= 0 or cut >= len(audio) - frame:
        return None
    return cut


def _queue_chunk(mixed: np.ndarray, reason: str, drop_label: str):
    """Enqueue (audio, reason); on a full queue, drop but leave a PERSISTENT,
    timestamped gap marker in the transcript (not just a 0.3s status flash) so
    the lost span is locatable and survives into the saved file."""
    try:
        _transcribe_queue.put((mixed, reason), timeout=2)
    except queue.Full:
        secs = max(1, round(len(mixed) / SAMPLE_RATE))
        print(f"WARN: transcribe queue full, dropping {drop_label} (~{secs}s)", file=sys.stderr)
        _append_line(f"⚠ 略過約 {secs} 秒音訊 — 轉錄跟不上錄音速度", tag="dropped")
        _set_status("⚠ Transcribe 跟不上速度,跳過一段")


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
        time.sleep(CHUNK_POLL)
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

            if reason == "pause-check":
                # Probe WITHOUT advancing the gain EMA (this runs every poll).
                if not _is_pause_boundary(_mix_buffers(sa, ma, update_gain=False)):
                    continue

            # We're emitting a chunk — build the mix once, advancing the EMA gains
            # exactly once per emitted chunk (not per poll).
            mixed = _mix_buffers(sa, ma, update_gain=True)

            if reason == "pause-check":
                # Sentence ended — clear everything. A stale-speech tail would
                # prime a phantom silent chunk and Whisper would hallucinate.
                _sys_buf.clear()
                _mic_buf.clear()
                reason = "pause"
            else:  # 'cap' — continuous speech hit the hard cap.
                cut = _best_cut_index(mixed)
                if cut is not None:
                    # Cut at an inter-word gap and carry the remainder forward
                    # (no overlap → no seam echo → no dedup needed downstream).
                    mixed = mixed[:cut]
                    _sys_buf[:] = [sa[cut:]] if len(sa) > cut else []
                    _mic_buf[:] = [ma[cut:]] if len(ma) > cut else []
                    reason = "cap-clean"
                else:
                    # No good gap — keep a 1s overlap tail for context; the seam
                    # echo is stripped downstream by _dedup_boundary (from_cap).
                    _sys_buf[:] = [sa[-overlap_samples:]] if len(sa) > overlap_samples else []
                    _mic_buf[:] = [ma[-overlap_samples:]] if len(ma) > overlap_samples else []
                    reason = "cap"

        _queue_chunk(mixed, reason, "chunk")
        _broadcast("queue", _transcribe_queue.qsize())

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
            item = _transcribe_queue.get(timeout=0.5)
        except queue.Empty:
            if _consumer_should_exit.is_set():
                break
            continue
        try:
            audio, reason = item if isinstance(item, tuple) else (item, "flush")
            _transcribe(audio, api_key, reason)
            _broadcast("queue", _transcribe_queue.qsize())
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
    # Give the translator the previous 1-2 source lines as context (NOT to be
    # translated) so pronouns / continuations resolve across chunk boundaries.
    prev = [c for c in _translate_ctx if c.strip()]
    if prev:
        sys_prompt += " 前文(僅供理解上下文,不要翻譯這段):" + " ".join(prev)
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
    _translate_ctx.append(text)  # rolling source context for the next line
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


def _transcribe(audio: np.ndarray, api_key: str, reason: str = "flush"):
    """Transcribe *audio*, dispatching to cloud (Groq) or local (whisper.cpp)
    based on `_backend`. *reason* is the chunk cut type ('cap' keeps a seam
    overlap → boundary dedup runs; everything else has no seam)."""
    global _transcribing

    # Two-layer silence gate against Whisper hallucination on near-silent input:
    #   (1) Overall RMS too low → entire chunk is quiet.
    #   (2) Voice-activity ratio: fraction of 100ms frames that exceed the
    #       speech threshold. Whisper hallucinates on brief-speech-then-silence.
    # Thresholds tuned permissive (catch soft speech) — repetition_trim +
    # loop detection still handle the false-positive case.
    def _gated(weak: bool):
        # When a chunk is gated out but had some audible (non-silence) energy,
        # leave a faint hint so the user knows audio was skipped rather than
        # assuming silence — instead of silently swallowing a quiet speaker.
        if weak and _recording:
            _set_status("· 偵測到微弱聲音,未轉錄")
        else:
            _restore_idle_status()

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < TRANSCRIBE_MIN_RMS:
        _gated(weak=rms >= SILENCE_THRESHOLD)
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
            _gated(weak=active_seconds > 0)
            return

    ts = datetime.now().strftime("%H:%M:%S")
    _transcribing = True
    _broadcast("transcribing", True)
    _set_status(f"Transcribing [{ts}]…")

    vocab = load_vocab()
    prompt = _build_prompt(vocab)

    try:
        if _backend == "local":
            text, conf = _transcribe_local(audio, prompt)
        else:
            text, conf = _transcribe_cloud(audio, api_key, prompt)

        text = _drop_hallucinations((text or "").strip(), _language)
        if text and _is_loop_hallucination(text):
            # The whole chunk is a repetition-loop hallucination (e.g. the
            # classic "…用途,…用途,…用途" on near-silent audio). Drop it entirely
            # and do NOT chain it — showing even a trimmed version is noise, and
            # chaining it snowballs the rest of the session.
            text = ""
        if text:
            # collapse punctuation-free repetition loops first, then sentence-
            # level trim, then boundary dedup (only on a 'cap' seam).
            collapsed = _collapse_runs(text)
            cleaned = _dedup_boundary(_trim_repetition(collapsed), from_cap=(reason == "cap"))
            cleaned = apply_vocab_corrections(_strip_display_labels(cleaned))
            if cleaned.strip():
                tr = _maybe_translate(cleaned, api_key)
                _append_line(cleaned, tr=tr, ts=ts, conf=conf)
                # If the result still shows a repetition loop, the chunk was
                # unreliable — don't poison the next chunk's prompt chain. Also
                # strip speaker labels before chaining so a hallucinated "余婷:"
                # prefix doesn't prime the next chunk.
                if not (_is_repetition_loop(collapsed) or _has_runaway_repeat(text)):
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


def _assemble_cloud(body: dict) -> tuple[str, str]:
    """Build (text, conf) from a Groq verbose_json body. Drops segments on a
    STRONG combined hallucination signal (no_speech_prob high AND avg_logprob
    very low), and flags the line 'low' on a softer low-confidence signal — a
    model-confidence layer the energy gate + text heuristics lack."""
    segs = body.get("segments") or []
    if not segs:
        return (body.get("text", "") or "").strip(), ""
    kept, nsps, lps = [], [], []
    for s in segs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        nsp, lp = s.get("no_speech_prob"), s.get("avg_logprob")
        if (isinstance(nsp, (int, float)) and isinstance(lp, (int, float))
                and nsp > CLOUD_NSP_DROP and lp < CLOUD_LOGPROB_DROP):
            continue  # confident-hallucination → drop the segment
        kept.append(t)
        if isinstance(nsp, (int, float)):
            nsps.append(nsp)
        if isinstance(lp, (int, float)):
            lps.append(lp)
    text = " ".join(kept).strip()
    low = (nsps and max(nsps) > CLOUD_NSP_LOWCONF) or (lps and min(lps) < CLOUD_LOGPROB_LOWCONF)
    return text, ("low" if low else "")


def _transcribe_cloud(audio: np.ndarray, api_key: str, prompt: str) -> tuple[str, str]:
    """Cloud backend — Groq Whisper API. Returns (text, conf)."""
    tmp = tempfile.NamedTemporaryFile(prefix="mt_", suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        # verbose_json so we can read per-segment no_speech_prob / avg_logprob
        # (the typed response only exposes .text → use with_raw_response.json()).
        kwargs: dict = dict(model=load_cloud_model(), response_format="verbose_json")
        if _language != "auto":
            kwargs["language"] = _language
        if prompt:
            kwargs["prompt"] = prompt

        with open(tmp.name, "rb") as f:
            kwargs["file"] = ("chunk.wav", f, "audio/wav")
            raw = Groq(api_key=api_key).audio.transcriptions.with_raw_response.create(**kwargs)
        return _assemble_cloud(raw.json())
    finally:
        os.unlink(tmp.name)


def _transcribe_local(audio: np.ndarray, prompt: str) -> tuple[str, str]:
    """Local backend — inference runs in the LocalWhisperWorker subprocess.
    Returns (text, conf).

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
    common audio/video containers, so no local decode is needed. Uses the
    higher-accuracy non-turbo large-v3 (batch/offline → no latency cost)."""
    kw: dict = dict(model=CLOUD_MODEL_BATCH, response_format="verbose_json")
    if language != "auto":
        kw["language"] = language
    if prompt:
        kw["prompt"] = prompt
    with open(path, "rb") as af:
        kw["file"] = (fname, af)
        raw = Groq(api_key=api_key).audio.transcriptions.with_raw_response.create(**kw)
    return _assemble_cloud(raw.json())[0]


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
    return worker.transcribe(audio, language, prompt)[0]


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
