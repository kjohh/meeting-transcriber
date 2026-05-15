from __future__ import annotations

import io
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd
from flask import Flask, Response, jsonify, request, send_file
from groq import Groq

# ─── Config ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
PORT = 8765
BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native/.build/release/coreaudio_tap")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".config.json")

app = Flask(__name__, static_folder="static")

# ─── Global state ─────────────────────────────────────────────────────────────

_recording = False
_paused = False
_language = "zh"  # default: Traditional Chinese
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

def load_api_key() -> str:
    if k := os.environ.get("GROQ_API_KEY", ""):
        return k
    try:
        return json.loads(open(CONFIG_FILE).read()).get("groq_api_key", "")
    except Exception:
        return ""


def save_api_key(key: str):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"groq_api_key": key}, f)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/events")
def events():
    q: queue.Queue = queue.Queue(maxsize=200)
    _sse_clients.append(q)

    def generate():
        try:
            # send initial state on connect
            yield f"data: {json.dumps({'type':'init','key':load_api_key(),'lines':_lines,'recording':_recording,'paused':_paused,'language':_language})}\n\n"
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
    key = (request.json or {}).get("key", "").strip()
    if key:
        save_api_key(key)
    return jsonify({"ok": True})


@app.route("/start", methods=["POST"])
def route_start():
    global _recording, _paused, _swift_proc, _mic_stream, _language

    data = request.json or {}
    key = data.get("key", "").strip()
    _language = data.get("language", "zh")
    if not key:
        return jsonify({"ok": False, "error": "No API key"})
    if not os.path.exists(BINARY):
        return jsonify({"ok": False, "error": "Binary missing — run: cd native && swift build -c release"})

    _recording = True
    _paused = False
    with _buf_lock:
        _sys_buf.clear()
        _mic_buf.clear()

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
    threading.Thread(target=_chunk_worker, args=(key,), daemon=True).start()

    return jsonify({"ok": True})


@app.route("/pause", methods=["POST"])
def route_pause():
    global _paused
    _paused = not _paused
    _broadcast("state", {"recording": _recording, "paused": _paused})
    _set_status("Paused" if _paused else "Recording…")
    return jsonify({"ok": True, "paused": _paused})


@app.route("/stop", methods=["POST"])
def route_stop():
    global _recording, _paused, _swift_proc, _mic_stream

    _recording = False
    _paused = False

    if _mic_stream:
        _mic_stream.stop()
        _mic_stream.close()
        _mic_stream = None
    if _swift_proc:
        _swift_proc.terminate()
        _swift_proc = None

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
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.save(tmp.name)
    fname = f.filename

    def _do():
        ts = datetime.now().strftime("%H:%M:%S")
        _set_status(f"Transcribing {fname}…")
        try:
            client = Groq(api_key=key)
            with open(tmp.name, "rb") as af:
                kw = dict(model="whisper-large-v3-turbo", file=(fname, af))
                if _language != "auto":
                    kw["language"] = _language
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
    _lines.clear()
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
        if data and not _paused:
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
            _set_status("Recording…")
        elif line.startswith("ERROR"):
            _set_status(
                "Audio error — grant Screen Recording permission in "
                "System Settings → Privacy & Security → Screen & System Audio Recording"
            )


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


def _chunk_worker(api_key: str):
    target = SAMPLE_RATE * CHUNK_SECONDS
    while _recording:
        time.sleep(1)
        if _paused:
            continue

        with _buf_lock:
            sys_len = sum(len(x) for x in _sys_buf)
            mic_len = sum(len(x) for x in _mic_buf)
            # Trigger when the longer of the two buffers reaches target
            # (allows mic-only if system audio isn't available)
            primary = mic_len if sys_len == 0 else (sys_len if mic_len == 0 else min(sys_len, mic_len))
            if primary < target:
                continue

            sys_full = np.concatenate(_sys_buf) if _sys_buf else np.array([], np.float32)
            mic_full = np.concatenate(_mic_buf) if _mic_buf else np.array([], np.float32)
            _sys_buf[:] = [sys_full[target:]] if len(sys_full) > target else []
            _mic_buf[:] = [mic_full[target:]] if len(mic_full) > target else []

        mixed = _mix_buffers(sys_full[:target], mic_full[:target])
        threading.Thread(target=_transcribe, args=(mixed, api_key), daemon=True).start()

    # flush remaining audio after stop
    with _buf_lock:
        sa = np.concatenate(_sys_buf) if _sys_buf else np.array([], np.float32)
        ma = np.concatenate(_mic_buf) if _mic_buf else np.array([], np.float32)
    audio = _mix_buffers(sa, ma)
    if len(audio) > SAMPLE_RATE // 2:  # at least 0.5 s
        _transcribe(audio, api_key)


def _transcribe(audio: np.ndarray, api_key: str):
    global _transcribing
    # Skip silent chunks to prevent Whisper hallucinations
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        if _recording:
            _set_status("Recording…")
        return

    ts = datetime.now().strftime("%H:%M:%S")
    _transcribing = True
    _broadcast("transcribing", True)
    _set_status(f"Transcribing [{ts}]…")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        kwargs = dict(model="whisper-large-v3-turbo", file=("chunk.wav", open(tmp.name, "rb"), "audio/wav"))
        if _language != "auto":
            kwargs["language"] = _language

        with open(tmp.name, "rb") as f:
            kwargs["file"] = ("chunk.wav", f, "audio/wav")
            result = Groq(api_key=api_key).audio.transcriptions.create(**kwargs)

        text = result.text.strip()
        if text:
            _append_line(f"[{ts}] {text}")
    except Exception as e:
        _append_line(f"[{ts}] Error: {e}")
    finally:
        os.unlink(tmp.name)
        _transcribing = False
        _broadcast("transcribing", False)

    if _recording:
        _set_status("Recording…")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webview

    # Flask runs in a daemon thread; dies automatically when the window closes
    threading.Thread(
        target=lambda: app.run(port=PORT, threaded=True, use_reloader=False, debug=False),
        daemon=True,
    ).start()
    time.sleep(0.6)  # let Flask start before opening the window

    window = webview.create_window(
        "Meeting Transcriber",
        f"http://localhost:{PORT}",
        width=820,
        height=660,
        min_size=(600, 440),
    )
    webview.start()
    # webview.start() blocks until window is closed — process exits cleanly
