"""Tests for the transcription-accuracy changes: prompt budget, longer live
chunks, and the whole-session recording + post-session re-transcription.

Run:  /opt/homebrew/bin/python3.13 -m unittest discover -s tests -v
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import wave
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402

SR = A.SAMPLE_RATE

# The vocab that filled the whole prompt before (25 terms, 242 chars).
LONG_TERMS = ["GoFreight", "Brad", "Figma", "Shipment", "action center", "to-do list", "HBL",
              "MBL", "Descartes", "Loki", "importer", "console", "design intent", "GoNexus",
              "FMS", "arrival notice", "drag and drop", "shipper", "consignee", "CY location",
              "firm code", "import list", "pick-up number", "co-loader", "Carpusher"]
CHAIN = "所以那個 enterprise 這條線我們先看 automation，另外一個是 accounting 的部分，workflow engine 那邊再對一下"


def tone(seconds, amp=0.1, freq=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


class PromptBudget(unittest.TestCase):
    def test_long_vocab_no_longer_crowds_out_the_chain(self):
        chain = CHAIN * 3  # longer than the chain reserve
        p = A.assemble_prompt(chain, LONG_TERMS, A._BILINGUAL_PROMPT)
        self.assertLessEqual(A.estimate_prompt_tokens(p), A.PROMPT_TOKEN_BUDGET)
        chain_part = p.split("專有名詞:")[0].strip()
        self.assertGreaterEqual(A.estimate_prompt_tokens(chain_part), A.PROMPT_CHAIN_MIN_TOKENS - 2)
        self.assertTrue(chain.endswith(chain_part), "chain keeps the NEWEST text")

    def test_chain_shorter_than_reserve_is_kept_whole(self):
        p = A.assemble_prompt(CHAIN, LONG_TERMS, A._BILINGUAL_PROMPT)
        self.assertTrue(p.startswith(CHAIN + " 專有名詞:"))

    def test_prime_is_whole_and_last(self):
        p = A.assemble_prompt(CHAIN, LONG_TERMS, A._BILINGUAL_PROMPT)
        self.assertTrue(p.endswith(A._BILINGUAL_PROMPT))

    def test_vocab_keeps_whole_terms_in_file_order(self):
        p = A.assemble_prompt(CHAIN, LONG_TERMS, A._BILINGUAL_PROMPT)
        vocab = p[p.index("專有名詞:") + len("專有名詞:"):p.index("。", p.index("專有名詞:"))]
        kept = vocab.split("、")
        self.assertEqual(kept, LONG_TERMS[:len(kept)])
        self.assertGreater(len(kept), 0)

    def test_short_vocab_leaves_room_for_more_chain(self):
        long_chain = CHAIN * 4
        short = A.assemble_prompt(long_chain, ["GoFreight"], A._BILINGUAL_PROMPT)
        long = A.assemble_prompt(long_chain, LONG_TERMS, A._BILINGUAL_PROMPT)
        self.assertGreater(len(short.split("專有名詞:")[0]), len(long.split("專有名詞:")[0]))
        self.assertLessEqual(A.estimate_prompt_tokens(short), A.PROMPT_TOKEN_BUDGET)

    def test_no_chain_lets_vocab_use_the_budget(self):
        p = A.assemble_prompt("", LONG_TERMS, A._BILINGUAL_PROMPT)
        self.assertTrue(p.startswith("專有名詞:"))
        self.assertLessEqual(A.estimate_prompt_tokens(p), A.PROMPT_TOKEN_BUDGET)

    def test_empty_parts(self):
        self.assertEqual(A.assemble_prompt("", [], ""), "")
        self.assertEqual(A.assemble_prompt("", [], A._BILINGUAL_PROMPT), A._BILINGUAL_PROMPT)


def simulate_chunks(audio, block=0.1):
    """Replay *audio* through the live chunker's trigger logic (mono, same
    helpers as _chunk_worker) and return (length_seconds, reason) per chunk."""
    blk = int(SR * block)
    buf = np.zeros(0, np.float32)
    out = []
    for i in range(0, len(audio), blk):
        buf = np.concatenate([buf, audio[i:i + blk]])
        trigger, reason = A._should_trigger(len(buf))
        if not trigger:
            continue
        if reason == "pause-check":
            if not A._is_pause_boundary(buf):
                continue
            out.append((len(buf) / SR, "pause"))
            buf = np.zeros(0, np.float32)
        else:
            cut = A._best_cut_index(buf)
            cut = cut if cut is not None else len(buf) - int(SR * A.OVERLAP)
            out.append((cut / SR, "cap"))
            buf = buf[cut:]
    return out


def conversation(seconds=600, seed=7):
    """Speech-like audio: 1.5–6 s utterances separated by 0.3–2.5 s gaps."""
    rng = np.random.default_rng(seed)
    parts, total = [], 0.0
    while total < seconds:
        talk = rng.uniform(1.5, 6.0)
        gap = rng.uniform(0.3, 2.5)
        parts += [tone(talk, amp=rng.uniform(0.05, 0.15)), silence(gap)]
        total += talk + gap
    return np.concatenate(parts)


class LongerChunks(unittest.TestCase):
    def test_short_buffer_no_longer_triggers_a_pause_check(self):
        self.assertEqual(A._should_trigger(int(SR * 5)), (False, ""))
        self.assertEqual(A._should_trigger(int(SR * A.MIN_CHUNK_DURATION)), (True, "pause-check"))
        self.assertEqual(A._should_trigger(int(SR * A.CHUNK_DURATION)), (True, "cap"))

    def test_no_pause_chunk_shorter_than_minimum(self):
        chunks = simulate_chunks(conversation())
        self.assertTrue(all(s >= A.MIN_CHUNK_DURATION for s, r in chunks if r == "pause"))
        self.assertTrue(all(s <= A.CHUNK_DURATION for s, _ in chunks))

    def test_chunks_are_longer_than_with_old_settings(self):
        audio = conversation()
        new = simulate_chunks(audio)
        with mock.patch.object(A, "PAUSE_DURATION", 1.5), mock.patch.object(A, "MIN_CHUNK_DURATION", 0.0):
            old = simulate_chunks(audio)
        mean = lambda c: sum(s for s, _ in c) / len(c)  # noqa: E731
        self.assertGreater(mean(new), mean(old) * 1.5)


class Recorder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _rec(self):
        return A.SessionRecorder(os.path.join(self.dir, "s.wav"))

    def test_writes_every_sample_once(self):
        rec = self._rec()
        rec.write(tone(3), end_time=1000.0)
        rec.write(tone(2), end_time=1010.0)
        rec.close()
        with wave.open(rec.path, "rb") as w:
            self.assertEqual(w.getnframes(), 5 * SR)
            self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth()), (1, SR, 2))

    def test_offsets_map_to_wall_clock_across_a_pause(self):
        rec = self._rec()
        rec.write(tone(3), end_time=1000.0)   # started at 997
        rec.write(tone(2), end_time=1060.0)   # paused, resumed at 1058
        rec.close()
        self.assertEqual(rec.wallclock(1.0).timestamp(), 998.0)
        self.assertEqual(rec.wallclock(4.0).timestamp(), 1059.0)

    def test_prune_deletes_only_old_recordings(self):
        old, new = os.path.join(self.dir, "old.m4a"), os.path.join(self.dir, "new.m4a")
        for p in (old, new):
            open(p, "wb").close()
        now = time.time()
        os.utime(old, (now - 8 * 86400, now - 8 * 86400))
        with mock.patch.object(A, "RECORDINGS_DIR", self.dir):
            A.prune_recordings(now=now)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(new))

    def test_compress_replaces_wav_with_smaller_m4a(self):
        rec = self._rec()
        rec.write(tone(10), end_time=1000.0)
        rec.close()
        wav_size = os.path.getsize(rec.path)
        out = A.compress_recording(rec.path)
        self.assertTrue(out.endswith(".m4a"))
        self.assertFalse(os.path.exists(rec.path))
        self.assertLess(os.path.getsize(out), wav_size / 3)


class RecorderInChunkWorker(unittest.TestCase):
    """Drive the real _chunk_worker (no audio devices) and check the session
    recording holds every input sample exactly once, across pause, clean-cap,
    overlap-cap and final-flush cuts."""

    def test_every_sample_recorded_once(self):
        d = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        # 80 s unbroken speech-like noise (forces cap cuts) + a talk/gap stretch.
        audio = np.concatenate([(rng.standard_normal(80 * SR) * 0.08).astype(np.float32),
                                conversation(60, seed=5)])
        blk = int(SR * 0.1)
        drained = []
        real_cut, calls = A._best_cut_index, []

        def every_other_cut_has_no_gap(a):  # exercise the overlap fallback too
            calls.append(1)
            return None if len(calls) % 2 else real_cut(a)

        def drain():
            while A._recording or not A._transcribe_queue.empty():
                try:
                    drained.append(A._transcribe_queue.get(timeout=0.05))
                except Exception:
                    pass
        with mock.patch.object(A, "CHUNK_POLL", 0.001), mock.patch.object(A, "_broadcast"), \
                mock.patch.object(A, "_append_line"), mock.patch.object(A, "_set_status"), \
                mock.patch.object(A, "_best_cut_index", side_effect=every_other_cut_has_no_gap):
            A._recorder = A.SessionRecorder(os.path.join(d, "s.wav"))
            A._sys_buf.clear()
            A._mic_buf.clear()
            A._recording, A._paused = True, False
            worker = threading.Thread(target=A._chunk_worker, args=("",))
            consumer = threading.Thread(target=drain)
            worker.start()
            consumer.start()
            for i in range(0, len(audio), blk):
                with A._buf_lock:
                    A._sys_buf.append(audio[i:i + blk])
                time.sleep(0.0015)
            A._recording = False
            worker.join(10)
            consumer.join(10)
            rec, A._recorder = A._recorder, None
            rec.close()
        try:
            reasons = {r for _, r in drained}
            self.assertTrue({"cap", "cap-clean", "pause"} <= reasons, reasons)
            self.assertLessEqual(abs(rec.samples - len(audio)), blk,
                                 f"recorded {rec.samples} of {len(audio)} samples")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class FakeWorker:
    def __init__(self, per_piece):
        self.per_piece = per_piece
        self.prompts = []

    def transcribe_segments(self, audio, language, prompt, timeout=None):
        self.prompts.append(prompt)
        return self.per_piece(len(self.prompts))


class Retranscribe(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.rec = A.SessionRecorder(os.path.join(self.dir, "s.wav"))
        # 200 s starting 11:00:00; the 2 s gap at 170 s is where piece 1 ends.
        self.rec.write(np.concatenate([tone(170), silence(2), tone(28)]),
                       end_time=time.mktime((2026, 10, 22, 11, 3, 20, 0, 0, -1)))
        self.rec.close()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_lines_are_cleaned_merged_and_clock_stamped(self):
        w = FakeWorker(lambda n: [(0.0, "我們先看 Shipment 的 workflow"), (4.0, "對"),
                                  (9.0, "[Music]"), (12.5, "第二句話在這裡講完")] if n == 1 else
                       [(1.0, "第二段的內容")])
        lines = A.retranscribe_recording(self.rec, w, "zh", threading.Event())
        self.assertEqual([l["ts"] for l in lines], ["11:00:00", "11:00:12", "11:02:51"])
        self.assertEqual(lines[0]["text"], "我們先看 Shipment 的 workflow 對")
        self.assertEqual(len(w.prompts), 2)  # 200 s → two 180 s pieces
        self.assertIn("第二句話", w.prompts[1], "next piece is primed with the previous text")

    def test_cancel_returns_none(self):
        cancel = threading.Event()
        cancel.set()
        self.assertIsNone(A.retranscribe_recording(self.rec, FakeWorker(lambda n: []), "zh", cancel))


class SyncThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


class ReplaceLiveLines(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.rec = A.SessionRecorder(os.path.join(self.dir, "s.wav"))
        self.rec.write(tone(5), end_time=time.mktime((2026, 10, 22, 11, 0, 5, 0, 0, -1)))
        self.rec.close()
        self._saved = (list(A._lines), A._lines_rev, A._session_first_line, A.DRAFT_FILE)
        A._lines[:] = [{"ts": "10:00:00", "text": "上一場", "tr": "", "tag": ""},
                       {"ts": "11:00:00", "text": "即時版本", "tr": "", "tag": ""}]
        A._session_first_line = 1
        A.DRAFT_FILE = os.path.join(self.dir, "draft.json")
        self.patches = [
            mock.patch.object(A, "model_local_path", return_value="/fake/model.bin"),
            mock.patch.object(A, "_ensure_local_worker",
                              return_value=FakeWorker(lambda n: [(0.0, "重新整理後的版本")])),
            mock.patch.object(A.threading, "Thread", SyncThread),
            mock.patch.object(A, "_broadcast"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        A._lines[:], A._lines_rev, A._session_first_line, A.DRAFT_FILE = self._saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_replaces_only_this_sessions_lines_and_compresses(self):
        A.start_retranscription(self.rec, "zh")
        self.assertEqual([l["text"] for l in A._lines], ["上一場", "重新整理後的版本"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "s.m4a")))

    def test_an_edit_during_the_pass_keeps_the_live_lines(self):
        real = A.retranscribe_recording

        def edit_midway(*a, **kw):
            A._lines_rev += 1  # what /line, /clear or an upload does
            return real(*a, **kw)
        with mock.patch.object(A, "retranscribe_recording", side_effect=edit_midway):
            A.start_retranscription(self.rec, "zh")
        self.assertEqual([l["text"] for l in A._lines], ["上一場", "即時版本"])

    def test_skipped_when_the_model_is_not_downloaded(self):
        with mock.patch.object(A, "model_local_path", return_value=None):
            A.start_retranscription(self.rec, "zh")
        self.assertEqual([l["text"] for l in A._lines], ["上一場", "即時版本"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "s.m4a")))


if __name__ == "__main__":
    unittest.main()
