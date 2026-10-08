"""Unit tests for the pure text / audio helpers in app.py.

Run:  /opt/homebrew/bin/python3.13 -m unittest discover -s tests -v
(No test framework to install; plain unittest.)
"""
import os
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402

SR = A.SAMPLE_RATE


def tone(seconds, amp=0.1, freq=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


class TrimRepetition(unittest.TestCase):
    def test_keeps_at_most_two_consecutive_copies(self):
        out = A._trim_repetition("好的,好的,好的,好的,我們開始")
        self.assertEqual(out.count("好的"), 2)
        self.assertIn("我們開始", out)

    def test_untouched_when_nothing_repeats(self):
        s = "今天先討論需求,再排時程。"
        self.assertEqual(A._trim_repetition(s), s)


class CollapseRuns(unittest.TestCase):
    def test_cjk_run_collapses_to_two_without_separator(self):
        self.assertEqual(A._collapse_runs("好的好的好的好的"), "好的好的")

    def test_latin_run_rejoined_with_space(self):
        self.assertEqual(A._collapse_runs("you know you know you know"), "you know you know")

    def test_single_digit_repeat_left_alone(self):
        self.assertEqual(A._collapse_runs("5 5 5"), "5 5 5")


class LoopDetection(unittest.TestCase):
    def test_dominant_long_clause_is_loop(self):
        self.assertTrue(A._is_loop_hallucination("我們可以了解它的用途," * 5))

    def test_short_emphatic_repeat_is_not_loop(self):
        self.assertFalse(A._is_loop_hallucination("對,對,對,我同意"))

    def test_repetition_loop_needs_three_in_a_row(self):
        self.assertTrue(A._is_repetition_loop("嗯,好,好,好"))
        self.assertFalse(A._is_repetition_loop("好,好,嗯"))


class DropHallucinations(unittest.TestCase):
    def test_nonspeech_markers_removed(self):
        self.assertEqual(A._drop_hallucinations("[Music]", "zh"), "")

    def test_zh_lock_drops_stock_english_filler(self):
        self.assertEqual(A._drop_hallucinations("Thank you.", "zh"), "")

    def test_zh_lock_keeps_real_english_sentence(self):
        s = "Let me share my screen."
        self.assertEqual(A._drop_hallucinations(s, "zh"), s)

    def test_en_lock_strips_cjk_drift_keeps_english(self):
        self.assertEqual(A._drop_hallucinations("Hello team 你好世界", "en"), "Hello team")

    def test_en_lock_drops_all_cjk_segment(self):
        self.assertEqual(A._drop_hallucinations("你好世界", "en"), "")

    def test_auto_leaves_mixed_text(self):
        s = "這個 feature 下週上線"
        self.assertEqual(A._drop_hallucinations(s, "auto"), s)


class DedupBoundary(unittest.TestCase):
    def setUp(self):
        self._saved = list(A._lines)
        A._lines.clear()
        A._lines.append({"ts": "", "text": "we should ship the new report next week", "tr": "", "tag": ""})

    def tearDown(self):
        A._lines.clear()
        A._lines.extend(self._saved)

    def test_strips_seam_echo_after_cap_cut(self):
        out = A._dedup_boundary("next week and then review it", from_cap=True)
        self.assertEqual(out, "and then review it")

    def test_no_dedup_after_pause_cut(self):
        s = "next week and then review it"
        self.assertEqual(A._dedup_boundary(s, from_cap=False), s)

    def test_short_overlap_not_clipped(self):
        s = "week, fine"
        self.assertEqual(A._dedup_boundary(s, from_cap=True), s)


class PauseBoundary(unittest.TestCase):
    def test_speech_then_silence_is_boundary(self):
        self.assertTrue(A._is_pause_boundary(np.concatenate([tone(3), silence(2)])))

    def test_continuous_speech_is_not(self):
        self.assertFalse(A._is_pause_boundary(tone(5)))

    def test_silence_only_is_not(self):
        self.assertFalse(A._is_pause_boundary(silence(5)))

    def test_too_short_is_not(self):
        self.assertFalse(A._is_pause_boundary(np.concatenate([tone(0.5), silence(1.6)])))


class Formatting(unittest.TestCase):
    def test_dropped_tag_not_printed(self):
        self.assertEqual(A._format_line({"ts": "10:00:00", "text": "x", "tr": "", "tag": "dropped"}),
                         "[10:00:00] x")

    def test_translation_on_second_line(self):
        out = A._format_line({"ts": "1", "text": "hi", "tr": "嗨", "tag": ""})
        self.assertEqual(out, "[1] hi\n    ↳ 嗨")

    def test_markdown_export(self):
        md = A.format_transcript([
            {"ts": "10:00:00", "text": "第一段", "tr": "", "tag": ""},
            {"ts": "10:00:30", "text": "hello", "tr": "你好", "tag": ""},
        ], "md")
        self.assertTrue(md.startswith("# 會議逐字稿"))
        self.assertIn("**10:00:00**  \n第一段", md)
        self.assertIn("> 你好", md)

    def test_offset_format(self):
        self.assertEqual(A._fmt_offset(3725.9), "01:02:05")

    def test_version_compare(self):
        self.assertGreater(A._version_tuple("v0.2.0"), A._version_tuple("0.1.12"))
        self.assertEqual(A._version_tuple("0.1.12"), (0, 1, 12))


class UploadSegments(unittest.TestCase):
    def test_segments_cover_whole_file_and_cut_in_silence(self):
        # 3 × (4 s tone + 1 s silence) = 15 s; 6 s segments must cut in the gaps.
        audio = np.concatenate([np.concatenate([tone(4), silence(1)]) for _ in range(3)])
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((audio * 32767).astype(np.int16).tobytes())
            segs = list(A.iter_wav_segments(path, seg_seconds=6))
            self.assertAlmostEqual(sum(len(a) for _, a in segs) / SR, 15.0, places=2)
            for start, a in segs[1:]:
                # every cut lands in a silent gap (4–5 s, 9–10 s, …)
                self.assertGreaterEqual(start % 5, 4.0 - 1e-6, f"cut at {start:.2f}s is inside speech")
        finally:
            os.unlink(path)


class Draft(unittest.TestCase):
    def setUp(self):
        self._file = A.DRAFT_FILE
        self._saved = list(A._lines)
        fd, A.DRAFT_FILE = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(A.DRAFT_FILE)
        A._lines.clear()

    def tearDown(self):
        A.clear_draft()
        A.DRAFT_FILE = self._file
        A._lines.clear()
        A._lines.extend(self._saved)

    def test_unsaved_draft_offered_saved_draft_discarded(self):
        A._lines.append({"ts": "1", "text": "a", "tr": "", "tag": ""})
        A.save_draft()
        self.assertEqual(len(A.load_draft()["lines"]), 1)
        A.save_draft(saved=True)
        self.assertIsNone(A.load_draft())
        self.assertFalse(os.path.exists(A.DRAFT_FILE))


if __name__ == "__main__":
    unittest.main()
