"""Count Whisper prompt tokens exactly, without loading a model.

Whisper keeps only the last 224 tokens of a prompt, so the prompt builder needs
real counts, not an estimate. Two tokenizers read the same prompt:

- Groq (cloud) runs OpenAI's tokenizer: byte-level BPE over the multilingual
  vocabulary.
- whisper.cpp (local) splits words with an ASCII-only regex and then takes the
  longest vocabulary match from the left, which often yields MORE tokens.

count() returns the larger of the two, so a prompt that fits is safe on both
backends. The vocabulary is OpenAI's `multilingual.tiktoken` (MIT, see
THIRD_PARTY_NOTICES.md), shared by every multilingual Whisper model including
large-v3 / turbo and Breeze. If the file is missing, count() falls back to the
UTF-8 byte length, a strict upper bound for both tokenizers.
"""
from __future__ import annotations

import base64
import re
import threading
from functools import lru_cache
from typing import Optional

# OpenAI whisper/tokenizer.py pre-tokenizer, with \p{L} / \p{N} expressed for
# Python's `re` ([^\W\d_] = letters, \d = digits).
_BPE_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
# whisper.cpp's tokenize(): std::regex in the "C" locale, so only ASCII counts
# as alpha / digit / space and every non-ASCII byte falls in the third class.
_CPP_SPLIT = re.compile(rb"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

_ranks: Optional[dict[bytes, int]] = None
_max_len = 0
_lock = threading.Lock()
_path: Optional[str] = None


def configure(vocab_path: str) -> None:
    """Point at the vocabulary file. Loaded lazily on the first count."""
    global _path, _ranks
    with _lock:
        _path, _ranks = vocab_path, None
    _count_cached.cache_clear()


def _load() -> Optional[dict[bytes, int]]:
    global _ranks, _max_len
    with _lock:
        if _ranks is None and _path:
            try:
                ranks = {}
                with open(_path, "rb") as f:
                    for line in f:
                        tok, rank = line.split()
                        ranks[base64.b64decode(tok)] = int(rank)
                _ranks, _max_len = ranks, max(map(len, ranks))
            except (OSError, ValueError):
                _ranks = {}
        return _ranks or None


def _bpe(piece: bytes, ranks: dict[bytes, int]) -> int:
    parts = [piece[i:i + 1] for i in range(len(piece))]
    while len(parts) > 1:
        best, best_rank = -1, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = i, r
        if best < 0:
            break
        parts[best:best + 2] = [parts[best] + parts[best + 1]]
    return len(parts)


def bpe_count(text: str) -> int:
    """Tokens in OpenAI's tokenizer (what Groq sees)."""
    ranks = _load()
    if ranks is None:
        return len(text.encode("utf-8"))
    return sum(_bpe(m.group().encode("utf-8"), ranks) for m in _BPE_SPLIT.finditer(text))


def cpp_count(text: str) -> int:
    """Tokens in whisper.cpp's greedy longest-match tokenizer (local models)."""
    ranks = _load()
    data = text.encode("utf-8")
    if ranks is None:
        return len(data)
    n = 0
    for m in _CPP_SPLIT.finditer(data):
        word, i = m.group(), 0
        while i < len(word):
            for j in range(min(len(word), i + _max_len), i, -1):
                if word[i:j] in ranks:
                    break
            n += 1
            i = j
    return n


@lru_cache(maxsize=4096)
def _count_cached(text: str) -> int:
    return max(bpe_count(text), cpp_count(text))


def count(text: str) -> int:
    """Prompt tokens on the stricter of Groq's and whisper.cpp's tokenizers."""
    return _count_cached(text) if text else 0
