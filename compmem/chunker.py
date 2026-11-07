"""Phase 1, Step A: split a long conversation into chunks of up to 256 tokens."""
import re
import zlib
from typing import List

MAX_TOKENS = 256


def tokenize(text: str) -> List[str]:
    """Very simple word-level tokenizer (keeps punctuation as separate tokens)."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Greedily pack whole sentences into chunks of <= max_tokens tokens.
    A single sentence longer than max_tokens is hard-split."""
    chunks, cur, cur_len = [], [], 0
    for sent in split_sentences(text):
        n = len(tokenize(sent))
        if n > max_tokens:  # hard split very long sentence
            if cur:
                chunks.append(" ".join(cur)); cur, cur_len = [], 0
            words = sent.split()
            for i in range(0, len(words), max_tokens):
                chunks.append(" ".join(words[i:i + max_tokens]))
            continue
        if cur_len + n > max_tokens:
            chunks.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(sent); cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def token_ids(text: str, vocab_size: int = 8192, max_len: int = MAX_TOKENS):
    """Hashing tokenizer: stable word -> id (1..vocab_size-1); 0 is padding.
    No external vocab/download needed."""
    ids = [(zlib.crc32(t.encode()) % (vocab_size - 1)) + 1 for t in tokenize(text)]
    return ids[:max_len]
