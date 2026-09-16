from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    text: str
    start: int
    end: int
    n_tokens: int


def token_offsets(text: str, encoding_model: str = "cl100k_base") -> list[tuple[int, int]]:
    """Character spans of token IDs, preserving exact source offsets.

    tiktoken does not expose character offsets; round-trip each token's bytes.
    UTF-8 token boundaries inside a codepoint are rounded down consistently.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Install tiktoken; never substitute a different tokenizer in the experiment") from exc
    enc = tiktoken.get_encoding(encoding_model)
    ids = enc.encode(text)
    offsets, byte_pos = [], 0
    source = text.encode("utf-8")
    # Linear byte-to-character mapping; repeatedly decoding source prefixes is quadratic.
    char_at = [0] * (len(source) + 1)
    position = 0
    for char_index, char in enumerate(text):
        width = len(char.encode("utf-8"))
        for j in range(width): char_at[position + j] = char_index
        position += width
        char_at[position] = char_index + 1
    for token in ids:
        piece = enc.decode_single_token_bytes(token)
        start_byte, byte_pos = byte_pos, byte_pos + len(piece)
        if source[start_byte:byte_pos] != piece:
            raise ValueError("Tokenizer did not round-trip source text")
        offsets.append((char_at[start_byte], char_at[byte_pos]))
    return offsets


def fixed_chunks(document: str, text: str, size: int, overlap_ratio: float = 0,
                 encoding_model: str = "cl100k_base") -> list[Chunk]:
    if size <= 0 or not 0 <= overlap_ratio < 1:
        raise ValueError("Invalid size or overlap ratio")
    offsets = token_offsets(text, encoding_model)
    overlap = int(size * overlap_ratio)
    step = size - overlap
    chunks = []
    for i in range(0, len(offsets), step):
        part = offsets[i:i + size]
        if not part:
            break
        start, end = part[0][0], part[-1][1]
        ident = hashlib.sha256(f"{document}:{size}:{i}:{start}:{end}".encode()).hexdigest()[:24]
        chunks.append(Chunk(ident, document, text[start:end], start, end, len(part)))
        if i + size >= len(offsets):
            break
    return chunks
