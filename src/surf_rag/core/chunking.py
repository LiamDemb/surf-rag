"""Token-based chunking using tiktoken (OpenAI cl100k_base).

Chunks are sized by token count to align with LLM context limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .corpus_schemas import Block

_encoding: Optional[object] = None


def _get_encoding():
    """Lazy-load tiktoken encoding."""
    global _encoding
    if _encoding is None:
        import tiktoken
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _count_tokens(text: str) -> int:
    """Return token count (cl100k_base)."""
    return len(_get_encoding().encode(text))


def _tail_tokens(text: str, token_count: int) -> str:
    """Return the last token_count tokens as decoded text for overlap."""
    enc = _get_encoding()
    ids = enc.encode(text)
    tail_ids = ids[-token_count:] if len(ids) >= token_count else ids
    return enc.decode(tail_ids) if tail_ids else ""


def _split_long_block(
    block: Block,
    max_tokens: int,
    overlap_tokens: int,
    char_cursor: int,
) -> Tuple[List["ChunkPiece"], int]:
    """Split a block exceeding max_tokens into overlapping ChunkPieces."""
    enc = _get_encoding()
    ids = enc.encode(block.text)
    if len(ids) <= max_tokens:
        return [], char_cursor

    chunks: List[ChunkPiece] = []
    stride = max_tokens - overlap_tokens
    pos = 0
    while pos < len(ids):
        chunk_ids = ids[pos : pos + max_tokens]
        chunk_text = enc.decode(chunk_ids)
        chunk_tokens = len(chunk_ids)
        start = char_cursor
        end = start + len(chunk_text)
        chunks.append(
            ChunkPiece(
                text=chunk_text,
                section_path=block.section_path,
                char_span_in_doc=(start, end),
                token_count=chunk_tokens,
            )
        )
        char_cursor = end
        pos += stride

    return chunks, char_cursor


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    section_path: List[str]
    char_span_in_doc: Tuple[int, int]
    token_count: int


def chunk_blocks(
    blocks: Sequence[Block],
    min_tokens: int = 0,
    max_tokens: int = 500,
    overlap_tokens: int = 100,
    tokenizer: Optional[object] = None,
) -> List[ChunkPiece]:
    """Chunk blocks by token count (tiktoken cl100k_base).

    Invariant: emitted chunks never exceed max_tokens.
    min_tokens is a best-effort target (used when it does not violate max_tokens).

    The tokenizer argument is deprecated and ignored (kept for backward compatibility).
    """
    chunks: List[ChunkPiece] = []
    buf: List[Block] = []
    buf_tokens = 0
    char_cursor = 0

    def _join_blocks(blocks_to_join: Sequence[Block]) -> str:
        return "\n\n".join(block.text for block in blocks_to_join if block.text)

    def _flush_buf() -> None:
        nonlocal buf, buf_tokens, char_cursor
        if not buf:
            return
        chunk_text = _join_blocks(buf)
        chunk_tokens = _count_tokens(chunk_text)
        start = char_cursor
        end = start + len(chunk_text)
        chunks.append(
            ChunkPiece(
                text=chunk_text,
                section_path=buf[-1].section_path,
                char_span_in_doc=(start, end),
                token_count=chunk_tokens,
            )
        )
        char_cursor = end
        tail = _tail_tokens(chunk_text, overlap_tokens)
        buf = [Block(text=tail, section_path=buf[-1].section_path, block_type="overlap")]
        buf_tokens = _count_tokens(tail)

    for block in blocks:
        if not block.text.strip():
            continue
        block_tokens = _count_tokens(block.text)

        if block_tokens > max_tokens:
            _flush_buf()
            sub_chunks, char_cursor = _split_long_block(
                block, max_tokens, overlap_tokens, char_cursor
            )
            chunks.extend(sub_chunks)
            buf = []
            buf_tokens = 0
            continue

        if not buf:
            buf.append(block)
            buf_tokens = block_tokens
            continue

        if buf_tokens + block_tokens <= max_tokens:
            buf.append(block)
            buf_tokens += block_tokens
            continue

        # Adding this block would exceed max_tokens. Flush current buffer even if it
        # hasn't reached min_tokens to maintain the hard cap.
        #
        # Special case: if the buffer is only an overlap tail and still prevents the
        # next block from fitting, drop it to avoid emitting tiny overlap-only chunks.
        if len(buf) == 1 and buf[0].block_type == "overlap" and buf_tokens < min_tokens:
            buf = []
            buf_tokens = 0
            buf.append(block)
            buf_tokens = block_tokens
            continue

        _flush_buf()

        # After flushing, we have an overlap tail in the buffer. If even overlap+block
        # can't fit, drop the overlap tail and start a fresh chunk from this block.
        if buf_tokens + block_tokens > max_tokens:
            buf = []
            buf_tokens = 0

        buf.append(block)
        buf_tokens += block_tokens

    if buf:
        chunk_text = _join_blocks(buf)
        chunk_tokens = _count_tokens(chunk_text)
        start = char_cursor
        end = start + len(chunk_text)
        chunks.append(
            ChunkPiece(
                text=chunk_text,
                section_path=buf[-1].section_path,
                char_span_in_doc=(start, end),
                token_count=chunk_tokens,
            )
        )

    return chunks
