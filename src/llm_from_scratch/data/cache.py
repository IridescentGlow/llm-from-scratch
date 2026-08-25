"""Token cache: skip re-encoding the raw corpus when a prior tokens.bin
already holds the exact result. See docs/token-cache.md."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from llm_from_scratch.tokenizer import BPETokenizer

from .tokens import TOKEN_DTYPE, write_token_ids

_TOKEN_ITEMSIZE = np.dtype(TOKEN_DTYPE).itemsize


@dataclass
class CacheMetadata:
    corpus_hash: str
    tokenizer_hash: str
    tokens_hash: str
    num_tokens: int


def _hash_corpus(corpus: str) -> str:
    return hashlib.sha256(corpus.encode("utf-8")).hexdigest()


def _hash_tokenizer(tokenizer: BPETokenizer) -> str:
    """Hash of the tokenizer's serialized form (same shape `save()` writes).

    Serializes via `save()` to a temp file rather than duplicating the
    merges/vocab/special_tokens/pretokenize dict-building logic here, so
    this can never silently drift out of sync with the actual persistence
    format. See docs/01-tokenization.md, "Tokenizer persistence".
    """
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        tokenizer.save(tmp.name)
        return hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_cache_metadata(path: Path, metadata: CacheMetadata) -> None:
    with open(path, "w") as f:
        json.dump(asdict(metadata), f)


def _load_cache_metadata(path: Path) -> CacheMetadata | None:
    """Returns None for anything short of a fully valid metadata file --
    missing, corrupt JSON, or missing fields. A broken cache is always
    safe to treat as "no cache" and rebuild; see docs/token-cache.md."""
    try:
        with open(path) as f:
            data = json.load(f)
        return CacheMetadata(
            corpus_hash=data["corpus_hash"],
            tokenizer_hash=data["tokenizer_hash"],
            tokens_hash=data["tokens_hash"],
            num_tokens=data["num_tokens"],
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def ensure_token_cache(
    corpus: str, tokenizer: BPETokenizer, tokens_path: str | Path, meta_path: str | Path
) -> bool:
    """Make sure `tokens_path` holds `tokenizer.encode(corpus)`, reusing a
    prior encoding if the cache is still valid.

    Returns True if a valid cache was reused (no encoding happened), False
    if the corpus was (re-)encoded and both files were (re-)written.
    See docs/token-cache.md for the validity rules.
    """
    tokens_path = Path(tokens_path)
    meta_path = Path(meta_path)

    corpus_hash = _hash_corpus(corpus)
    tokenizer_hash = _hash_tokenizer(tokenizer)

    metadata = _load_cache_metadata(meta_path)
    if (
        metadata is not None
        and metadata.corpus_hash == corpus_hash
        and metadata.tokenizer_hash == tokenizer_hash
        and tokens_path.exists()
        and tokens_path.stat().st_size == metadata.num_tokens * _TOKEN_ITEMSIZE
        and _hash_file(tokens_path) == metadata.tokens_hash
    ):
        return True

    token_ids = tokenizer.encode(corpus)
    write_token_ids(token_ids, tokens_path)
    _save_cache_metadata(
        meta_path,
        CacheMetadata(
            corpus_hash=corpus_hash,
            tokenizer_hash=tokenizer_hash,
            tokens_hash=_hash_file(tokens_path),
            num_tokens=len(token_ids),
        ),
    )
    return False
