"""Tests for the token cache. See docs/token-cache.md."""

import json

from llm_from_scratch.data import ensure_token_cache, load_token_ids
from llm_from_scratch.tokenizer import BPETokenizer


def _tiny_tokenizer(corpus: str) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=260)
    return tokenizer


def test_first_run_creates_cache(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"

    reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    assert reused is False
    assert tokens_path.exists()
    assert meta_path.exists()
    loaded = load_token_ids(tokens_path)
    assert loaded.tolist() == tokenizer.encode(corpus)


def test_valid_cache_is_reused_without_reencoding(tmp_path, monkeypatch):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    original_encode = tokenizer.encode

    def fail_if_called(*args, **kwargs):
        raise AssertionError("encode() should not be called on a valid-cache hit")

    monkeypatch.setattr(tokenizer, "encode", fail_if_called)

    reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    assert reused is True
    # sanity: the file on disk still holds what the real encode() produces
    monkeypatch.setattr(tokenizer, "encode", original_encode)
    assert load_token_ids(tokens_path).tolist() == original_encode(corpus)


def test_unchanged_corpus_and_tokenizer_produce_identical_token_ids(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"

    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
    first = load_token_ids(tokens_path).tolist()

    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
    second = load_token_ids(tokens_path).tolist()

    assert first == second == tokenizer.encode(corpus)


def test_changed_corpus_invalidates_cache(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    new_corpus = "the dog sat. the dog ran. a completely different sentence."
    reused = ensure_token_cache(new_corpus, tokenizer, tokens_path, meta_path)

    assert reused is False
    assert load_token_ids(tokens_path).tolist() == tokenizer.encode(new_corpus)


def test_changed_tokenizer_invalidates_cache(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer_a = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer_a, tokens_path, meta_path)

    tokenizer_b = BPETokenizer()
    tokenizer_b.train(corpus, vocab_size=270)  # different vocab_size -> different merges

    reused = ensure_token_cache(corpus, tokenizer_b, tokens_path, meta_path)

    assert reused is False
    assert load_token_ids(tokens_path).tolist() == tokenizer_b.encode(corpus)


def test_missing_metadata_rebuilds_without_error(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
    meta_path.unlink()

    reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    assert reused is False
    assert meta_path.exists()
    assert load_token_ids(tokens_path).tolist() == tokenizer.encode(corpus)


def test_corrupt_metadata_rebuilds_without_error(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
    meta_path.write_text("{ not valid json")

    reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    assert reused is False
    assert json.loads(meta_path.read_text())  # a fresh, valid metadata file now
    assert load_token_ids(tokens_path).tolist() == tokenizer.encode(corpus)


def test_missing_tokens_bin_rebuilds_without_error(tmp_path):
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)
    tokens_path.unlink()

    reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    assert reused is False
    assert tokens_path.exists()


def test_tokens_bin_overwritten_by_something_else_is_detected(tmp_path):
    """Simulates scripts/finetune.py writing a different token stream to
    the same tokens.bin path without updating this metadata -- see
    docs/token-cache.md, "Known interaction with scripts/finetune.py"."""
    corpus = "the cat sat. the cat ran."
    tokenizer = _tiny_tokenizer(corpus)
    tokens_path = tmp_path / "tokens.bin"
    meta_path = tmp_path / "tokens.meta.json"
    ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    from llm_from_scratch.data import write_token_ids

    write_token_ids([1, 2, 3, 4, 5], tokens_path)  # unrelated overwrite, metadata untouched

    reused = ensure_token_cache(corpus, tokenizer, tokens_path, meta_path)

    assert reused is False
    assert load_token_ids(tokens_path).tolist() == tokenizer.encode(corpus)
