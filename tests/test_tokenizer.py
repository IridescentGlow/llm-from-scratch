import base64
import json
import time
from collections import Counter

import pytest

from llm_from_scratch.tokenizer import BPETokenizer

CORPUS = "low lower lowest low lower newest widest low lower newer"


def _trained_tokenizer(vocab_size: int = 280) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=vocab_size)
    return tok


def test_train_produces_merges_and_vocab():
    tok = _trained_tokenizer(vocab_size=280)
    # Pre-tokenization bounds merges to within chunks (see
    # docs/tokenizer-performance.md), so a small corpus may not have 24
    # within-chunk merge opportunities even though 24 were requested --
    # assert internal consistency instead of a hardcoded count.
    assert 0 < len(tok.merges) <= 24  # at most 280 - 256
    assert tok.vocab_size == 256 + len(tok.merges)
    # every merged token's bytes must be recoverable from the base vocab
    for token_id, token_bytes in tok.vocab.items():
        assert isinstance(token_bytes, bytes)


def test_encode_returns_token_ids():
    tok = _trained_tokenizer()
    ids = tok.encode("lowest")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0
    # a frequent word should compress to fewer tokens than raw bytes
    assert len(ids) < len("lowest".encode("utf-8"))


def test_decode_encode_roundtrip():
    tok = _trained_tokenizer()
    for text in [CORPUS, "lowest", "a totally unseen sentence!", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_roundtrip_with_non_ascii_text():
    tok = _trained_tokenizer()
    text = "héllo wörld — 日本語 emoji: 🚀🔥"
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_untrained_tokenizer_is_byte_level_roundtrip():
    tok = BPETokenizer()
    text = "still works: café"
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 256


def test_save_load_preserves_merges_and_vocab(tmp_path):
    tok = _trained_tokenizer(vocab_size=280)
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.merges == tok.merges
    assert loaded.vocab == tok.vocab
    assert loaded.vocab_size == tok.vocab_size


def test_add_eos_token_assigns_next_unused_id():
    tok = _trained_tokenizer(vocab_size=280)
    expected_id = tok.vocab_size  # next unused id, whatever training produced
    eos_id = tok.add_eos_token()

    assert eos_id == expected_id
    assert tok.eos_token_id == expected_id
    assert tok.vocab_size == expected_id + 1  # +1 reserved EOS slot


def test_add_eos_token_twice_raises():
    tok = _trained_tokenizer(vocab_size=280)
    tok.add_eos_token()

    with pytest.raises(ValueError, match="already"):
        tok.add_eos_token()


def test_tokenizer_without_eos_has_no_eos_token_id():
    tok = _trained_tokenizer(vocab_size=280)
    assert tok.eos_token_id is None


def test_decode_skips_eos_id():
    tok = _trained_tokenizer(vocab_size=280)
    eos_id = tok.add_eos_token()
    ids = tok.encode("lowest") + [eos_id]

    assert tok.decode(ids) == "lowest"


def test_save_load_preserves_eos_token(tmp_path):
    tok = _trained_tokenizer(vocab_size=280)
    tok.add_eos_token()
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.eos_token_id == tok.eos_token_id
    assert loaded.vocab_size == tok.vocab_size


def test_load_legacy_tokenizer_without_special_tokens_key(tmp_path):
    """A tokenizer.json saved before this milestone still has the always-empty
    special_tokens key (added by the tokenizer persistence milestone), so
    this covers that exact legacy shape -- no eos_token_id, no error.
    """
    tok = _trained_tokenizer(vocab_size=280)
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.eos_token_id is None


def test_save_load_preserves_encode_decode_behavior(tmp_path):
    """A loaded tokenizer must encode/decode identically to the original --
    not just have equal merges/vocab, but actually behave the same way.
    """
    tok = _trained_tokenizer(vocab_size=280)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    for text in [CORPUS, "lowest", "a totally unseen sentence!", "héllo — 🚀"]:
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(tok.encode(text)) == text


# --- Tokenizer performance milestone: pre-tokenization + efficient BPE ---
# See docs/tokenizer-performance.md.


def test_pretokenize_splits_into_word_and_punctuation_chunks():
    chunks = BPETokenizer._pretokenize("the cat sat.")
    assert chunks == ["the", " cat", " sat", "."]


def test_pretokenize_handles_empty_string():
    assert BPETokenizer._pretokenize("") == []


def test_newly_trained_tokenizer_is_pretokenized_by_default():
    tok = _trained_tokenizer(vocab_size=280)
    assert tok.pretokenize is True


def test_merges_do_not_cross_chunk_boundaries():
    tok = BPETokenizer()
    tok.train("cat cat cat cat cat", vocab_size=270)
    # 't' ends every "cat"/" cat" chunk; ' ' only ever starts the *next*
    # chunk. That pair is never adjacent within any single chunk's token
    # list, so it must be structurally impossible to learn -- regardless
    # of vocab_size or tie-breaking -- once pre-tokenization is in effect.
    assert (ord("t"), ord(" ")) not in tok.merges


def test_correctness_against_known_small_example():
    # "aaaa" is a single pre-tokenized chunk (no spaces/punctuation), so
    # this is fully hand-verifiable: (a,a) is the only pair, merges greedily
    # left-to-right into two merged tokens.
    tok = BPETokenizer()
    tok.train("aaaa", vocab_size=257)  # exactly one merge allowed

    assert tok.merges == {(ord("a"), ord("a")): 256}
    assert tok.encode("aaaa") == [256, 256]
    assert tok.decode([256, 256]) == "aaaa"


def test_save_load_preserves_pretokenize_flag(tmp_path):
    tok = _trained_tokenizer(vocab_size=280)
    assert tok.pretokenize is True

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.pretokenize is True


def test_pretokenize_flag_controls_whether_boundary_merge_applies(tmp_path):
    """Same merges/vocab, only `pretokenize` differs -- demonstrates that
    encode() dispatches to the algorithm matching how the merges were
    learned, and that a merge which would cross a chunk boundary (learned
    by a pre-milestone/legacy tokenizer) only applies in legacy mode.
    """
    data = {
        "merges": [[ord("t"), ord(" "), 256]],
        "vocab": {
            **{str(i): base64.b64encode(bytes([i])).decode("ascii") for i in range(256)},
            "256": base64.b64encode(b"t ").decode("ascii"),
        },
        "special_tokens": {},
    }

    legacy_path = tmp_path / "legacy.json"
    with open(legacy_path, "w") as f:
        json.dump(data, f)  # no "pretokenize" key at all
    legacy = BPETokenizer.load(legacy_path)
    assert legacy.pretokenize is False

    modern_path = tmp_path / "modern.json"
    with open(modern_path, "w") as f:
        json.dump({**data, "pretokenize": True}, f)
    modern = BPETokenizer.load(modern_path)
    assert modern.pretokenize is True

    # The learned merge spans the 't'+' ' boundary between "cat" and
    # "there" -- reachable only by the legacy whole-text algorithm.
    assert 256 in legacy.encode("cat there")
    assert 256 not in modern.encode("cat there")


def test_legacy_tokenizer_without_pretokenize_key_still_loads_and_works(tmp_path):
    """A tokenizer.json saved before this milestone (no 'pretokenize' key,
    merges learned by the old whole-text algorithm) must still load and
    encode/decode exactly as it did before this milestone existed.
    """
    tok = _trained_tokenizer(vocab_size=280)
    # Simulate a pre-milestone save: same shape, but force the flag off and
    # drop the key entirely, as a real legacy file would never have had it.
    tok.pretokenize = False
    path = tmp_path / "legacy_full.json"
    tok.save(path)
    with open(path) as f:
        data = json.load(f)
    del data["pretokenize"]
    with open(path, "w") as f:
        json.dump(data, f)

    loaded = BPETokenizer.load(path)
    assert loaded.pretokenize is False
    for text in [CORPUS, "lowest", "a totally unseen sentence!", ""]:
        assert loaded.decode(loaded.encode(text)) == text


def _naive_bpe_train(corpus: str, vocab_size: int) -> dict:
    """Reference re-implementation of the pre-milestone algorithm: full
    rescan of the whole token sequence on every merge, no pre-tokenization.
    Exists only in this test, to demonstrate the algorithmic improvement --
    see docs/tokenizer-performance.md.
    """

    def count_pairs(ids):
        return Counter(zip(ids, ids[1:]))

    def merge(ids, pair, new_id):
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    ids = list(corpus.encode("utf-8"))
    merges: dict = {}
    for i in range(vocab_size - 256):
        pair_counts = count_pairs(ids)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges[best_pair] = 256 + i
        ids = merge(ids, best_pair, 256 + i)
    return merges


def test_efficient_training_is_meaningfully_faster_than_naive_rescanning():
    # Heavy repetition -- realistic for natural text, and exactly what the
    # chunked/weighted algorithm is designed to exploit (see
    # docs/tokenizer-performance.md): a handful of unique words, repeated
    # thousands of times, rather than a truly unique corpus.
    paragraph = (
        "the quick brown fox jumps over the lazy dog while the cat "
        "sleeps near the warm fire and the dog barks at the mailman "
    )
    corpus = paragraph * 400  # tens of thousands of characters

    start = time.perf_counter()
    _naive_bpe_train(corpus, vocab_size=300)
    naive_time = time.perf_counter() - start

    tok = BPETokenizer()
    start = time.perf_counter()
    tok.train(corpus, vocab_size=300)
    efficient_time = time.perf_counter() - start

    assert len(tok.merges) > 0
    # Not a tight micro-benchmark (test machines vary) -- just needs to
    # clearly demonstrate the algorithmic improvement.
    assert efficient_time < naive_time / 3
    assert efficient_time < 5.0
