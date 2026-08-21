from llm_from_scratch.tokenizer import BPETokenizer

CORPUS = "low lower lowest low lower newest widest low lower newer"


def _trained_tokenizer(vocab_size: int = 280) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=vocab_size)
    return tok


def test_train_produces_merges_and_vocab():
    tok = _trained_tokenizer(vocab_size=280)
    assert len(tok.merges) == 24  # 280 - 256
    assert tok.vocab_size == 280
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
