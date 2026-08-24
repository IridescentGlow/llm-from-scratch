from dataclasses import asdict

import torch

from llm_from_scratch.finetune import load_pretrained_model
from llm_from_scratch.generate import generate
from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.tokenizer import BPETokenizer


def _tiny_model_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=256,  # byte-level base vocab only, no merges needed
        context_length=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _tokenizer(vocab_size: int) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train("hello world, this is a tiny corpus for testing.", vocab_size=vocab_size)
    return tokenizer


def test_generate_decodes_to_a_string():
    config = _tiny_model_config()
    model = GPT(config)
    tokenizer = _tokenizer(config.vocab_size)

    text = generate(model, tokenizer, "hello", max_new_tokens=5)

    assert isinstance(text, str)
    assert text.startswith("hello")


def test_generate_respects_max_new_tokens():
    """generate() must add exactly max_new_tokens ids before decoding.

    Checked by monkeypatching GPT.generate to record the sequence length it
    was asked to return -- decoding to text can't be used to count tokens,
    since replacement characters for invalid UTF-8 bytes don't round-trip
    to the same token count on re-encode.
    """
    config = _tiny_model_config()
    model = GPT(config)
    tokenizer = _tokenizer(config.vocab_size)
    prompt_len = len(tokenizer.encode("hello"))

    original_generate = GPT.generate
    seen_shapes = []

    def spy_generate(self, idx, max_new_tokens, temperature=0.0, eos_token_id=None):
        out = original_generate(self, idx, max_new_tokens, temperature, eos_token_id)
        seen_shapes.append(out.shape)
        return out

    model.generate = spy_generate.__get__(model, GPT)
    generate(model, tokenizer, "hello", max_new_tokens=5)

    assert seen_shapes == [(1, prompt_len + 5)]


def test_generate_truncates_long_prompt_to_context_length():
    config = _tiny_model_config(context_length=4)
    model = GPT(config)
    tokenizer = _tokenizer(config.vocab_size)

    long_prompt = "hello world, this is a tiny corpus for testing." * 3
    # Should not raise even though the prompt is far longer than context_length.
    text = generate(model, tokenizer, long_prompt, max_new_tokens=2)

    assert isinstance(text, str)


def test_generate_passes_eos_token_id_through_to_model_generate():
    """See docs/eos-generation-stopping.md: generate() must forward the
    tokenizer's eos_token_id to GPT.generate so it can stop early.
    """
    config = _tiny_model_config()
    model = GPT(config)
    tokenizer = _tokenizer(config.vocab_size)
    tokenizer.add_eos_token()

    seen_eos_ids = []
    original_generate = GPT.generate

    def spy_generate(self, idx, max_new_tokens, temperature=0.0, eos_token_id=None):
        seen_eos_ids.append(eos_token_id)
        return original_generate(self, idx, max_new_tokens, temperature, eos_token_id)

    model.generate = spy_generate.__get__(model, GPT)
    generate(model, tokenizer, "hello", max_new_tokens=5)

    assert seen_eos_ids == [tokenizer.eos_token_id]


def test_generate_with_legacy_tokenizer_passes_no_eos_id():
    """A tokenizer with no EOS token (legacy, pre-milestone) must not
    invent one -- generate() passes None through, unchanged behavior.
    """
    config = _tiny_model_config()
    model = GPT(config)
    tokenizer = _tokenizer(config.vocab_size)
    assert tokenizer.eos_token_id is None

    seen_eos_ids = []
    original_generate = GPT.generate

    def spy_generate(self, idx, max_new_tokens, temperature=0.0, eos_token_id=None):
        seen_eos_ids.append(eos_token_id)
        return original_generate(self, idx, max_new_tokens, temperature, eos_token_id)

    model.generate = spy_generate.__get__(model, GPT)
    generate(model, tokenizer, "hello", max_new_tokens=5)

    assert seen_eos_ids == [None]


def test_generate_reuses_load_pretrained_model(tmp_path):
    config = _tiny_model_config()
    model = GPT(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": asdict(config)}, checkpoint_path
    )

    loaded = load_pretrained_model(checkpoint_path)
    tokenizer = _tokenizer(config.vocab_size)

    text = generate(loaded, tokenizer, "hello", max_new_tokens=3)

    assert isinstance(text, str)
