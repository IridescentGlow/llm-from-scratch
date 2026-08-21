import torch

from llm_from_scratch.model import GPT, GPTConfig


def _tiny_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=50,
        context_length=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_model_constructs():
    model = GPT(_tiny_config())
    assert isinstance(model, torch.nn.Module)
    assert model.config.vocab_size == 50


def test_forward_logits_shape():
    config = _tiny_config()
    model = GPT(config)
    batch_size, seq_len = 4, 5
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    config = _tiny_config()
    model = GPT(config)
    batch_size, seq_len = 4, 5
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0


def test_forward_rejects_sequence_longer_than_context_length():
    config = _tiny_config(context_length=4)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))
    try:
        model(idx)
        assert False, "expected ValueError for sequence longer than context_length"
    except ValueError:
        pass


def test_attention_is_causal():
    """Changing a later token must not change an earlier position's logits."""
    config = _tiny_config()
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 6))
    logits_a, _ = model(idx)

    idx_modified = idx.clone()
    idx_modified[0, -1] = (idx_modified[0, -1] + 1) % config.vocab_size
    logits_b, _ = model(idx_modified)

    # positions before the modified (last) token must be unaffected
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)
    # the modified position's own logits (or later ones) generally should differ
    assert not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :], atol=1e-5)


def test_generate_extends_sequence_with_valid_token_ids():
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 3))

    out = model.generate(idx, max_new_tokens=4)

    assert out.shape == (2, 3 + 4)
    assert torch.equal(out[:, :3], idx)  # original tokens preserved
    assert out.min() >= 0
    assert out.max() < config.vocab_size


def test_generate_respects_context_length_when_prompt_exceeds_it():
    config = _tiny_config(context_length=4)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 6))  # longer than context_length

    out = model.generate(idx, max_new_tokens=2)

    assert out.shape == (1, 6 + 2)


def test_generate_greedy_is_deterministic():
    """temperature <= 0 (the default) must always pick argmax, same output every call."""
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 3))

    out_a = model.generate(idx, max_new_tokens=5, temperature=0.0)
    out_b = model.generate(idx, max_new_tokens=5, temperature=0.0)

    assert torch.equal(out_a, out_b)


def test_generate_temperature_sampling_produces_valid_token_ids():
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 3))

    out = model.generate(idx, max_new_tokens=4, temperature=1.0)

    assert out.shape == (2, 3 + 4)
    assert torch.equal(out[:, :3], idx)  # original tokens preserved
    assert out.min() >= 0
    assert out.max() < config.vocab_size
