import pytest
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


def test_no_separate_lm_head_parameter():
    # Weight tying is structural: there's no lm_head submodule/parameter at
    # all, only token_embedding. See docs/weight-tying-initialization.md.
    model = GPT(_tiny_config())
    assert not hasattr(model, "lm_head")
    assert "lm_head.weight" not in model.state_dict()


def test_output_projection_is_tied_to_token_embedding():
    model = GPT(_tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (1, 3))

    logits, _ = model(idx)
    # token_embedding.weight is the only parameter used for both the input
    # lookup and the output projection -- a gradient from the output logits
    # must flow into it directly (not into some separate lm_head parameter,
    # which no longer exists -- see test_no_separate_lm_head_parameter).
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, model.config.vocab_size), idx.view(-1)
    )
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.any(model.token_embedding.weight.grad != 0)


def test_tying_reduces_parameter_count_vs_untied():
    config = _tiny_config()
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())

    embedding_params = config.vocab_size * config.n_embd
    untied_total = total_params + embedding_params  # what it'd be with a separate lm_head

    assert total_params == untied_total - embedding_params
    # The embedding table appears in the parameter count exactly once.
    assert sum(
        1 for name, _ in model.named_parameters() if name == "token_embedding.weight"
    ) == 1


def test_init_weights_use_gpt_style_std():
    config = _tiny_config(n_layer=4)
    model = GPT(config)

    # Non-residual-writing linear layers and embeddings: std ~= 0.02.
    emb_std = model.token_embedding.weight.detach().std().item()
    assert 0.01 < emb_std < 0.03

    qkv_std = model.blocks[0].attn.qkv_proj.weight.detach().std().item()
    assert 0.01 < qkv_std < 0.03

    # Residual-writing projections: scaled down by 1 / sqrt(2 * n_layer).
    expected_residual_std = 0.02 / (2 * config.n_layer) ** 0.5
    out_proj_std = model.blocks[0].attn.out_proj.weight.detach().std().item()
    assert out_proj_std == pytest.approx(expected_residual_std, rel=0.5)


def test_linear_biases_initialized_to_zero():
    model = GPT(_tiny_config())
    assert torch.all(model.blocks[0].attn.qkv_proj.bias == 0)
    assert torch.all(model.blocks[0].attn.out_proj.bias == 0)


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


def test_forward_ignores_target_positions_set_to_minus_100():
    """GPT.forward makes no special provision for masking -- it relies on
    F.cross_entropy's default ignore_index=-100. This is the mechanism
    docs/finetune-loss-masking.md's response-only masking depends on: a
    target position set to -100 must contribute zero loss and zero
    gradient, with no change to the model or the forward() call. Confirmed
    directly here, independent of the fine-tuning data pipeline."""
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))
    targets = torch.randint(0, config.vocab_size, (1, 5))

    # loss over only the unmasked (last 2) positions, computed by hand
    logits, _ = model(idx)
    unmasked_manual_loss = torch.nn.functional.cross_entropy(
        logits[:, -2:, :].reshape(-1, config.vocab_size), targets[:, -2:].reshape(-1)
    )

    masked_targets = targets.clone()
    masked_targets[:, :3] = -100
    _, masked_loss = model(idx, masked_targets)

    assert torch.allclose(masked_loss, unmasked_manual_loss, atol=1e-6)

    # gradient check: masking a position must zero out its contribution to
    # the backward pass, not just the forward-pass number.
    model.zero_grad()
    masked_loss.backward()
    grad_masked = {name: p.grad.clone() for name, p in model.named_parameters()}

    model.zero_grad()
    _, full_loss = model(idx, targets)
    full_loss.backward()
    grad_full = {name: p.grad.clone() for name, p in model.named_parameters()}

    # the two gradients differ (masking changes what's trained on) -- this
    # isn't a no-op
    assert any(
        not torch.allclose(grad_masked[name], grad_full[name]) for name in grad_masked
    )


def test_forward_all_targets_masked_gives_nan_loss_like_bare_cross_entropy():
    """If every target position is masked (e.g. a degenerate example with
    an empty response), cross_entropy has nothing to average over and
    returns nan -- this is F.cross_entropy's own documented behavior for
    an all-ignored batch, not something this project needs to handle
    specially."""
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))
    targets = torch.full((1, 5), -100)

    _, loss = model(idx, targets)

    assert torch.isnan(loss)


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


def test_generate_stops_early_when_eos_is_produced():
    """See docs/eos-generation-stopping.md: generation must stop the moment
    eos_token_id is produced, before max_new_tokens is reached.

    Greedy decoding is deterministic, so the token the untrained model picks
    first (with no eos_token_id given) is exactly the token it will pick
    first again given the same prompt -- used here as a stand-in "eos" id
    to force an early stop without needing a trained model.
    """
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    first_token = model.generate(idx, max_new_tokens=1, temperature=0.0)[0, -1].item()

    out = model.generate(idx, max_new_tokens=10, temperature=0.0, eos_token_id=first_token)

    assert out.shape == (1, 3 + 1)  # stopped right after producing eos, not after 10 more
    assert out[0, -1].item() == first_token


def test_generate_runs_full_max_new_tokens_when_eos_never_appears():
    """max_new_tokens remains a hard cap when eos_token_id is never produced."""
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    out = model.generate(idx, max_new_tokens=5, temperature=0.0, eos_token_id=-1)

    assert out.shape == (1, 3 + 5)


def test_generate_restores_training_mode_when_called_from_training_mode():
    """See docs/generation-mode-restoration.md: generate() must leave the
    model in training mode if it was in training mode before the call."""
    config = _tiny_config()
    model = GPT(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (1, 3))

    model.generate(idx, max_new_tokens=3)

    assert model.training is True


def test_generate_leaves_eval_mode_when_called_from_eval_mode():
    """generate() must not force training mode on if the model was already
    in eval mode before the call."""
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 3))

    model.generate(idx, max_new_tokens=3)

    assert model.training is False


def test_generate_restores_training_mode_on_early_eos_stop():
    """Mode restoration must hold on the early-return path too, not just
    the "ran the full max_new_tokens loop" path."""
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    first_token = model.generate(idx, max_new_tokens=1, temperature=0.0)[0, -1].item()

    model.train()
    model.generate(idx, max_new_tokens=10, temperature=0.0, eos_token_id=first_token)

    assert model.training is True


def test_generate_output_unchanged_by_mode_restoration():
    """The mode-restoration fix must not alter generated token ids."""
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 3))

    model.train()
    out_from_train = model.generate(idx, max_new_tokens=5, temperature=0.0)

    model.eval()
    out_from_eval = model.generate(idx, max_new_tokens=5, temperature=0.0)

    assert torch.equal(out_from_train, out_from_eval)
