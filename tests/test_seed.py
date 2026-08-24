import numpy as np
import torch

from llm_from_scratch.data import TokenDataset
from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.seed import set_seed
from llm_from_scratch.train import TrainConfig, train_model


def _tiny_model_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=20,
        context_length=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.2,  # nonzero so dropout randomness is actually exercised
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _tiny_train_config(**overrides) -> TrainConfig:
    defaults = dict(
        batch_size=4,
        learning_rate=1e-2,
        min_lr=1e-3,
        warmup_steps=2,
        max_steps=5,
        grad_clip=1.0,
        eval_every=5,
        checkpoint_dir="",  # set per-test via tmp_path
        checkpoint_every=5,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _tiny_dataset(vocab_size: int, context_length: int, length: int = 200) -> TokenDataset:
    # Not seeded -- this is fixture data, not part of what the seed under
    # test is responsible for reproducing.
    tokens = np.random.default_rng(0).integers(0, vocab_size, size=length)
    return TokenDataset(tokens, context_length=context_length)


def test_same_seed_produces_identical_model_init():
    config = _tiny_model_config()

    set_seed(42)
    model_a = GPT(config)

    set_seed(42)
    model_b = GPT(config)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_different_seeds_produce_different_model_init():
    config = _tiny_model_config()

    set_seed(1)
    model_a = GPT(config)

    set_seed(2)
    model_b = GPT(config)

    params_equal = all(
        torch.equal(p_a, p_b) for p_a, p_b in zip(model_a.parameters(), model_b.parameters())
    )
    assert not params_equal


def test_same_seed_produces_identical_training_result(tmp_path):
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt_a"))

    set_seed(7)
    model_a = GPT(model_config)
    train_dataset_a = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset_a = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)
    result_a = train_model(
        model_a, train_dataset_a, val_dataset_a, train_config, log_fn=lambda _msg: None
    )

    train_config_b = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt_b"))
    set_seed(7)
    model_b = GPT(model_config)
    train_dataset_b = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset_b = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)
    result_b = train_model(
        model_b, train_dataset_b, val_dataset_b, train_config_b, log_fn=lambda _msg: None
    )

    assert result_a["train_losses"] == result_b["train_losses"]
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_different_seeds_produce_different_training_result(tmp_path):
    model_config = _tiny_model_config()

    set_seed(1)
    model_a = GPT(model_config)
    train_dataset_a = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset_a = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)
    train_config_a = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt_a"))
    result_a = train_model(
        model_a, train_dataset_a, val_dataset_a, train_config_a, log_fn=lambda _msg: None
    )

    set_seed(2)
    model_b = GPT(model_config)
    train_dataset_b = _tiny_dataset(model_config.vocab_size, model_config.context_length)
    val_dataset_b = _tiny_dataset(model_config.vocab_size, model_config.context_length, length=80)
    train_config_b = _tiny_train_config(checkpoint_dir=str(tmp_path / "ckpt_b"))
    result_b = train_model(
        model_b, train_dataset_b, val_dataset_b, train_config_b, log_fn=lambda _msg: None
    )

    assert result_a["train_losses"] != result_b["train_losses"]


def test_greedy_generation_is_deterministic_regardless_of_seed():
    config = _tiny_model_config(dropout=0.0)
    set_seed(0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    out_a = model.generate(idx.clone(), max_new_tokens=5, temperature=0.0)
    out_b = model.generate(idx.clone(), max_new_tokens=5, temperature=0.0)

    assert torch.equal(out_a, out_b)


def test_seeded_stochastic_generation_is_reproducible():
    config = _tiny_model_config(dropout=0.0)
    set_seed(0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    set_seed(123)
    out_a = model.generate(idx.clone(), max_new_tokens=10, temperature=1.0)

    set_seed(123)
    out_b = model.generate(idx.clone(), max_new_tokens=10, temperature=1.0)

    assert torch.equal(out_a, out_b)


def test_different_seeds_can_produce_different_stochastic_generation():
    config = _tiny_model_config(dropout=0.0)
    set_seed(0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    set_seed(1)
    out_a = model.generate(idx.clone(), max_new_tokens=10, temperature=1.0)

    set_seed(2)
    out_b = model.generate(idx.clone(), max_new_tokens=10, temperature=1.0)

    assert not torch.equal(out_a, out_b)
