import math

import numpy as np
import torch

from llm_from_scratch.data import TokenDataset
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.train import TrainConfig, train_model


def _tiny_model_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=20,
        context_length=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _tiny_dataset(vocab_size: int, context_length: int, length: int = 100) -> TokenDataset:
    tokens = np.random.default_rng(0).integers(0, vocab_size, size=length)
    return TokenDataset(tokens, context_length=context_length)


def test_evaluate_model_returns_expected_keys():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length)

    result = evaluate_model(model, dataset, batch_size=4)

    assert set(result) == {"loss", "perplexity", "num_batches", "num_tokens"}
    assert isinstance(result["loss"], float)
    assert isinstance(result["perplexity"], float)
    assert isinstance(result["num_batches"], int)
    assert isinstance(result["num_tokens"], int)


def test_perplexity_is_exp_of_loss():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length)

    result = evaluate_model(model, dataset, batch_size=4)

    assert math.isclose(result["perplexity"], math.exp(result["loss"]), rel_tol=1e-9)


def test_num_batches_and_tokens_cover_whole_dataset():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length, length=100)

    batch_size = 4
    result = evaluate_model(model, dataset, batch_size=batch_size)

    expected_num_batches = math.ceil(len(dataset) / batch_size)
    assert result["num_batches"] == expected_num_batches
    assert result["num_tokens"] == len(dataset) * config.context_length


def test_max_batches_limits_evaluation():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length, length=100)

    result = evaluate_model(model, dataset, batch_size=4, max_batches=3)

    assert result["num_batches"] == 3
    assert result["num_tokens"] == 3 * 4 * config.context_length


def test_evaluate_model_does_not_update_weights():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length)

    weights_before = model.lm_head.weight.clone()
    evaluate_model(model, dataset, batch_size=4)
    weights_after = model.lm_head.weight

    assert torch.equal(weights_before, weights_after)
    assert model.training  # restored to train mode afterwards


def test_evaluate_model_is_deterministic_across_repeated_calls():
    config = _tiny_model_config()
    model = GPT(config)
    dataset = _tiny_dataset(config.vocab_size, config.context_length)

    result_a = evaluate_model(model, dataset, batch_size=4)
    result_b = evaluate_model(model, dataset, batch_size=4)

    assert result_a == result_b


def test_evaluate_model_preserves_original_training_mode():
    config = _tiny_model_config()
    model = GPT(config)
    model.eval()
    dataset = _tiny_dataset(config.vocab_size, config.context_length)

    evaluate_model(model, dataset, batch_size=4)

    assert not model.training


def test_val_loss_and_perplexity_improve_after_training(tmp_path):
    """End-to-end sanity check: a model trained on a repeating pattern
    should evaluate with lower val loss/perplexity afterwards than before.
    """
    model_config = _tiny_model_config(vocab_size=10, context_length=8, n_embd=32)
    pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tokens = np.array(pattern * 20)
    dataset = TokenDataset(tokens, context_length=model_config.context_length)

    model = GPT(model_config)
    before = evaluate_model(model, dataset, batch_size=4)

    train_config = TrainConfig(
        batch_size=4,
        learning_rate=5e-3,
        warmup_steps=5,
        max_steps=100,
        grad_clip=1.0,
        eval_every=100,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    train_model(model, dataset, dataset, train_config, log_fn=lambda _msg: None)

    after = evaluate_model(model, dataset, batch_size=4)

    assert after["loss"] < before["loss"]
    assert after["perplexity"] < before["perplexity"]
