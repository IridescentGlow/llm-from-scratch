import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from llm_from_scratch.data import TokenDataset
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.finetune import (
    InstructionExample,
    build_corpus,
    build_masked_token_ids,
    build_token_ids,
    format_example,
    load_examples_jsonl,
    load_pretrained_model,
    load_tokenizer_for_checkpoint,
)
from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.tokenizer import BPETokenizer
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


def test_format_example_uses_fixed_template():
    example = InstructionExample(instruction="Translate 'hi' to French.", response="Salut.")
    text = format_example(example)
    assert text == "Instruction: Translate 'hi' to French.\nResponse: Salut.\n"


def test_build_corpus_joins_multiple_examples():
    examples = [
        InstructionExample(instruction="A?", response="a."),
        InstructionExample(instruction="B?", response="b."),
    ]
    corpus = build_corpus(examples)
    assert "Instruction: A?\nResponse: a.\n" in corpus
    assert "Instruction: B?\nResponse: b.\n" in corpus
    # Both examples present in order, separated.
    assert corpus.index("A?") < corpus.index("B?")


def test_load_examples_jsonl_reads_file(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps({"instruction": "A?", "response": "a."})
        + "\n"
        + json.dumps({"instruction": "B?", "response": "b."})
        + "\n"
    )

    examples = load_examples_jsonl(path)

    assert examples == [
        InstructionExample(instruction="A?", response="a."),
        InstructionExample(instruction="B?", response="b."),
    ]


def test_load_pretrained_model_restores_weights(tmp_path):
    config = _tiny_model_config()
    model = GPT(config)
    checkpoint_path = tmp_path / "pretrained.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": asdict(config)}, checkpoint_path
    )

    loaded = load_pretrained_model(checkpoint_path)

    assert loaded.config == config
    for name, param in model.named_parameters():
        assert torch.equal(param, loaded.state_dict()[name])


def test_load_pretrained_model_rejects_legacy_pickled_config(tmp_path):
    """A checkpoint saved before the checkpoint format hardening milestone
    has model_config pickled as a real GPTConfig object -- load_pretrained_model
    (used by scripts/generate.py, scripts/finetune.py, and scripts/evaluate.py)
    must fail clearly, not silently accept it via weights_only=False. See
    docs/checkpoint-format.md.
    """
    config = _tiny_model_config()
    model = GPT(config)
    checkpoint_path = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": config}, checkpoint_path)

    with pytest.raises(ValueError, match="[Rr]egenerate"):
        load_pretrained_model(checkpoint_path)


def test_load_pretrained_model_rejects_legacy_untied_checkpoint(tmp_path):
    """A checkpoint saved before weight tying has an independent
    lm_head.weight key that the tied GPT has no parameter for. strict=True
    state_dict loading must fail loudly, not silently drop or merge it --
    see docs/weight-tying-initialization.md, "What happens to old (untied,
    default-init) checkpoints".
    """
    config = _tiny_model_config()
    model = GPT(config)
    state_dict = model.state_dict()
    # Simulate a pre-tying checkpoint: an extra, independently-trained
    # lm_head.weight key alongside token_embedding.weight.
    state_dict["lm_head.weight"] = torch.randn(config.vocab_size, config.n_embd)
    checkpoint_path = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": state_dict, "model_config": asdict(config)}, checkpoint_path)

    with pytest.raises(RuntimeError, match="lm_head.weight"):
        load_pretrained_model(checkpoint_path)


def test_load_tokenizer_for_checkpoint_loads_saved_tokenizer(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train("hello world, this is a tiny corpus for testing.", vocab_size=260)
    checkpoint_path = tmp_path / "pretrained.pt"
    checkpoint_path.write_bytes(b"")  # only the path/parent matter here
    tokenizer.save(tmp_path / "tokenizer.json")

    loaded = load_tokenizer_for_checkpoint(checkpoint_path)

    assert loaded.merges == tokenizer.merges
    assert loaded.vocab == tokenizer.vocab


def test_load_tokenizer_for_checkpoint_raises_clear_error_when_missing(tmp_path):
    """A checkpoint saved before tokenizer persistence existed has no
    tokenizer.json -- this must fail loudly, not silently retrain a
    (possibly different) tokenizer. See docs/01-tokenization.md,
    "Migration and compatibility with existing checkpoints".
    """
    checkpoint_path = tmp_path / "pretrained.pt"
    checkpoint_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="predates tokenizer persistence"):
        load_tokenizer_for_checkpoint(checkpoint_path)


def test_build_token_ids_appends_eos_after_each_example():
    tokenizer = BPETokenizer()
    tokenizer.train("Instruction: A?\nResponse: a.\nInstruction: B?\nResponse: b.\n", vocab_size=260)
    tokenizer.add_eos_token()
    examples = [
        InstructionExample(instruction="A?", response="a."),
        InstructionExample(instruction="B?", response="b."),
    ]

    token_ids = build_token_ids(examples, tokenizer)

    expected = (
        tokenizer.encode(format_example(examples[0]))
        + [tokenizer.eos_token_id]
        + tokenizer.encode(format_example(examples[1]))
        + [tokenizer.eos_token_id]
    )
    assert token_ids == expected


def test_build_token_ids_raises_clearly_when_tokenizer_has_no_eos():
    """See docs/eos-generation-stopping.md: fine-tuning against a checkpoint
    whose tokenizer predates EOS must fail loudly, not silently proceed
    without ever teaching the model to stop.
    """
    tokenizer = BPETokenizer()
    tokenizer.train("Instruction: A?\nResponse: a.\n", vocab_size=260)
    examples = [InstructionExample(instruction="A?", response="a.")]

    with pytest.raises(ValueError, match="no EOS token"):
        build_token_ids(examples, tokenizer)


def test_build_masked_token_ids_masks_prompt_and_keeps_response_and_eos():
    """See docs/finetune-loss-masking.md: prompt tokens (through the
    'Response: ' label) are False, response tokens and EOS are True."""
    tokenizer = BPETokenizer()
    tokenizer.train("Instruction: A?\nResponse: a.\n", vocab_size=260)
    tokenizer.add_eos_token()
    examples = [InstructionExample(instruction="A?", response="a.")]

    token_ids, loss_mask = build_masked_token_ids(examples, tokenizer)

    prefix_ids = tokenizer.encode("Instruction: A?\nResponse: ")
    full_ids = tokenizer.encode(format_example(examples[0]))
    expected_ids = full_ids + [tokenizer.eos_token_id]
    expected_mask = [False] * len(prefix_ids) + [True] * (len(full_ids) - len(prefix_ids)) + [True]

    assert token_ids == expected_ids
    assert loss_mask == expected_mask
    assert len(loss_mask) == len(token_ids)
    # the EOS position itself is unmasked -- it must be a supervised target
    assert loss_mask[-1] is True


def test_build_masked_token_ids_across_multiple_examples_stays_aligned():
    tokenizer = BPETokenizer()
    tokenizer.train("Instruction: A?\nResponse: a.\nInstruction: B?\nResponse: b.\n", vocab_size=260)
    tokenizer.add_eos_token()
    examples = [
        InstructionExample(instruction="A?", response="a."),
        InstructionExample(instruction="B?", response="b."),
    ]

    token_ids, loss_mask = build_masked_token_ids(examples, tokenizer)

    assert len(loss_mask) == len(token_ids)
    # every example boundary (EOS) is unmasked
    eos_positions = [i for i, t in enumerate(token_ids) if t == tokenizer.eos_token_id]
    assert len(eos_positions) == 2
    for pos in eos_positions:
        assert loss_mask[pos] is True
    # at least one prompt token (the very first, "Instruction:") is masked
    assert loss_mask[0] is False


def test_build_masked_token_ids_raises_clearly_when_tokenizer_has_no_eos():
    tokenizer = BPETokenizer()
    tokenizer.train("Instruction: A?\nResponse: a.\n", vocab_size=260)
    examples = [InstructionExample(instruction="A?", response="a.")]

    with pytest.raises(ValueError, match="no EOS token"):
        build_masked_token_ids(examples, tokenizer)


def test_finetuning_improves_loss_on_instruction_corpus(tmp_path):
    """End-to-end: fine-tuning a pretrained checkpoint on a tiny instruction
    corpus should lower val loss/perplexity, same reuse of train_model/
    evaluate_model as pretraining and Stage 5 evaluation.
    """
    model_config = _tiny_model_config(vocab_size=20, context_length=8, n_embd=32)
    model = GPT(model_config)

    pattern = list(range(1, 10))
    tokens = np.array(pattern * 20)
    dataset = TokenDataset(tokens, context_length=model_config.context_length)

    before = evaluate_model(model, dataset, batch_size=4)

    finetune_config = TrainConfig(
        batch_size=4,
        learning_rate=5e-3,
        min_lr=5e-4,
        warmup_steps=5,
        max_steps=100,
        grad_clip=1.0,
        eval_every=100,
        checkpoint_dir=str(tmp_path / "finetuned"),
        checkpoint_every=100,
        weight_decay=0.01,
    )
    result = train_model(model, dataset, dataset, finetune_config, log_fn=lambda _msg: None)

    after = evaluate_model(model, dataset, batch_size=4)

    assert after["loss"] < before["loss"]
    assert after["perplexity"] < before["perplexity"]

    checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
    assert checkpoint["model_config"] == asdict(model_config)


def test_masked_finetuning_trains_only_on_response_and_eos_positions(tmp_path):
    """Controlled comparison: build the same tiny instruction corpus both
    with and without loss masking, run one training step each from
    identical starting weights, and confirm the resulting parameter
    updates differ -- masking changes what's actually trained on, not
    just what's reported. See docs/finetune-loss-masking.md."""
    tokenizer = BPETokenizer()
    tokenizer.train(
        "Instruction: A?\nResponse: alpha.\nInstruction: B?\nResponse: beta.\n" * 5,
        vocab_size=260,
    )
    tokenizer.add_eos_token()
    examples = [
        InstructionExample(instruction="A?", response="alpha."),
        InstructionExample(instruction="B?", response="beta."),
    ] * 5

    unmasked_token_ids = build_token_ids(examples, tokenizer)
    masked_token_ids, loss_mask = build_masked_token_ids(examples, tokenizer)
    assert unmasked_token_ids == masked_token_ids  # same underlying tokens either way
    assert len(loss_mask) == len(masked_token_ids)
    assert any(m is False for m in loss_mask)  # some positions actually get masked
    assert any(m is True for m in loss_mask)  # and some don't

    model_config = _tiny_model_config(
        vocab_size=tokenizer.vocab_size, context_length=16, n_embd=32
    )

    def _one_step_params(loss_mask_array):
        torch.manual_seed(0)
        model = GPT(model_config)
        tokens = np.array(masked_token_ids)
        dataset = TokenDataset(tokens, context_length=model_config.context_length, loss_mask=loss_mask_array)
        config = TrainConfig(
            batch_size=2,
            learning_rate=1e-2,
            min_lr=1e-3,
            warmup_steps=0,
            max_steps=1,
            grad_clip=1.0,
            eval_every=1,
            checkpoint_dir=str(tmp_path / f"run_{loss_mask_array is None}"),
            checkpoint_every=1,
            weight_decay=0.01,
        )
        torch.manual_seed(0)  # re-seed so DataLoader shuffling matches across the two runs
        train_model(model, dataset, dataset, config, log_fn=lambda _msg: None)
        return {name: p.detach().clone() for name, p in model.named_parameters()}

    params_unmasked = _one_step_params(None)
    params_masked = _one_step_params(np.array(loss_mask))

    assert any(
        not torch.allclose(params_unmasked[name], params_masked[name])
        for name in params_unmasked
    )


def test_masked_finetuning_end_to_end_improves_response_loss(tmp_path):
    """End-to-end smoke test of the actual fine-tuning path
    (build_masked_token_ids -> TokenDataset(loss_mask=...) -> train_model
    -> evaluate_model), mirroring what scripts/finetune.py now does.
    Fine-tuning with masking must still lower val loss, same guarantee
    Stage 6 already had without masking."""
    tokenizer = BPETokenizer()
    tokenizer.train(
        "Instruction: A?\nResponse: alpha.\nInstruction: B?\nResponse: beta.\n" * 5,
        vocab_size=260,
    )
    tokenizer.add_eos_token()
    examples = [
        InstructionExample(instruction="A?", response="alpha."),
        InstructionExample(instruction="B?", response="beta."),
    ] * 10

    token_ids, loss_mask = build_masked_token_ids(examples, tokenizer)
    # context_length must comfortably exceed any single example's prompt
    # length (here ~21 tokens) -- otherwise a sliding window can land
    # entirely inside a prompt, with zero unmasked targets, which makes
    # F.cross_entropy return nan for that batch (see
    # docs/finetune-loss-masking.md, "Known limitation"). Real configs use
    # context_length=256 against short instructions, so this doesn't
    # happen in practice; this test picks a context_length generous enough
    # relative to the tiny test corpus to avoid it here too.
    model_config = _tiny_model_config(
        vocab_size=tokenizer.vocab_size, context_length=64, n_embd=32
    )
    model = GPT(model_config)

    tokens = np.array(token_ids)
    mask = np.array(loss_mask)
    dataset = TokenDataset(tokens, context_length=model_config.context_length, loss_mask=mask)

    before = evaluate_model(model, dataset, batch_size=4)

    finetune_config = TrainConfig(
        batch_size=4,
        learning_rate=5e-3,
        min_lr=5e-4,
        warmup_steps=5,
        max_steps=100,
        grad_clip=1.0,
        eval_every=100,
        checkpoint_dir=str(tmp_path / "finetuned_masked"),
        checkpoint_every=100,
        weight_decay=0.01,
    )
    train_model(model, dataset, dataset, finetune_config, log_fn=lambda _msg: None)

    after = evaluate_model(model, dataset, batch_size=4)

    assert after["loss"] < before["loss"]
    assert after["perplexity"] < before["perplexity"]
