import json

import numpy as np
import torch

from llm_from_scratch.data import TokenDataset
from llm_from_scratch.eval import evaluate_model
from llm_from_scratch.finetune import (
    InstructionExample,
    build_corpus,
    format_example,
    load_examples_jsonl,
    load_pretrained_model,
)
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
    torch.save({"model_state_dict": model.state_dict(), "model_config": config}, checkpoint_path)

    loaded = load_pretrained_model(checkpoint_path)

    assert loaded.config == config
    for name, param in model.named_parameters():
        assert torch.equal(param, loaded.state_dict()[name])


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
        warmup_steps=5,
        max_steps=100,
        grad_clip=1.0,
        eval_every=100,
        checkpoint_dir=str(tmp_path / "finetuned"),
    )
    result = train_model(model, dataset, dataset, finetune_config, log_fn=lambda _msg: None)

    after = evaluate_model(model, dataset, batch_size=4)

    assert after["loss"] < before["loss"]
    assert after["perplexity"] < before["perplexity"]

    checkpoint = torch.load(result["checkpoint_path"], weights_only=False)
    assert checkpoint["model_config"] == model_config
