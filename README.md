# llm-from-scratch

A small GPT-style language model, built stage by stage to understand every
part of it — tokenizer, data pipeline, architecture, pretraining,
evaluation, and instruction fine-tuning.

Trained from raw text to instruction-following, in seven stages:

1. **Tokenization** — byte-level BPE, text to token IDs.
2. **Data pipeline** — token IDs to batched, windowed training examples.
3. **Architecture** — a GPT-style decoder-only transformer, built from
   scratch (hand-written causal self-attention, no `nn.TransformerEncoder`).
4. **Pretraining loop** — forward, cross-entropy loss, backward, optimizer
   step, with LR warmup and gradient clipping.
5. **Evaluation** — validation loss and perplexity on held-out text.
6. **Fine-tuning** — instruction-tuning a pretrained checkpoint on
   (instruction, response) pairs.
7. **Generation / inference** — sampling text from a trained checkpoint,
   greedy or temperature-based.

Start here: [`docs/00-roadmap.md`](docs/00-roadmap.md) for the full arc, then
`docs/01-tokenization.md` onward for the "what and why" of each stage.

## Layout

```
docs/       concept docs, one per stage — read these to understand, not just run code
src/        the actual implementation, mirrors the stage order
configs/    yaml configs for model size / training hyperparameters
scripts/    thin CLI entry points (train / generate / evaluate)
tests/      one test module per src module
data/       raw/processed corpora (gitignored)
checkpoints/ saved model weights (gitignored)
notebooks/  scratch exploration, not source of truth
```

## Quickstart

```bash
pip install -e .
python scripts/train.py --config configs/small.yaml
python scripts/evaluate.py --checkpoint checkpoints/latest.pt
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once upon a time"
python scripts/finetune.py --checkpoint checkpoints/latest.pt
```

Training saves a checkpoint (`latest.pt`) alongside the exact tokenizer used
to produce its training data (`tokenizer.json`) — evaluation, fine-tuning,
and generation all load that tokenizer from the checkpoint directory rather
than retraining one, so token IDs always mean what the model learned them to
mean.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Technical details

- PyTorch, decoder-only transformer (GPT-style), pre-norm blocks, learned
  absolute positional embeddings.
- Byte-level BPE tokenizer, no external tokenizer library.
- Config-driven: model size, learning rate, batch size, etc. live in
  `configs/*.yaml`, not hardcoded in scripts.
- Small enough to train on a single consumer GPU, or run as a CPU smoke test.
