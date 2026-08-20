# llm-from-scratch

A small GPT-style language model, built stage by stage to understand every
part of it — tokenizer, data pipeline, architecture, pretraining,
evaluation, and instruction fine-tuning.

Start here: [`docs/00-roadmap.md`](docs/00-roadmap.md)

Working in this repo with Claude Code? Read [`CLAUDE.md`](CLAUDE.md) first —
it defines how work is done here (docs before code, stage order, style).

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
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once upon a time"
```
