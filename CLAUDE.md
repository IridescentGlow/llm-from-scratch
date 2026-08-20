# CLAUDE.md — Project Operating Instructions

This file is the standing brief for every Claude Code session in this repo.
Read it fully before doing anything else.

## What this project is

A small, from-scratch LLM, built stage by stage so the person building it
*understands every part* — not a black-box wrapper around a pretrained model.
Style reference: Sebastian Raschka's "Build a Large Language Model (From
Scratch)" — fast, plain-English concept explanations, then minimal clean code.

Default scale assumption (change in `configs/small.yaml` if this is wrong):
- PyTorch, decoder-only transformer (GPT-style)
- Small enough to train on a single consumer GPU, or CPU for smoke tests
- BPE tokenizer (not char-level) — closer to real-world practice, still simple
- Trained on a modest public text corpus (a few hundred MB), not internet-scale

## The one rule that matters most: docs before code

For every stage in `docs/`, the doc must be written or updated to explain:
1. **What this stage is** — in plain English, 3–6 sentences max
2. **Why it matters** — what breaks or is missing without it
3. **The one key idea to hold onto** — the thing that, once understood, makes
   the code obvious

Only after that doc reads clearly should implementation code for that stage
be written. If asked to "just build X," still produce/update the doc first,
briefly note that you did, then implement. No exceptions, no skipping ahead.

Keep docs short. A wall of text defeats the purpose. If a doc runs past
roughly one screen, cut it down, not up.

## Stage order (do not reorder without discussion)

1. Tokenization — `docs/01-tokenization.md` → `src/llm_from_scratch/tokenizer/`
2. Data pipeline — `docs/02-data-pipeline.md` → `src/llm_from_scratch/data/`
3. Architecture — `docs/03-architecture.md` → `src/llm_from_scratch/model/`
4. Pretraining loop — `docs/04-pretraining.md` → `src/llm_from_scratch/train/`
5. Evaluation — `docs/05-evaluation.md` → `src/llm_from_scratch/eval/`
6. Fine-tuning / instruction-tuning — `docs/06-finetuning.md` → `src/llm_from_scratch/finetune/`

Each stage should be runnable and testable before moving to the next.
Don't build stage N+1 on top of an unverified stage N.

## Explanation style (applies to chat responses, not just docs)

- Concept before code, every time.
- Short sentences. No padding, no "as we can see," no filler transitions.
- Prefer one small worked example over abstract description.
- If a term appears that hasn't been explained yet (e.g. "attention mask"),
  explain it inline in one sentence before using it.
- When something is a simplification of how production LLMs really do it,
  say so in one line ("real GPT-3 does X differently because Y — we're
  skipping that for now").

## Coding conventions

- Python 3.11+, type hints on all function signatures.
- PyTorch only — no training-framework abstractions (no Lightning/HF Trainer)
  until the fundamentals are solid. The point is to see the loop.
- No premature abstraction. Duplicate a little code across stages rather
  than build a shared framework too early.
- Every module in `src/llm_from_scratch/` gets a matching test in `tests/`.
- Config values (model size, learning rate, batch size, etc.) live in
  `configs/*.yaml` — never hardcoded in scripts.
- Scripts in `scripts/` are thin entry points that call into `src/`, not
  where logic lives.

## Current stage tracker

Update this section as you go — this is the single source of truth for
"where are we."

```
Stage:     4 - Pretraining loop
Status:    implemented and tested
Last run:  2026-08-20 — tests/ (full suite), 31 passed; manual scripts/train.py
           smoke run on a tiny hand-written corpus, train_loss 5.0 -> 3.2 over 30 steps.
Notes:     Stage 1 (Tokenization): implemented and tested.
           BPETokenizer (byte-level BPE) in
           src/llm_from_scratch/tokenizer/bpe.py, exported from
           tokenizer/__init__.py. Surface: .train(corpus, vocab_size),
           .encode(text), .decode(ids). No regex pre-tokenization yet
           (see simplification note in docs/01-tokenization.md).

           Stage 2 (Data pipeline): implemented and tested.
           src/llm_from_scratch/data/tokens.py — write_token_ids,
           load_token_ids (numpy.memmap-backed), train_val_split.
           src/llm_from_scratch/data/dataset.py — TokenDataset (windowing,
           input/target shift-by-one) + get_dataloader (batching, random
           sampling via shuffle=True). Token ids stored as uint16 on disk
           (fits vocab sizes up to 65,536).

           Stage 3 (Architecture): implemented and tested.
           src/llm_from_scratch/model/gpt.py — GPTConfig, CausalSelfAttention,
           FeedForward, Block, GPT. Surface: GPT(config).forward(idx,
           targets=None) -> (logits, loss); GPT.generate(idx, max_new_tokens)
           (greedy argmax sampling). Pre-norm transformer blocks, learned
           absolute positional embeddings, hand-written multi-head causal
           attention (no nn.TransformerEncoder). No rotary embeddings /
           fused attention kernels yet (see simplification note in
           docs/03-architecture.md).

           Stage 4 (Pretraining loop): implemented and tested.
           src/llm_from_scratch/train/loop.py — TrainConfig, load_config,
           get_lr (linear warmup then flat), estimate_loss (forward-only),
           train_model (forward -> loss -> backward -> grad clip ->
           optimizer step, per max_steps; logs + checkpoints). scripts/train.py
           wires tokenizer + data pipeline + model + train_model end to end
           (trains a tokenizer on data/raw/*.txt at run time; no tokenizer
           persistence yet). See decisions log: PyYAML bare-exponent float bug.
           Next: Stage 5 - Evaluation (not started, awaiting go-ahead).
```

## Decisions log

Append short entries here when a non-obvious choice is made, so future
sessions don't relitigate it.

```
- PyYAML's default resolver only treats bare-exponent numbers (e.g. `3e-4`)
  as floats if they include a decimal point (`3.0e-4`); without one, it
  silently parses them as strings. configs/small.yaml uses the bare form
  for learning_rate, which broke AdamW's lr check. Fixed by coercing
  learning_rate to float explicitly in train.loop.load_config, rather than
  rewriting every config file to use the decimal-point form.
```

## Running things

```
pip install -e .
python scripts/train.py --config configs/small.yaml
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "..."
python scripts/evaluate.py --checkpoint checkpoints/latest.pt
```

## What "done" looks like for this project

A person with no prior LLM background can read `docs/00-roadmap.md` through
`docs/06-finetuning.md` in order, understand what an LLM is and how it's
built at every layer, and separately run the code and watch it actually
train and generate text. Both halves — understanding and working code —
are required. Neither alone is a finished stage.
