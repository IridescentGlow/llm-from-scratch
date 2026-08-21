# 05 — Evaluation

## What it is
Measuring whether the trained model is actually good — beyond watching
training loss decrease, which only tells you it fits the training data.

## Why it matters
A model can have great training loss and still be useless: overfit,
repetitive, incoherent past a few tokens, or simply memorizing. Evaluation
is what tells you whether to move to fine-tuning or go back and fix
something upstream.

## The one key idea
Always evaluate on held-out text the model never trained on. The gap
between train loss and validation loss tells you if you're overfitting;
qualitative generation samples tell you if it's coherent at all — numbers
alone can look fine while the output reads like nonsense.

## Why training loss alone isn't enough

Stage 4's training loop drives down *training* loss — the average
"surprise" (cross-entropy) on batches the model is actively learning from.
But a model can get very good at training loss by essentially memorizing
the training text, including its quirks and noise, rather than learning
general patterns of language. That's **overfitting**. A model that has
memorized its training data will look great on the numbers you watched
during training and still fail on anything new.

## Train loss vs. validation loss

Stage 2 already set aside a **validation split** — a slice of the corpus
the model never sees during training (`train_val_split` in
`configs/small.yaml`). Evaluation means running the exact same
forward-pass-plus-cross-entropy computation from Stage 4, but on this
held-out data, with no backward pass and no weight update.

- **Train loss ≈ validation loss, both low** → the model generalized well.
- **Train loss low, validation loss much higher** → overfitting: the model
  learned the training text specifically, not language in general.
- **Both losses high** → the model hasn't learned much yet (undertrained,
  too small, or too little data).

## Perplexity: the same number, easier to reason about

Cross-entropy loss is a number like `2.3` — technically meaningful, but
not intuitive. **Perplexity** is just `exp(loss)` (`e` raised to the loss
value), and it has a plain-English reading: *"the model was, on average,
as confused as if it were guessing uniformly among this many tokens."*

- Loss `0` → perplexity `1` → the model was completely certain, every time
  (in practice this only happens by memorizing, or on a trivial/tiny
  vocabulary).
- Loss `ln(8000) ≈ 8.99` → perplexity `8000` → the model is no better than
  guessing uniformly at random among all 8000 vocabulary tokens (this
  project's vocab size) — i.e., it's learned nothing.
- Real, reasonably-trained small models land somewhere in between —
  perplexity in the tens to low hundreds is a believable range for a
  small GPT on a small corpus after meaningful training; lower is better,
  and it should keep dropping as training improves.

Perplexity doesn't add new information over the loss — it's the same
number, reshaped to be readable as "effective number of tokens the model
is confused between," instead of an abstract log-probability.

## How we evaluate a trained checkpoint

1. Load the checkpoint (Stage 4 saves `model_state_dict` + `model_config`
   via `torch.save`) and rebuild the `GPT` model from it.
2. Rebuild the validation `TokenDataset` the same way Stage 4's training
   script does (same corpus, same tokenizer, same `train_val_split`) —
   the point is to reuse Stage 2's data pipeline unchanged, not build a
   separate evaluation-only data path.
3. Put the model in `eval()` mode (disables dropout — evaluation should
   measure the model's real behavior, not a randomly-perturbed training
   version of it) and run forward passes only, no `loss.backward()`, no
   optimizer step — exactly the read-only half of Stage 4's training step.
4. Average the loss across all validation batches, then report both the
   average loss and its perplexity (`exp(loss)`), plus how many batches
   and tokens that average was computed over — a loss from 2 batches
   means much less than one from 200.

## What a useful evaluation result looks like

Not a single number in isolation, but a comparison: validation loss
(and perplexity) alongside the training loss from the same checkpoint.
Falling validation loss across checkpoints, tracking close to training
loss, is the useful signal — the same single number from one run tells
you much less.

## Worked example

Suppose after training, the average cross-entropy loss on 40 validation
batches is `4.2`:

```
val_loss   = 4.2
perplexity = exp(4.2) ≈ 66.7
```

Reading it: "on average, the model was as uncertain about the next token
as if it were guessing uniformly among about 67 options" — much better
than guessing among all 8000 vocabulary tokens (perplexity 8000 if
untrained), but still clearly room to improve (a well-trained small model
on a small, simple corpus could get well under that). If training loss on
the same checkpoint were `1.1` (perplexity ≈ 3), the large gap between
`1.1` and `4.2` would say: this model has overfit the training data.

## Simplification note

Production evaluation suites run many held-out benchmarks (multiple-
choice QA, coding tasks, human preference comparisons) beyond a single
validation-loss number, and often evaluate the same checkpoint across
several curated datasets to catch narrow failure modes. We compute one
number (validation loss / perplexity, on this project's own held-out
split) — sufficient to catch overfitting and undertraining, not to
compare against other models or claim general capability.

## What we build here
A validation-loss/perplexity check on held-out data, computed with the
existing model, data pipeline, and checkpoint format from Stages 2–4 —
no new data path, no new model code.

## Status: implemented and tested

`src/llm_from_scratch/eval/metrics.py` implements `evaluate_model(model,
dataset, batch_size, device="cpu", max_batches=None)`: forward-only,
deterministic (no shuffling), model temporarily set to `eval()` mode and
restored afterward. Returns `{"loss", "perplexity", "num_batches",
"num_tokens"}` — `perplexity = exp(loss)`, and the batch/token counts make
clear how much data the number is based on. Reuses Stage 2's
`get_dataloader` and Stage 3's `GPT` directly — no new data or model code.
`scripts/evaluate.py` loads a checkpoint's `model_state_dict` +
`model_config`, rebuilds the validation split, and prints the four
numbers.

Tests in `tests/test_eval.py` cover: perplexity is exactly `exp(loss)`,
batch/token counts match the dataset size, `max_batches` truncates
correctly, weights are never updated, repeated calls are deterministic,
the model's original train/eval mode is restored, and — as an end-to-end
check — evaluating the same model before and after a tiny overfit run
shows both val loss and perplexity improve.

A full manual run was also verified: trained a tiny checkpoint via
`scripts/train.py` on a hand-written corpus (train_loss 5.2 → 3.1 over 30
steps), then ran `scripts/evaluate.py` against it — reported
`val_loss=3.83`, `perplexity=46.06` over 4 batches / 104 tokens, matching
the training run's own last validation check exactly.

One limitation, not an integration bug: Stage 1's tokenizer has no
save/load yet, so `scripts/evaluate.py` retrains a tokenizer on the same
corpus + `vocab_size` to deterministically reproduce the same token
stream and split used at training time. This only holds if `data/raw/`
hasn't changed since the checkpoint was trained — tokenizer persistence
is a natural improvement for a later pass, not added here to stay in
scope for this stage.
