# 04 — Pretraining Loop

## What it is
The actual training process: feed batches through the model, compare
predicted next-token distribution to the real next token (cross-entropy
loss), backpropagate, update weights, repeat — for many steps.

## Why it matters
This is where "architecture" becomes "a model that knows things." A
correct architecture with a broken training loop (bad learning rate,
no gradient clipping, wrong loss) produces nothing useful. This stage is
mostly about getting the mechanics right and watching loss actually fall.

## The one key idea
The loss is just: "how surprised was the model by the real next token?"
Lower surprise, on average, across the whole corpus = better model. Every
training trick (learning rate warmup, gradient clipping, weight decay)
exists to make that number go down smoothly instead of exploding or
stalling.

## What training actually does

Stage 3 gave us a model that turns token IDs into logits — a score per
vocabulary word — but with random, untrained weights, those scores are
meaningless. Training is the process of nudging every weight in the model,
over and over, so that the logits it produces get closer to "assign a high
score to the token that actually comes next." Nothing else changes about
the model's shape or code — only the numbers inside its weight matrices.

## From logits to a loss number: cross-entropy

The model's output for one position is `vocab_size` raw scores (logits).
Turn them into a probability distribution with **softmax** (exponentiate,
divide by the sum — the classic feed-forward way to turn arbitrary numbers
into "probabilities that sum to 1"). **Cross-entropy loss** then looks up
the probability the model assigned to the *actual* next token and takes
`-log(that probability)`.

- If the model gave the correct token high probability (close to 1),
  `-log(prob)` is close to 0 — low loss, low "surprise."
- If the model gave the correct token low probability (close to 0),
  `-log(prob)` is large — high loss, high "surprise."

`torch.nn.functional.cross_entropy` does the softmax and the `-log` and
the lookup in one call, given raw logits and the target token ID — which
is exactly what `GPT.forward(idx, targets)` already does (Stage 3).

## Targets: the labels are just the input, shifted

Stage 2's `TokenDataset` already produces `(input_ids, target_ids)` pairs
where `target_ids` is `input_ids` shifted one position right. So "the
correct next token for position `i`" is just `target_ids[i]` — no separate
labeling step. This is what "language modeling only needs one label: the
next token" (Stage 2's key idea) cashes out to at training time.

## One training step: forward → loss → backward → optimizer step

For one batch:

1. **Forward pass** — run `logits, loss = model(input_ids, target_ids)`.
   The model computes its guesses and the loss in one call.
2. **Backward pass** — `loss.backward()`. PyTorch's autograd walks
   backward through every operation that produced `loss`, computing how
   much each weight contributed to the error (the weight's *gradient*: the
   direction and amount that would make loss worse if you moved the
   weight that way).
3. **Optimizer step** — `optimizer.step()` nudges every weight a small
   amount in the *opposite* direction of its gradient (downhill on the
   loss). The **optimizer** (we use `AdamW`, the standard choice for
   transformers) also tracks per-weight momentum so it doesn't overreact
   to noisy single-batch gradients.
4. **Zero gradients** — `optimizer.zero_grad()` before the next step,
   since PyTorch accumulates gradients by default (adds new ones to old
   ones) unless told to clear them.

Repeat for many batches. That's the entire loop — everything else in this
stage is refinements to make it stable and observable.

## Steps, not epochs

An **epoch** is one full pass over the training data. This project counts
**steps** instead (one step = one batch), via `max_steps` in
`configs/small.yaml` — simpler to reason about when random windows are
sampled from a memmapped corpus rather than iterated in fixed epochs.

## Learning rate and warmup

The **learning rate** controls how big each optimizer step is. Too high
and the loss oscillates or diverges (it overshoots the "downhill"
direction); too low and training crawls. **Warmup** (`warmup_steps` in
the config) starts the learning rate near 0 and ramps it up over the
first N steps — early in training, weights are random and gradients can
be large/noisy, so a full-size step could be destructive before the model
has "settled."

## Gradient clipping

Occasionally a batch produces an unusually large gradient (e.g. a
surprising sequence). Without a cap, that one step could throw weights
far off course. `grad_clip` (in the config) rescales gradients so their
overall norm never exceeds a fixed value, before the optimizer step —
cheap insurance against rare bad batches.

## Validation loss: are we learning, or memorizing?

Every `eval_every` steps, run the model (no backward pass, no weight
update — just forward + loss) on batches from the **held-out validation
split** (Stage 2's `train_val_split`). If training loss keeps falling but
validation loss stalls or rises, the model is memorizing training data
rather than learning general patterns — a signal to stop, get more data,
or add regularization.

## Worked example: one training step by hand

Config: `context_length=4`, `batch_size=1` (real config uses bigger
numbers; shrunk here to trace by hand). Token stream: `[7, 2, 9, 4, 1]`.

```
input_ids  = [[7, 2, 9, 4]]      # tokens 0-3
target_ids = [[2, 9, 4, 1]]      # same tokens, shifted by one
```

1. Forward: model outputs logits of shape `(1, 4, vocab_size)` — one
   score-per-vocab-word for each of the 4 positions.
2. Loss: at position 0, the model saw `[7]` and should have predicted
   `2`; cross-entropy checks how much probability mass it put on token
   `2`. Same idea at positions 1, 2, 3. `cross_entropy` averages the
   "surprise" across all 4 positions into one number.
3. Backward: `loss.backward()` computes a gradient for every weight —
   "if I increase this weight slightly, does loss go up or down, and by
   how much?"
4. Step: `optimizer.step()` moves every weight slightly in the
   loss-reducing direction. `optimizer.zero_grad()` clears gradients for
   the next batch.

Do this thousands of times (`max_steps`), with different random windows
each time, and the loss number — on average — goes down.

## Simplification note

Production training adds distributed training (many GPUs), mixed-
precision (fp16/bf16) for speed, more sophisticated LR schedules (cosine
decay after warmup, not just warmup), and resuming from checkpoints
mid-run. We implement a single-device loop with linear warmup and a flat
learning rate after, and start-of-run-only checkpointing at the end —
enough to watch loss fall and produce a usable checkpoint, not tuned for
speed.

## What we build here
A plain PyTorch training loop with checkpointing, logging (loss/step,
tokens/sec), gradient clipping, and a simple LR schedule — no external
training framework, so every step is visible.

## Status: implemented and tested

`src/llm_from_scratch/train/loop.py` implements `TrainConfig`,
`load_config` (reads `configs/*.yaml` into `GPTConfig`/`TrainConfig`),
`get_lr` (linear warmup then flat), `estimate_loss` (forward-only, for
validation), and `train_model` (the loop itself: forward → loss →
backward → clip → optimizer step, per `max_steps`, logging and
checkpointing to `checkpoint_dir`). `scripts/train.py` wires it to the
tokenizer and data pipeline end to end: reads `.txt` files from
`data/raw_path`, trains a `BPETokenizer` on them, writes token ids via
`write_token_ids`, splits train/val, builds `TokenDataset`s, builds a
`GPT`, and calls `train_model`.

Tests in `tests/test_train.py` cover the LR schedule, that
`estimate_loss` never updates weights, that a training step does update
weights, that a full run produces a loadable checkpoint, and — as an
end-to-end smoke test — that loss clearly decreases when a tiny model
overfits a short repeating token pattern. A full manual run of
`scripts/train.py` on a small hand-written corpus was also verified
directly: train_loss fell from ~5.0 to ~3.2 over 30 steps.

One integration issue surfaced and fixed: PyYAML's default resolver only
recognizes bare-exponent numbers like `3e-4` as floats when they have a
decimal point (`3.0e-4`) — without one, `3e-4` parses as a string, which
silently broke `AdamW`'s learning-rate check. `configs/small.yaml` uses
the bare-exponent form, so `load_config` now coerces `learning_rate` to
`float` explicitly (see `tests/test_train.py::test_load_config_coerces_bare_exponent_learning_rate`).

Update (tokenizer persistence milestone): `train_model` now takes an
optional `tokenizer` argument and, when given one, saves it as
`tokenizer.json` next to `latest.pt` in `checkpoint_dir` — see
docs/01-tokenization.md, "Tokenizer persistence". `scripts/train.py` passes
its trained tokenizer through. This makes a checkpoint directory a
self-contained unit (weights + config + the exact tokenizer used to
produce its training data), which Stages 5–7 now rely on instead of
retraining a tokenizer from the raw corpus each time.
