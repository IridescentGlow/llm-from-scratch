# Learning rate decay (cross-cutting milestone)

Not a new stage — this extends Stage 4's `get_lr` (docs/04-pretraining.md,
"Learning rate and warmup") the same way seeding and checkpoint/resume
extended earlier stages without becoming a new numbered stage.

## What this is

Today `get_lr` does linear warmup, then holds the learning rate flat at
`base_lr` for the rest of training, all the way to the final step. This
milestone replaces "flat after warmup" with **cosine decay**: after warmup
ends, the learning rate follows a smooth downward curve from `base_lr` to a
small floor (`min_lr`), reaching that floor right as training reaches
`max_steps`.

## Why it matters

Early in training, weights are far from any good solution, so large steps
make fast progress — that's what the flat, warmed-up `base_lr` is good at.
Late in training, weights are close to a good solution, and that same
large step size starts to do more harm than good: it overshoots small
improvements and makes the loss curve noisy instead of settling. A model
trained with a flat learning rate the whole way through typically ends up
at a measurably worse loss than one that eases off the step size as it
approaches convergence — for a fixed compute budget (fixed `max_steps`),
decaying the learning rate is one of the cheapest wins available, since it
costs no extra forward/backward passes, only a different number plugged
into the same optimizer step.

## The one key idea to hold onto

Warmup and decay are the same underlying idea applied at opposite ends of
training: match the step size to how much the model still has to learn.
Early on, everything is wrong, so warmup ramps *up* to a productive step
size. Late on, the model is close to done, so decay ramps *down* to a
careful one. The learning rate over a full run looks like a ramp up
followed by a long, smooth ramp down — not a cliff at either end.

## Why cosine, specifically

A **cosine curve** goes from 1 to 0 (or in this case, from `base_lr` to
`min_lr`) slowly at first, steeply through the middle, then slowly again
near the end — unlike a straight linear ramp-down, which decreases at a
constant rate the whole way. That shape matches training dynamics well in
practice: it keeps the learning rate close to `base_lr` a bit longer after
warmup (still making fast progress while gains are cheap), then eases
through the middle, then spends the final stretch making small, careful
adjustments instead of dropping off a cliff right before the run ends. This
is the standard choice in most modern LLM training recipes (GPT-2/3-style),
which is why it's the one implemented here rather than a linear or step
decay.

## Why a floor (`min_lr`) instead of decaying to zero

Decaying all the way to exactly 0 means the last stretch of training does
effectively nothing — steps that size don't move the weights in any
meaningful way, wasting `max_steps` budget on no-op updates. A small
nonzero floor (conventionally around 10% of `base_lr`) keeps the model
still learning, just carefully, right up to the last step.

## Design

`get_lr` gains two new required inputs — `max_steps` (already known to the
caller; it's how long the whole run is) and `min_lr` (the floor) — and its
after-warmup behavior changes from "return `base_lr`" to a cosine curve
from `base_lr` down to `min_lr` over the steps from `warmup_steps` to
`max_steps`:

```
if step < warmup_steps:              # unchanged: linear warmup
    return base_lr * (step + 1) / warmup_steps

decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
coeff = 0.5 * (1 + cos(pi * decay_ratio))   # 1 -> 0 over the decay window
return min_lr + coeff * (base_lr - min_lr)
```

No new CLI flag, no `lr_decay: cosine | none` config switch: the project
already has exactly one learning rate schedule at a time, and cosine decay
is a strict improvement over flat with no real downside, so it simply
replaces the old "flat after warmup" behavior everywhere `get_lr` is used
— pretraining and fine-tuning alike. `TrainConfig` gains one new required
field, `min_lr`, next to the existing `learning_rate`; both
`configs/small.yaml` and `configs/finetune.yaml` need it added explicitly,
matching how every other `TrainConfig` field already works (no defaults —
every config file states every value).

## Interaction with `--resume`

`get_lr` is a pure function of `step` and the config values — it holds no
state of its own between calls. A resumed run (docs/checkpoint-resume.md)
picks up at `start_step` and calls `get_lr(step, ...)` with the same
`warmup_steps`/`max_steps`/`min_lr` values as the original run, so the
learning rate curve continues exactly where it left off, with no special
handling needed. This is the same reason resume already works correctly
for the existing warmup logic.

## Worked example

`base_lr=3e-4`, `min_lr=3e-5` (10% of `base_lr`), `warmup_steps=100`,
`max_steps=5000` (`configs/small.yaml`'s shape):

- Steps 0-99: linear warmup, `3e-4 * (step+1)/100`, ending at `3e-4`.
- Step 100 (decay_ratio=0): `cos(0)=1` -> full `base_lr`, `3e-4`.
- Step 2550 (decay_ratio=0.5, roughly the midpoint of the decay window):
  `cos(pi/2)=0` -> halfway between `base_lr` and `min_lr`, `1.65e-4`.
- Step 5000 (decay_ratio=1): `cos(pi)=-1` -> `min_lr`, `3e-5`.

Compare to today: steps 100-5000 all return the same flat `3e-4`.

## What this milestone does *not* do

No warm restarts / cyclical schedules (a single warmup-then-decay curve
per run is enough for this project's scale). No per-parameter-group
learning rates. No change to warmup itself, gradient clipping, or the
optimizer (`AdamW`) — only what `get_lr` returns after warmup changes.

## Status: implemented and tested

`get_lr` (`src/llm_from_scratch/train/loop.py`) now takes
`(step, warmup_steps, max_steps, base_lr, min_lr)` and implements exactly
the design above: linear warmup, then cosine decay from `base_lr` to
`min_lr` over `[warmup_steps, max_steps]`, holding at `min_lr` for any step
`>= max_steps`. `TrainConfig` gained a required `min_lr` field, right next
to `learning_rate`; `load_config` coerces it to `float` the same way it
already does for `learning_rate` (the same PyYAML bare-exponent issue
applies equally to `min_lr`), and `scripts/finetune.py`'s own YAML loading
(it doesn't go through `load_config`) got the matching coercion line.
`configs/small.yaml` sets `min_lr: 3e-5` (10% of its `learning_rate:
3e-4`); `configs/finetune.yaml` sets `min_lr: 3e-6` (10% of its
`learning_rate: 3e-5`) — the conventional ~10%-of-peak floor mentioned
above. No new CLI flag or config switch, exactly as designed: cosine decay
replaced flat-after-warmup everywhere `get_lr` is called, for pretraining
and fine-tuning alike.

Tests updated/added in `tests/test_train.py`: the old
`test_get_lr_warmup_then_flat` became `test_get_lr_linear_warmup` (warmup
portion unchanged) plus two new tests, `test_get_lr_cosine_decay_after_warmup`
(checks the start of the decay window, the midpoint, and `max_steps`
exactly against hand-computed values) and `test_get_lr_returns_min_lr_past_max_steps`;
`test_get_lr_handles_zero_warmup` updated for the new signature;
`test_load_config_coerces_bare_exponent_learning_rate` extended to also
assert `min_lr` coerces to `float` correctly. All other `TrainConfig(...)`
construction sites across `tests/test_train.py`, `tests/test_eval.py`,
`tests/test_finetune.py`, `tests/test_seed.py`, and
`tests/test_train_resume_tokenizer.py` (both Python-level and
embedded-YAML fixtures) updated to supply `min_lr`. Full suite: `118
passed` (previous 116 + 2 net new — one test renamed in place, two added).

Manual verification: computed `get_lr` directly against the worked example
above (`base_lr=3e-4, min_lr=3e-5, warmup_steps=100, max_steps=5000`) and
confirmed it matches exactly (step 0: `3e-6`; step 99: `3e-4`; step 100:
`3e-4`; step 2550: `1.65e-4`; step 5000 and beyond: `3e-5`). Ran a real
`scripts/train.py --device cpu --seed 1` smoke run (vocab_size 300,
context_length 16, n_layer 2, n_embd 32, `warmup_steps=5`, `max_steps=40`,
`learning_rate=1e-2`, `min_lr=1e-3`) on a synthetic corpus — logged `lr`
rose to `1.00e-02` at the end of warmup (step 5) then decayed smoothly
step over step down to `1.02e-03` at step 40, visibly curving (not a
straight line) as expected from cosine decay. Separately ran a 20-step
run, then `--resume` with `max_steps` raised to 40: the resumed run's
step-25 `lr` (`4.90e-03`) matched exactly what an uninterrupted 40-step
run logged at its own step 25, confirming the schedule continues correctly
across a resume rather than resetting or drifting — the same guarantee
`--resume` already gave the old flat-after-warmup schedule.
