# Weight decay parameter grouping (cross-cutting milestone)

Not a new stage — this extends Stage 4's optimizer construction
(docs/04-pretraining.md, "One training step") the same way LR decay
extended `get_lr` without becoming a new numbered stage.

## What this is

`train_model` used to build the optimizer with a single call,
`torch.optim.AdamW(model.parameters(), lr=...)` — no `weight_decay`
argument, so every parameter got AdamW's library default,
`weight_decay=0.01`, applied uniformly. This milestone splits parameters
into two optimizer groups: **weight matrices** (`nn.Linear`/`nn.Embedding`
weights) get real weight decay; **biases and LayerNorm gains** get none.

## Why it matters

Weight decay pulls every parameter it applies to a little closer to zero
each step, independent of the gradient — a soft regularizer against
overfitting. For a weight matrix, "closer to zero" is a meaningful
regularization: it discourages any single connection from growing large
and memorizing training-specific patterns. For a bias or a LayerNorm gain,
it's not — those parameters exist to represent a shift or a scale the
model has genuinely learned it needs (e.g. LayerNorm's gain defaults to 1,
meaning "don't rescale"; decaying it toward 0 pulls it away from that
default for no regularization benefit). Decaying them doesn't fight
overfitting, it just adds noise the model has to correct for. This is
standard practice in GPT-style training recipes (see nanoGPT's
`configure_optimizers`), not a novel choice.

## The one key idea to hold onto

Not every number in the model is the same *kind* of number. Weight
matrices encode learned connections between features — regularizing them
makes sense. Biases and LayerNorm gains encode a shift/scale the model
needs to represent its data correctly — regularizing them doesn't. Split
by shape: anything 2-D or higher (a matrix) gets weight decay; anything
1-D (a vector — every bias and every LayerNorm weight in this
architecture) doesn't.

## Design

New `configure_optimizer(model, learning_rate, weight_decay) ->
torch.optim.AdamW` in `src/llm_from_scratch/train/loop.py`:

```python
decay, no_decay = [], []
for param in model.parameters():
    (decay if param.dim() >= 2 else no_decay).append(param)
return torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=learning_rate,
)
```

`param.dim() >= 2` is the entire rule — no special-casing by module type
or parameter name. It happens to sort cleanly along architectural lines in
this model: `token_embedding.weight`, `position_embedding.weight`, every
`nn.Linear` weight (`qkv_proj`, `out_proj`, the two `FeedForward` layers)
are all 2-D and land in `decay`; every `nn.Linear` bias and every
`nn.LayerNorm` weight/bias (`ln1`, `ln2`, `ln_final`) are 1-D and land in
`no_decay` — but the rule itself is purely about shape, which is what
keeps it simple and keeps it correct automatically if the architecture
ever changes (a new 1-D parameter is automatically excluded; a new matrix
is automatically included).

`TrainConfig` gains one new required field, `weight_decay`, next to the
existing `learning_rate`/`min_lr` — no default, matching how every other
`TrainConfig` field already works. `train_model` calls
`configure_optimizer(model, config.learning_rate, config.weight_decay)`
in place of the old direct `AdamW(...)` call; nothing else about the
training loop (LR schedule, gradient clipping, checkpointing, resume)
changes. `configure_optimizer` is exported from `llm_from_scratch.train`
so it can be tested and used directly, the same way `get_lr` is.

## Interaction with `--resume`

No interaction needed. `configure_optimizer` builds the same two
parameter groups every time it's called with the same model and config,
in the same parameter order (`model.parameters()` iterates in a fixed,
deterministic order for a given architecture) — so
`optimizer.load_state_dict(...)` on a resumed run (docs/checkpoint-resume.md)
lines back up correctly with the groups it was saved from, exactly as it
already did before this milestone's group split.

## Magnitude: unchanged

This milestone is about *where* weight decay applies, not *how much*.
`configs/small.yaml` and `configs/finetune.yaml` both set
`weight_decay: 0.01` — the same value every parameter used to get
unconditionally under AdamW's old default — so a run's *decay group*
behaves exactly as before; only the *no-decay group* (biases, LayerNorm
gains) actually changes behavior, from `weight_decay=0.01` to `0.0`.

## What this milestone does *not* do

No new CLI flag, no `weight_decay: 0.0` opt-out switch — one grouping
scheme, applied unconditionally, the same reasoning as the LR decay
milestone's single schedule. No change to gradient clipping, the LR
schedule, or anything about *which* optimizer is used (still `AdamW`).

## Status: implemented and tested

`configure_optimizer` (`src/llm_from_scratch/train/loop.py`, exported from
`llm_from_scratch.train`) builds two `AdamW` parameter groups by
`param.dim()`, as described above. `TrainConfig` gained a required
`weight_decay` field; `configs/small.yaml` and `configs/finetune.yaml`
both set `weight_decay: 0.01`. `train_model` uses `configure_optimizer` in
place of the old direct `AdamW(model.parameters(), lr=...)` call — the
only change to the training loop itself.

At `configs/small.yaml`'s real shape (vocab_size 8001, context_length 256,
n_layer 6, n_head 6, n_embd 384) the split was checked directly:
13,818,240 total parameters, 13,787,520 in the decay group (26 weight
tensors) and 30,720 in the no-decay group (50 bias/LayerNorm tensors) —
every parameter accounted for exactly once, no overlap.

Tests added in `tests/test_train.py`:
`test_configure_optimizer_groups_params_by_ndim` (two param groups with
the expected `weight_decay` values, every parameter split correctly by
`ndim`, and named-parameter spot checks confirming `*.bias` and
`*norm*.weight` land in no-decay while `token_embedding.weight` lands in
decay) and `test_weight_decay_shrinks_weight_matrices_but_not_biases` (a
direct behavioral check: with all gradients forced to zero and several
`optimizer.step()` calls, decay-group parameters move — AdamW's decoupled
weight decay shrinks them independent of gradient — while no-decay-group
parameters stay bit-identical to their starting values). All other
`TrainConfig(...)` construction sites across `tests/test_train.py`,
`tests/test_eval.py`, `tests/test_finetune.py`, `tests/test_seed.py`,
`tests/test_train_resume_cache.py`, and `tests/test_train_resume_tokenizer.py`
(both Python-level factories and embedded-YAML fixtures) updated to supply
`weight_decay`. Full suite: `148 passed` (previous 146 + 2 new).

## Related: `max_steps` raised in `configs/small.yaml`

Alongside this milestone, `configs/small.yaml`'s `max_steps` was raised
from `5000` to `20000`. This is unrelated to weight decay grouping itself
but was identified in the same training-readiness review: at
`batch_size=32` and `context_length=256` (≈8,192 tokens/step), 5000 steps
sees only ~41M tokens — too little to make meaningful use of a real
tens-of-MB training corpus, or to let cosine LR decay (docs/lr-decay.md)
do meaningful work over the run. No other `configs/small.yaml` value
changed.
