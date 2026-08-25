# Milestone — Weight Tying + GPT-Style Initialization

## What this stage is

Two related, small changes to `GPT` (`src/llm_from_scratch/model/gpt.py`),
neither touching the forward-pass logic or shapes described in
`docs/03-architecture.md`:

1. **Initialization**: replace PyTorch's default weight init with the
   scheme GPT-2 uses — every linear/embedding weight drawn from
   `N(0, 0.02²)`, with the two per-block projections that feed directly
   into the residual stream (`CausalSelfAttention.out_proj`,
   `FeedForward.net`'s second `Linear`) scaled down further by
   `1 / sqrt(2 * n_layer)`.
2. **Weight tying**: `lm_head` stops being its own learned weight matrix
   and instead reuses `token_embedding`'s weight matrix directly.

## Why it matters

Right now `GPT.__init__` never touches initialization at all — every
`nn.Linear` and `nn.Embedding` keeps PyTorch's library default. That
default wasn't chosen for transformers; it's a general-purpose default
tuned for typical feedforward nets. Two concrete problems follow from
that:

- `nn.Embedding`'s default is `N(0, 1)` — a much larger initial scale
  than what stacked pre-norm transformer blocks expect. Combined with no
  down-scaling on the projections that write into the residual stream,
  the variance of the residual stream tends to grow with depth from step
  one, before training has done anything to correct it. `configs/small.yaml`
  already uses `n_layer: 6`; this only gets worse as the project trains
  on a larger, more meaningful corpus with deeper/wider configs.
- The embedding table and the output projection are separate, independently
  learned weight matrices today, even though they do closely related jobs
  (mapping between token identity and a `n_embd`-dimensional space) and are
  the same shape. Tying them is a well-established practice (Press &
  Wolf, 2017; used in GPT-2) that removes redundant parameters and tends
  to improve the model's grasp of token identity, since the same vector
  now has to work for both "what does this token mean going in" and
  "which token does this vector predict coming out."

Neither problem is visible in the tiny smoke-test checkpoints this project
has trained so far (a few hundred characters of text, ~45K params, a
handful of layers) — but both are exactly the kind of thing that's cheap
to fix now and expensive to notice only after a real training run looks
worse than it should.

## The one key idea to hold onto

Initialization controls how big the model's numbers are *before any
training happens*; weight tying controls *how many independent numbers*
the model has in the first place. Fixing the first keeps depth trainable;
doing the second removes a redundant copy of the same information. Neither
changes what the model computes once trained — a well-trained tied,
correctly-initialized model and a well-trained untied, default-initialized
model could in principle reach similar quality, but the correctly-initialized
one gets there more reliably and touches fewer parameters getting there.

---

## Current state (read directly from the code)

`GPT.__init__` (`src/llm_from_scratch/model/gpt.py:104-112`):

```python
self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
self.dropout = nn.Dropout(config.dropout)
self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
self.ln_final = nn.LayerNorm(config.n_embd)
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

No call to `.apply(...)`, no custom `_init_weights`, no `torch.nn.init.*`
call anywhere in the file. Every weight is initialized by whatever
`nn.Linear`/`nn.Embedding`/`nn.LayerNorm` do by default:

- `nn.Embedding.weight` — `N(0, 1)` (`torch.nn.init.normal_`, default
  `mean=0, std=1`).
- `nn.Linear.weight` — Kaiming-uniform, `U(-1/√fan_in, 1/√fan_in)`.
- `nn.Linear.bias` (where present — every `nn.Linear` in this model except
  `lm_head`, which is `bias=False`) — `U(-1/√fan_in, 1/√fan_in)`, same
  bound as the weight.
- `nn.LayerNorm` — weight (gain) initialized to `1`, bias to `0`. This is
  already the standard choice and is **not changing** as part of this
  milestone.

`token_embedding` and `lm_head` are currently two entirely independent
`nn.Parameter` tensors, both shaped `(vocab_size, n_embd)` — same shape,
never linked.

Checkpoint save/load (`src/llm_from_scratch/checkpoint.py`,
`train/loop.py`'s `save_checkpoint`, `finetune/checkpoint.py`'s
`load_pretrained_model`) round-trip `model.state_dict()` verbatim: whatever
parameter names and tensors `GPT` currently has are exactly what gets
written to and read from `latest.pt`. Nothing in the checkpoint format
itself assumes tied or untied weights — it just stores whatever
`state_dict()` produces. This matters for the legacy-checkpoint policy
below.

Model tests (`tests/test_model.py`) construct `GPTConfig`/`GPT` freshly in
every test via a `_tiny_config()` helper and never assert anything about
specific initialization values or about `token_embedding`/`lm_head` being
distinct — so this change doesn't invalidate any existing assertion, only
adds new ones.

`configs/small.yaml` (`n_layer: 6, n_embd: 384, vocab_size: 8000`) and
`configs/finetune.yaml` (no `model:` section — architecture always comes
from the loaded checkpoint) need no new fields for this milestone —
initialization and tying are properties of `GPT.__init__`, not
configuration knobs. There is exactly one initialization scheme, applied
unconditionally, the same way there's exactly one LR schedule
(`docs/lr-decay.md` made the same call).

---

## Why GPT-style init uses `std ≈ 0.02`

`0.02` isn't a magic constant — it comes from GPT-2's released config and
has since become the de facto standard starting point for this style of
model. The intuition: at initialization, before any training, you want
every layer's *output* variance to stay in a similar, small, stable range
regardless of `n_embd` — not too large (unstable gradients / saturated
softmax early on) and not so small the model can't distinguish tokens.
`0.02` was chosen empirically for models in the `n_embd` range this
project's `configs/small.yaml` (`384`) and GPT-2-small (`768`) both sit in
— small enough to keep the embedded representations and layer outputs
well-behaved through many stacked pre-norm blocks (`docs/03-architecture.md`
already covers *why* pre-norm exists; this is what makes its initial
conditions correct), large enough that different tokens/positions still
start out distinguishable rather than collapsing toward each other.

Concretely, this milestone applies `N(0, 0.02²)` to:
- `token_embedding.weight`
- `position_embedding.weight`
- every `nn.Linear.weight` in `CausalSelfAttention` and `FeedForward`
  (`qkv_proj`, `out_proj`, both `FeedForward` linears) — **except** the two
  residual-writing projections, which get the extra scaling below on top
  of the same `0.02` base.
- `nn.Linear.bias` — set to `0` (not sampled at all; GPT-2's convention,
  and simpler than the current Kaiming-uniform bias default).

`lm_head` gets no independent init at all under this milestone — see
weight tying, below.

## Why residual/output projections get `1 / sqrt(2 * n_layer)` scaling

Every transformer block adds two deltas onto the residual stream:
`x = x + attn(ln1(x))` and `x = x + ff(ln2(x))` (`docs/03-architecture.md`,
"Residual connections"). Each addition is unscaled by design — that's what
makes the residual path a clean gradient shortcut. But it also means the
variance contributed by each block's two sub-layers *accumulates* additively
as depth increases: with `n_layer` blocks, each contributing a roughly
independent random delta of similar scale, the stream's variance after all
blocks grows roughly proportional to `n_layer` if every projection is
initialized the same way regardless of depth.

The fix (from the GPT-2 paper, "a modified initialization... to account
for the accumulation on the residual path"): scale down specifically the
*last* linear layer in each sub-layer — the one whose output gets added
directly onto the residual stream (`CausalSelfAttention.out_proj`, and the
second `nn.Linear` inside `FeedForward.net`, after the GELU) — by
`1 / sqrt(2 * n_layer)`. The `2` accounts for there being two such
residual-writing sub-layers per block (attention and feed-forward); the
`n_layer` accounts for there being that many blocks stacked. This keeps
the *expected total variance added to the residual stream* roughly
constant regardless of how deep the model is, so a 6-layer model
(`configs/small.yaml`) and a much deeper future config both start training
from similarly well-behaved numbers.

Concretely: those two projections get `N(0, (0.02 / sqrt(2 * n_layer))²)`
instead of the plain `N(0, 0.02²)` every other linear layer gets.

Worked example at `configs/small.yaml`'s `n_layer: 6`:
`1 / sqrt(2 * 6) = 1 / sqrt(12) ≈ 0.2887`, so `out_proj.weight` and
`FeedForward.net[2].weight` get `std ≈ 0.02 * 0.2887 ≈ 0.00577` instead of
`0.02` — roughly 3.5x smaller than the rest of the network's linear
layers.

---

## What weight tying means

`token_embedding.weight` maps a token id to a vector
(`(vocab_size, n_embd)`, looked up by row). `lm_head.weight` maps a
vector back to a score per token id (`(vocab_size, n_embd)`, used as
`x @ weight.T`). These are literally the same shape, and both are, in a
real sense, "the model's opinion of what each token looks like in
`n_embd`-dimensional space" — one direction reads it, the other writes it.
**Weight tying** means using the *same* `nn.Parameter` tensor for both,
instead of two independently-initialized, independently-trained copies.
Concretely: `lm_head.weight is token_embedding.weight` — not
"initialized to the same values," but the same underlying tensor object,
so a gradient update to one *is* an update to the other.

### Why `lm_head.weight` can share `token_embedding.weight`

The shapes already match exactly (`(vocab_size, n_embd)` for both, with
no bias on `lm_head` — already `bias=False` today, so there's no
mismatched bias term to reconcile). Beyond the shape coincidence, sharing
makes conceptual sense: a token whose *input* embedding is close to
another token's (the model treats them similarly when reading) should
plausibly also be a token the model considers when deciding what to
*output* in a similar context — the two roles reinforce the same learned
notion of "what this token means," rather than the model having to learn
two separate, potentially inconsistent, `vocab_size × n_embd`-sized
notions of the same thing.

### How tying reduces parameter count

Removing one of the two `(vocab_size, n_embd)` matrices removes
`vocab_size * n_embd` parameters entirely.

**Worked example**, using `configs/small.yaml`'s numbers
(`vocab_size: 8000`, `n_embd: 384` — the same scale
`docs/03-architecture.md`'s own worked example uses):

```
untied: token_embedding (8000 x 384) + lm_head (8000 x 384)
      = 3,072,000 + 3,072,000 = 6,144,000 parameters

tied:   one shared (8000 x 384) matrix, used both ways
      = 3,072,000 parameters

reduction: 3,072,000 parameters removed (50% of what these two
           layers used to cost together)
```

At `n_embd: 384` with 6 transformer blocks, the embedding/output pair is
a substantial fraction of the model's total parameter count relative to
the transformer blocks themselves (each block is roughly `12 * n_embd²`
parameters — `12 * 384² ≈ 1.77M` per block, `~10.6M` across 6 blocks —
so the embedding/output pair currently rivals the entire stack of blocks
combined). Halving it is not a rounding-error savings at this project's
scale.

---

## What changes for new checkpoints

- `GPT.__init__` gains an explicit initialization pass (e.g. an
  `_init_weights` method applied via `self.apply(self._init_weights)`,
  the standard PyTorch pattern) implementing the `N(0, 0.02²)` /
  `1 / sqrt(2 * n_layer)` scheme above, replacing every layer's library
  default.
- `lm_head` stops being an independently-initialized parameter. The
  recommended implementation: **remove `lm_head` as a separate
  `nn.Linear` module entirely**, and compute the final logits directly
  against `token_embedding.weight` (`F.linear(x, self.token_embedding.weight)`
  — a plain matrix multiply against the embedding table, no separate
  module or parameter needed). This is preferred over keeping `lm_head`
  as an `nn.Linear` and reassigning its `.weight` attribute after
  construction, because it makes the tying unconditional and structural
  (there is no `lm_head.weight` parameter to accidentally re-initialize
  or diverge later) rather than a fragile post-construction patch.
- Every new checkpoint's `model_state_dict` will therefore no longer
  contain an `lm_head.weight` key at all — only `token_embedding.weight`,
  used for both directions. `model_config` (`GPTConfig`) itself needs no
  new field: tying is a property of the code, the same way LR decay
  became the only schedule with no `lr_decay: cosine|none` switch
  (`docs/lr-decay.md` made the identical call, and the same reasoning
  applies here — there is exactly one initialization/tying scheme, not a
  configurable choice per run).
- `save_checkpoint` (`train/loop.py`) and every existing checkpoint
  reader (`load_checkpoint_dict`, `load_pretrained_model`,
  `load_checkpoint_for_resume`) need **no code changes** — they already
  just serialize/deserialize whatever `model.state_dict()` /
  `GPT(model_config)` + `load_state_dict(...)` produce. A smaller
  `state_dict` with one fewer large tensor round-trips through the exact
  same atomic-write (`docs/checkpoint-atomicity.md`) and
  `weights_only=True` (`docs/checkpoint-format.md`) machinery unchanged.

## What happens to old (untied, default-init) checkpoints

This is a real, structural incompatibility, not a cosmetic one — and it
needs to fail loudly, the same way every prior compatibility break in this
project has (tokenizer persistence, checkpoint format hardening, EOS).
Two distinct failure modes, both already exist for free once `lm_head` is
removed as a module:

**Loading an old checkpoint into the new architecture (`load_pretrained_model`, `load_checkpoint_for_resume`).**
An old checkpoint's `model_state_dict` has *both* `token_embedding.weight`
and `lm_head.weight` as independent keys (they were trained
independently — their values differ). The new `GPT` has no `lm_head`
submodule at all, so it has no `lm_head.weight` parameter to receive
that key. `model.load_state_dict(checkpoint["model_state_dict"])` is
called with PyTorch's default `strict=True` everywhere in this codebase
today (`finetune/checkpoint.py`, `train/loop.py` — neither passes
`strict=False`), so this **already raises a clear, specific
`RuntimeError`**: `Unexpected key(s) in state_dict: "lm_head.weight"`.
No new code is needed to detect this — exactly like the checkpoint-format
milestone used `pickle.UnpicklingError` as the natural detection signal
for a legacy checkpoint instead of adding a version field, this milestone
can rely on `load_state_dict`'s own strict-mode error as the detection
signal for an untied legacy checkpoint. **Decision for this milestone:
do not catch or reword this error.** It already clearly names the exact
missing/unexpected key, which is more diagnostic than a generic message
would be, and every other legacy-checkpoint policy in this project
(tokenizer persistence, checkpoint format hardening) has consistently
chosen "fail immediately and clearly" over "guess" or "silently patch it
up." The fix for a user hitting this is the same as every prior
migration: regenerate the checkpoint with the current `scripts/train.py`.
No auto-migration (e.g. detecting the old shape and copying
`lm_head.weight` over `token_embedding.weight`, discarding one of the two
independently-trained matrices) is in scope — silently picking one of two
divergent matrices to keep would be exactly the kind of unverifiable,
silently-lossy fallback this project has consistently refused to add
elsewhere.

**Resuming training on an old checkpoint (`--resume`).**
`load_checkpoint_for_resume` already validates
`checkpoint["model_config"] == model_config` before anything else, but
`GPTConfig`'s fields (`vocab_size`, `context_length`, `n_layer`, `n_head`,
`n_embd`, `dropout`) say nothing about tying — an old checkpoint can have
a `model_config` that matches exactly and still be structurally
incompatible. Resuming hits the same `load_state_dict` strict-mode error
as above, at the same point in `scripts/train.py` (`model.load_state_dict(checkpoint["model_state_dict"])`,
right after `load_checkpoint_for_resume` returns) — so `--resume` against
a pre-tying checkpoint fails the same way, for the same reason, with the
same fix (start a fresh pretraining run instead).

**No migration path, no dual-mode loader, no `tied: bool` config flag.**
Every checkpoint that exists today is from tiny smoke tests
(`docs/checkpoint-format.md` made the same observation about the previous
compatibility break) — there is nothing valuable to preserve by adding
migration complexity for checkpoints nobody depends on. This mirrors the
existing "old checkpoints are not migrated -- regenerate them" precedent
already recorded in `CLAUDE.md`'s tracker for the checkpoint format
hardening milestone.

---

## Explicitly out of scope for this milestone

- Any change to `docs/03-architecture.md`'s described data flow, tensor
  shapes, attention mechanism, or block structure — only *how weights
  start* and *whether two of them are the same tensor* change.
- Any new CLI flag or config field — one initialization/tying scheme,
  applied unconditionally, same reasoning as `docs/lr-decay.md`.
- Auto-migrating old checkpoints to the tied architecture.
- Re-tuning `configs/small.yaml`/`configs/finetune.yaml` hyperparameters
  (learning rate, warmup, etc.) to account for the new initialization —
  worth watching in a real training run, not a change bundled into this
  milestone.

## Status: implemented and tested

Implemented exactly as designed above. `GPT.__init__`
(`src/llm_from_scratch/model/gpt.py`) no longer has an `lm_head` submodule
at all; `forward` computes logits via
`F.linear(x, self.token_embedding.weight)`. `_init_weights` (a
`self.apply(...)`-driven static method) sets `N(0, 0.02²)` on every
`nn.Linear`/`nn.Embedding` weight and zeros every `nn.Linear` bias;
`GPT.__init__` then re-initializes `block.attn.out_proj.weight` and
`block.ff.net[2].weight` for every block with the extra
`1 / sqrt(2 * n_layer)` scaling. `LayerNorm` is untouched (still its
PyTorch default). No changes to `CausalSelfAttention`, `FeedForward`,
`Block`, `GPTConfig`, or any shape/flow described in
`docs/03-architecture.md`.

No checkpoint-loading code changed (`checkpoint.py`,
`train/loop.py`'s `save_checkpoint`/`load_checkpoint_for_resume`,
`finetune/checkpoint.py`'s `load_pretrained_model`) — they already just
serialize/deserialize whatever `state_dict()`/`GPTConfig(...)` produce.

Tests added in `tests/test_model.py` (5 new):
`test_no_separate_lm_head_parameter` (no `lm_head` attribute, no
`"lm_head.weight"` key in `state_dict()`),
`test_output_projection_is_tied_to_token_embedding` (a backward pass
through the output logits produces a nonzero gradient directly on
`token_embedding.weight` — proof it's the same tensor serving both
roles, not a separate copy), `test_tying_reduces_parameter_count_vs_untied`
(total parameter count matches exactly one `vocab_size * n_embd` matrix,
not two), `test_init_weights_use_gpt_style_std` (measures the empirical
std of a non-residual linear/embedding weight against `~0.02`, and a
residual-writing projection's weight against the scaled-down expected
std), `test_linear_biases_initialized_to_zero`. One new test in
`tests/test_finetune.py`,
`test_load_pretrained_model_rejects_legacy_untied_checkpoint`: builds a
synthetic pre-tying checkpoint (a real tied model's `state_dict()` plus
an extra, independently-random `lm_head.weight` key bolted on to imitate
what an actual legacy checkpoint looks like) and confirms
`load_pretrained_model` raises `RuntimeError` mentioning `lm_head.weight`
— exactly the natural `strict=True` failure this doc predicted, with no
new detection code required. Three pre-existing tests
(`tests/test_eval.py::test_evaluate_model_does_not_update_weights`,
`tests/test_train.py::test_estimate_loss_does_not_update_weights`,
`tests/test_train.py::test_weights_change_after_a_training_step`)
referenced `model.lm_head.weight` and were updated to reference
`model.token_embedding.weight` instead — same test intent (weights
frozen during eval, weights move during a training step), pointed at the
parameter that now actually serves that role. Full suite: `135 passed`
(previous 129 + 6 new).

Manual verification: ran `scripts/train.py --device cpu --seed 1` on a
synthetic ~26K-character corpus (`vocab_size: 300` requested → `301`
after EOS, `context_length: 32`, `n_layer: 2`, `n_embd: 32`,
`max_steps: 40`) — trained cleanly, `train_loss`/`val_loss` decreased
step over step exactly like before this milestone. Directly inspected the
saved checkpoint's `model_state_dict` keys and confirmed exactly one
embedding-shaped key (`token_embedding.weight`) and no `lm_head.weight`
key at all. Confirmed the parameter-count reduction directly: this tiny
config's `36,128` total parameters vs. `45,760` an untied version would
have had (`36,128 + vocab_size(301) * n_embd(32) = 45,760`) — the
embedding table appears exactly once. Ran `scripts/evaluate.py` and
`scripts/generate.py` against the checkpoint successfully (generation
output was gibberish, as expected for a model this tiny/undertrained —
same caveat every prior smoke test in this project's history has noted).
Ran `--resume` with `max_steps` raised from 40 to 60: resumed cleanly
from step 40, no `lm_head`-related error, tokenizer/optimizer state
carried over correctly exactly as `docs/checkpoint-resume.md` and
`docs/resume-tokenizer-consistency.md` already guarantee. Ran
`scripts/finetune.py` against the pretrained checkpoint on a 19-example
instruction set; fine-tuning ran end to end (`train_loss` logged every
step, checkpoint saved), and the resulting fine-tuned checkpoint's
`model_state_dict` was confirmed to have the same tied-only key shape.
(The before/after val loss printed `nan` in this particular smoke run —
that's `docs/finetune-loss-masking.md`'s already-documented "window
entirely inside one long prompt" known limitation surfacing because this
smoke test's validation split was tiny, not a regression from this
milestone; `train_loss` itself, which doesn't hit that edge case, stayed
finite and moved normally throughout.) Directly hand-built a legacy-style
checkpoint (a real trained `state_dict()` plus a bolted-on independent
`lm_head.weight` key) and confirmed `load_pretrained_model` raises
`RuntimeError: ... Unexpected key(s) in state_dict: "lm_head.weight"` —
matching this doc's predicted legacy-checkpoint behavior exactly, with no
new code needed to produce it.
