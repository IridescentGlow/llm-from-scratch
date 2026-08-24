# Generation Training-Mode Restoration

## What this is

`GPT.generate()` needs the model in evaluation mode while it runs, but it
never puts the model back the way it found it. This is a mode-leak bug: call
`generate()` from code that's in the middle of training, and training mode
is gone afterward — silently.

## Training mode vs. evaluation mode

Every `nn.Module` carries a `self.training` boolean, toggled by
`model.train()` (sets it `True`) and `model.eval()` (sets it `False`). A few
layers behave differently depending on it — in this model, `nn.Dropout` and
the `F.dropout` call inside `CausalSelfAttention.forward`
(`src/llm_from_scratch/model/gpt.py`). Most layers (linear, layer norm,
embeddings) don't care about this flag at all.

## Why generation needs eval mode

Dropout randomly zeroes out some activations on each forward pass and
rescales the rest — see the "one key idea" in a moment. That randomness is a
training-time regularizer: it stops the model from leaning too hard on any
one connection, by making sure no connection can be relied on every step. It
is not something you want during generation, where the goal is the model's
actual best prediction, not a regularized approximation of it.

**The one key idea to hold onto**: `self.training` isn't a training-vs-not
switch for the whole model — it's specifically "should dropout (and any
other train-only layer) be active right now." `generate()` needs it off,
because dropout would make generation nondeterministic in a way that has
nothing to do with `temperature`, and would just add noise instead of
useful variation. Two calls to `generate()` with `temperature=0` (greedy)
should produce the exact same output every time; dropout left on would
break that.

## The current bug

`GPT.generate` (`src/llm_from_scratch/model/gpt.py`, line 166) does this:

```python
def generate(self, idx, max_new_tokens, temperature=0.0, eos_token_id=None):
    self.eval()
    for _ in range(max_new_tokens):
        ...
    return idx
```

It calls `self.eval()` unconditionally on entry, and never calls
`self.train()` (or anything else) before returning. Whatever mode the model
was in *before* `generate()` was called is simply overwritten and lost.

## Why this is dangerous

Right now nothing in this codebase calls `generate()` mid-training, so the
bug is latent — every existing caller (`scripts/generate.py`,
`load_pretrained_model` callers) either loads a fresh checkpoint (never in
training mode to begin with) or doesn't train afterward. But the method
itself makes no such promise, and a very natural thing to do is call
`generate()` periodically *during* a training run, to eyeball how the model
is improving, then continue training:

```python
model.train()
for step in range(max_steps):
    ...  # forward, backward, optimizer step -- relies on dropout being on
    if step % 500 == 0:
        sample = model.generate(prompt_ids, max_new_tokens=20)
        print(tokenizer.decode(sample[0].tolist()))
    ...  # training continues, but the model is now stuck in eval mode
```

After that `generate()` call, `model.training` is `False` and stays that
way. Every subsequent training step runs with dropout silently disabled —
not a crash, not a warning, just a model that trains as if `dropout=0.0`
from that point on regardless of what the config says. That's a correctness
bug that's very hard to notice: training doesn't error, loss curves still
go down, and nothing points at `generate()` as the cause.

## The correct behavior

`generate()` should leave `self.training` exactly as it found it:

- if the model was in training mode before the call, it should be back in
  training mode after the call returns
- if the model was already in eval mode before the call, it should still be
  in eval mode after — eval mode is *not* forced to flip to train mode

This also must hold on every exit path — including the early return when
EOS is produced, not just the "ran the full `max_new_tokens` loop" path.

### Worked example

```python
model.train()          # model.training == True
model.generate(...)    # internally needs eval mode to run correctly
# model.training == True again here -- restored, not left at False
model.train()           # a no-op in this case, but shown for clarity: the
                         # caller can keep training exactly as before
```

Without restoration, the line `model.training == True` after `generate()`
would actually read `False`, and the caller's own `model.train()` (if they
even remembered to call it — most callers won't, since they never turned
eval mode on in the first place) is the only thing standing between them
and silently-disabled dropout.

## Why `torch.no_grad()` and mode restoration are different problems

Both wrap `generate()`, but they fix unrelated things:

- **`torch.no_grad()`** controls whether PyTorch builds a computation graph
  for backpropagation. It's about *memory and compute during this call* —
  skipping graph-building because `generate()` never calls `.backward()`.
  It has no effect on `self.training` and no lasting effect after the call
  — there's nothing to "leak," since `no_grad()` is a context manager that
  cleanly exits on its own.
- **Mode restoration** controls whether layers like dropout behave
  train-style or eval-style. It's about *what state the model is left in
  after this call*, for whoever uses the model next — potentially a
  training loop that assumed nothing about its own mode changed.

Fixing one doesn't fix the other: `generate()` could keep `torch.no_grad()`
exactly as-is and still leak training mode (today's actual bug), or restore
mode correctly while still wastefully building gradients (a different,
hypothetical bug). They're independent, and both are already partially
handled here — `no_grad()` correctly, mode restoration not yet.

## Status: implemented and tested

`GPT.generate()` (`src/llm_from_scratch/model/gpt.py`) now captures
`was_training = self.training` before switching to eval mode, and restores
it via `self.train(was_training)` in a `finally` block — on every exit
path, including the early-EOS `break` and any exception. No change to
sampling, EOS handling, context truncation, or the function's
signature/return value. Tests added in `tests/test_model.py` (4 new):
`test_generate_restores_training_mode_when_called_from_training_mode`,
`test_generate_leaves_eval_mode_when_called_from_eval_mode`,
`test_generate_restores_training_mode_on_early_eos_stop`,
`test_generate_output_unchanged_by_mode_restoration`. Full suite: 116
passed (previous 112 + 4 new). Manual verification: constructed a model
with `dropout=0.3`, called `model.train()` then `generate()`, confirmed
`model.training` reads `True` immediately afterward (not `False`); called
`generate()` from eval mode and confirmed it stays `False` afterward.

### Implementation (smallest safe change)

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=0.0, eos_token_id=None):
    was_training = self.training
    self.eval()
    try:
        for _ in range(max_new_tokens):
            ...  # unchanged
    finally:
        self.train(was_training)
    return idx
```

- `was_training = self.training` captures the mode on entry, before it's
  overwritten.
- `self.train(was_training)` — `nn.Module.train(mode: bool)` sets
  `self.training = mode` directly; `self.train(True)` is the same as
  `self.train()`, `self.train(False)` is the same as `self.eval()`. This is
  the one-line way to restore either mode without an `if/else`.
- `try/finally` ensures restoration happens even if the loop body raises
  (e.g. a future change that adds validation, or an unexpected tensor
  shape) — generation failing shouldn't *also* leave the model's mode
  corrupted for whatever code runs next.
- No change to sampling logic, EOS handling, context truncation, or the
  function's signature/return value — this only touches mode bookkeeping
  around the existing loop.
