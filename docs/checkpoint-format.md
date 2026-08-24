# Checkpoint format hardening (cross-cutting milestone)

Not a new stage — this extends the checkpoint/resume milestone
(`docs/checkpoint-resume.md`) and tokenizer persistence milestone
(`docs/01-tokenization.md`), the same way earlier cross-cutting
milestones extended existing stages without becoming a new numbered one.

## What the current checkpoint contains

`latest.pt` (written by `save_checkpoint()` in
`src/llm_from_scratch/train/loop.py`) is a single file holding four
things, all passed to `torch.save()` together:

- `model_state_dict` — the model's learned weights (a dict of tensors).
- `model_config` — a `GPTConfig` **object** (the dataclass defined in
  `src/llm_from_scratch/model/gpt.py`), not a plain dict.
- `optimizer_state_dict` — AdamW's momentum/variance state (see
  `docs/checkpoint-resume.md`).
- `step` — how many training steps have completed.

Every reader (`load_pretrained_model` in
`src/llm_from_scratch/finetune/checkpoint.py`,
`load_checkpoint_for_resume` in `src/llm_from_scratch/train/loop.py`, and
`scripts/evaluate.py`'s own inline `torch.load` call) reads it back the
same way: `torch.load(path, weights_only=False, map_location="cpu")`.

## Why `GPTConfig` is currently serialized through pickle

`torch.save()` doesn't know how to write an arbitrary Python object like
`GPTConfig` (a `@dataclass`) as plain structured data — tensors have a
dedicated binary format, but a `GPTConfig` instance is just a regular
Python object with no such format. So `torch.save` falls back to
Python's built-in `pickle` module for anything that isn't a tensor.
Pickle doesn't just save *data* — it saves enough information to
**reconstruct the exact Python object**, including which class to
instantiate and how. That's convenient (today's code gets a real,
ready-to-use `GPTConfig` back from `torch.load`, no extra step), but it's
also the entire reason `weights_only=False` exists and is needed today.

## Why `torch.load(..., weights_only=False)` is a security concern for untrusted checkpoints

Unpickling isn't just "reading data" — it's a small interpreter that can
be told to construct arbitrary objects, including ones whose
construction has side effects. `weights_only=False` tells `torch.load`
"trust this file enough to run its embedded unpickling instructions
without restriction." For a checkpoint you trained yourself, on your own
machine, that's harmless. But a checkpoint is just a file — nothing
stops one from being downloaded, shared, or copied from somewhere else,
the same way a Python `.pkl` file can be. **A malicious `.pt` file can be
crafted to execute arbitrary code the moment it's loaded** — not when
its weights are *used*, but during the `torch.load()` call itself, before
a single number has been read out of it. This project doesn't currently
ship or download checkpoints from anyone else, but "we control the only
source today" isn't a safety property of the *code* — it's a property
of *current usage*, which can change (a checkpoint shared over Slack, a
downloaded pretrained model, a CI artifact from a fork). The loading code
should be safe regardless of where the file came from, not merely safe
today because nobody's tried to exploit it yet.

## Why a checkpoint should contain plain data plus tensor weights, not executable Python objects

The fix isn't "stop using pickle for everything" — `torch.save`/`load`
fundamentally use pickle as their container format, and that's fine for
the *tensors themselves* (PyTorch's own tensor unpickling is safe and
restricted, which is exactly what `weights_only=True` relies on — see
below). The fix is narrower: **stop asking pickle to reconstruct
arbitrary custom Python classes**, and only ever ask it to reconstruct
things from a short, known-safe list — tensors, and plain built-in types
like `dict`, `list`, `int`, `str`, `float`, `bool`. A `GPTConfig` object
is the one piece of the current checkpoint that isn't on that safe list.
Everything else already is: `model_state_dict` and `optimizer_state_dict`
are already plain dicts of tensors/ints/dicts, and `step` is already a
plain `int`.

## How the model configuration can be represented as a plain dictionary

`GPTConfig` has exactly six fields, all plain Python values already
(`vocab_size: int`, `context_length: int`, `n_layer: int`, `n_head: int`,
`n_embd: int`, `dropout: float`). Nothing about it needs custom-class
behavior to survive a save/load round trip — `dataclasses.asdict(config)`
turns it into an ordinary `dict[str, int | float]`, which is exactly the
kind of value `weights_only=True` already allows. Saving that dict
instead of the `GPTConfig` instance loses nothing: every field is still
there, in the same shape, just not wrapped in a class pickle has to be
trusted to reconstruct. The checkpoint becomes JSON-compatible data
(the dict) plus tensors (the weights/optimizer state) plus one plain int
(`step`) — nothing else.

## How `weights_only=True` changes the trust model

With `weights_only=True`, `torch.load` restricts what unpickling is
allowed to construct to a fixed, safe allow-list — tensors and
storages via PyTorch's own restricted tensor-unpickler, plus ordinary
Python containers and primitives (`dict`, `list`, `tuple`, `int`, `float`,
`str`, `bool`, `None`, and a few more). It does **not** allow
reconstructing arbitrary classes like a hand-defined dataclass, and
critically, it does not run arbitrary constructor code at all — loading
either succeeds by building only safe, inert data, or it refuses and
raises an error. That's the actual security property: not "this
particular file happens to be fine," but "this loading *code path*
cannot be tricked into executing anything, no matter what bytes the file
contains." Once `model_config` is a plain dict instead of a `GPTConfig`
object, nothing in the checkpoint needs anything outside that safe list,
and `weights_only=True` (the actual default in current PyTorch versions,
confirmed against this project's installed `torch==2.13.0`) works
without any change in what data comes back out.

## What must remain in the checkpoint

Every consumer's needs stay exactly as they are today — this milestone
changes *how* `model_config` is encoded, not what information exists or
what any script needs to keep working:

- **Model weights** (`model_state_dict`) — needed by every script that
  loads a checkpoint at all.
- **Optimizer state** (`optimizer_state_dict`) — needed by `--resume`
  only, per `docs/checkpoint-resume.md`; untouched by this milestone
  (already plain dicts/tensors, already safe).
- **Training step** (`step`) — needed by `--resume` only; already a
  plain `int`, already safe.
- **Model configuration** — needed by every script, to reconstruct a
  `GPT` before loading weights into it. This is the one thing changing
  shape (object → dict), not disappearing.
- **Tokenizer metadata/path behavior** — completely untouched.
  `tokenizer.json` is a separate file, saved and loaded independently of
  `latest.pt` (see `docs/01-tokenization.md`, "Tokenizer persistence").
  Nothing about this milestone touches how or where the tokenizer is
  saved, loaded, or referenced.

## Why the checkpoint filename/path should stay the same

`latest.pt` remains `latest.pt`, in the same `checkpoint_dir`, saved
via the same atomic temp-file-then-`os.replace` sequence
(`docs/checkpoint-atomicity.md`) — this milestone changes what's inside
the pickle, not the file layout around it. Nothing downstream
(`--resume`'s checkpoint discovery, `tokenizer.json`'s sibling-file
convention, `docs/checkpoint-resume.md`'s design) has any reason to
change paths just because the internal encoding of one field changed.
Keeping the path stable also means this remains a drop-in change for
anyone with scripts or tooling that already point at
`checkpoint_dir/latest.pt`.

## How old checkpoints will be handled

**Refuse, don't reconstruct** — the same policy this project already
uses for a checkpoint missing `tokenizer.json`
(`docs/01-tokenization.md`) and for a checkpoint missing
`optimizer_state_dict`/`step` (`docs/checkpoint-resume.md`). A checkpoint
saved before this milestone has `model_config` pickled as a real
`GPTConfig` object, which `weights_only=True` will refuse to construct —
that refusal is not a bug to work around, it's the exact protection this
milestone exists to add. Loaders catch that specific failure and raise a
clear, explicit error explaining the checkpoint predates safe checkpoint
serialization and must be regenerated with the current
`scripts/train.py`. There is **no fallback to `weights_only=False`**
anywhere in this milestone's code — that would silently reintroduce the
exact risk being closed, defeating the point for the one case (an
untrusted or old file) where it matters most. No new "checkpoint version"
field is introduced to detect this — `weights_only=True`'s own refusal
*is* the detection mechanism, so a separate version marker would be
redundant.

## Worked example: current checkpoint vs. the new safe structure

**Today**, `torch.save()` on a checkpoint effectively pickles something
shaped like:

```python
{
    "model_state_dict": OrderedDict[str, Tensor],          # safe
    "model_config": GPTConfig(vocab_size=8000, context_length=256,
                               n_layer=6, n_head=6, n_embd=384,
                               dropout=0.1),                # <-- a real object
    "optimizer_state_dict": {...tensors and ints...},       # safe
    "step": 5000,                                           # safe
}
```

Loading this with `weights_only=True` fails immediately, before any
weights are read, because unpickling refuses to construct the
`GPTConfig` instance — it's not on the safe list. That's the *bug* this
milestone fixes, demonstrated directly: right now, the only thing
standing between this project and `weights_only=True` is one dataclass
that doesn't need to be a dataclass on disk at all.

**After this milestone**, the same checkpoint is saved as:

```python
{
    "model_state_dict": OrderedDict[str, Tensor],           # unchanged
    "model_config": {                                       # <-- now a plain dict
        "vocab_size": 8000, "context_length": 256,
        "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.1,
    },
    "optimizer_state_dict": {...tensors and ints...},        # unchanged
    "step": 5000,                                            # unchanged
}
```

`torch.load(path, weights_only=True)` now succeeds — every value in the
file is either a tensor or a built-in Python type. The loader then does
one explicit step: `GPTConfig(**checkpoint["model_config"])`, turning the
plain dict back into the real `GPTConfig` object the rest of the code
already expects (`GPT(model_config)`, `load_checkpoint_for_resume`'s
config-match check, etc.) — reconstruction is explicit code the project
controls, not something pickle is trusted to do implicitly.

## What this milestone does *not* do

No change to `BPETokenizer`/`tokenizer.json` (already plain JSON, already
safe — see `docs/01-tokenization.md`). No change to the model
architecture (`GPT`, `GPTConfig`'s fields, or how they're used once
reconstructed). No new checkpoint versioning field. No compatibility path
that still calls `weights_only=False` under any circumstance, including
for old checkpoints. No change to `checkpoint_every`/atomic-write
behavior (`docs/checkpoint-atomicity.md`). No change to
`--resume`'s validation *logic* (missing-file, missing-fields,
mismatched-config checks all stay — only the mechanics of how
`model_config` is read back change underneath them).

## Status: implemented and tested

New module `src/llm_from_scratch/checkpoint.py` adds
`load_checkpoint_dict(path, map_location="cpu") -> dict`, the one place
in the project that calls `torch.load` on a checkpoint. It always uses
`weights_only=True`; if that raises `pickle.UnpicklingError` (the case
for any checkpoint saved before this milestone, whose `model_config` was
pickled as a real `GPTConfig` object), it's caught and re-raised as a
`ValueError` with an explicit "regenerate with the current
scripts/train.py" explanation -- never retried with
`weights_only=False`. On success, `model_config` (a plain dict on disk)
is reconstructed into a real `GPTConfig` via `GPTConfig(**...)` before
the checkpoint dict is returned, so every caller still gets the same
shape of object as before.

`save_checkpoint()` (`src/llm_from_scratch/train/loop.py`) now writes
`"model_config": asdict(model.config)` instead of the `GPTConfig` object
itself -- the only change to what gets written to disk.
`load_checkpoint_for_resume` (same file) now calls
`load_checkpoint_dict` instead of its own `torch.load`, keeping all of
its existing validation (missing `optimizer_state_dict`/`step`,
mismatched `model_config`) unchanged -- it just runs against the
reconstructed `GPTConfig` the shared loader now returns.
`load_pretrained_model` (`src/llm_from_scratch/finetune/checkpoint.py`)
likewise now calls `load_checkpoint_dict` instead of its own
`torch.load`. `scripts/evaluate.py` no longer hand-rolls a `torch.load`
call at all -- it calls `load_pretrained_model`, the same helper
`scripts/generate.py` and `scripts/finetune.py` (via
`load_pretrained_model`) already used, removing the last duplicated
checkpoint-loading code path. No change to
`docs/checkpoint-resume.md`'s validation *logic*, `docs/checkpoint-
atomicity.md`'s atomic-write mechanics, `docs/01-tokenization.md`'s
tokenizer persistence, or `GPT`/`GPTConfig` themselves.

Tests: new `tests/test_checkpoint.py` (6) covering
`load_checkpoint_dict` directly -- loads a current-format checkpoint
under `weights_only=True`, reconstructs `GPTConfig` correctly, raises
`FileNotFoundError` for a missing file, raises a clear `ValueError` (not
a silent accept) for a legacy pickled-`GPTConfig` checkpoint, confirms
no fallback to `weights_only=False` occurs, and a grep-level guardrail
asserting no file under `src/` or `scripts/` contains a
`weights_only=False` call anywhere. Two more targeted tests were added
alongside the existing ones: `test_load_pretrained_model_rejects_
legacy_pickled_config` (`tests/test_finetune.py`) and
`test_load_checkpoint_for_resume_rejects_legacy_pickled_config`
(`tests/test_train.py`), so both public loaders are proven to reject a
legacy checkpoint, not just the shared low-level function. Existing
tests that hand-crafted checkpoints (`tests/test_train.py`,
`tests/test_finetune.py`, `tests/test_generate.py`,
`tests/test_train_resume_tokenizer.py`) were updated to save
`model_config` via `dataclasses.asdict(...)` (matching the new safe
format) except where a test's specific purpose is to construct a legacy
checkpoint on purpose; all remaining direct `torch.load(...)` calls in
tests were switched from `weights_only=False` to `weights_only=True`.
Full suite: `112 passed` (previous 104 + 8 new).

Manual end-to-end smoke test covering every consumer, not just unit
tests: trained a tiny checkpoint (`vocab_size 300`, `context_length 16`,
`n_layer 2`, `n_embd 32`, `max_steps 100`) via `scripts/train.py --device
cpu`; confirmed `torch.load(..., weights_only=True)` on the resulting
`latest.pt` succeeds directly and `model_config` comes back as a plain
`dict`. Ran `scripts/evaluate.py` (val_loss/perplexity printed
successfully) and `scripts/generate.py` (produced coherent continuation
text) against it. Ran `scripts/finetune.py` against it on a small
instruction set -- val loss dropped from 9.39 to 0.49, generation
visibly adapted to the instruction/response format. Ran `scripts/train.py
--resume` with a higher `max_steps` -- printed "Resuming with tokenizer
loaded from ..." and "Resuming from step 100/150", trained through to
step 150/150, and the resulting checkpoint again loaded successfully
under `weights_only=True`. Separately, hand-crafted a legacy-format
checkpoint (`model_config` saved as a real `GPTConfig` object, exactly
how every checkpoint was written before this milestone) and confirmed
both `scripts/generate.py` and `scripts/evaluate.py` fail immediately
with the explicit "predates safe checkpoint serialization ... Regenerate
it with the current scripts/train.py" error, not a silent
`weights_only=False` fallback.
