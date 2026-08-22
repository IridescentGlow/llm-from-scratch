# Mid-training checkpointing + resume (cross-cutting milestone)

Not a new stage — this extends Stage 4's pretraining loop, the same way
the tokenizer persistence and device support milestones extended Stages
4–7 without becoming a new numbered stage. See `docs/01-tokenization.md`
and `docs/device-support.md` for that precedent.

## Why saving only at the end is fragile

Today, `train_model` (`src/llm_from_scratch/train/loop.py`) writes
`latest.pt` exactly once, after the `for step in range(config.max_steps)`
loop finishes every single step. If training is interrupted at any point
before that — a crash, a power loss, a killed process, a pre-emptible
cloud GPU reclaimed mid-run — **nothing has been saved**. All compute
spent on every step so far is lost, and the only way forward is to start
over from step 0.

For `configs/small.yaml`'s 5,000 steps this is annoying. For a real run
of hundreds of thousands of steps over hours or days, it's the difference
between "lost a few minutes" and "lost the whole run."

## What a checkpoint is

A **checkpoint** is a snapshot of everything needed to put training back
into *exactly* the state it was in at some point, so it can continue as
if it had never stopped — not just the numbers the model outputs, but
everything the training *process* was carrying at that moment.

## What needs to be saved to resume correctly

The current checkpoint saves two things: `model_state_dict` (the model's
weights) and `model_config` (architecture params, needed to reconstruct
a `GPT` before loading those weights into it). That's enough to run
`evaluate.py` or `generate.py` — pure inference doesn't care how training
got here. It is **not** enough to resume training correctly:

- **`optimizer_state_dict`** — AdamW doesn't just apply the current
  gradient; it keeps a running per-weight estimate of momentum and
  variance (see docs/04-pretraining.md, "One training step") built up
  over every step so far. Restarting with a fresh `AdamW` throws that
  away — the first few steps after resume would behave like the very
  start of training again (noisy, oversized updates), not a smooth
  continuation. **This is why model weights alone are not enough**: the
  weights say *where* the model is, but the optimizer state says *how it
  was moving*, and losing that causes a visible loss spike right after
  resume.
- **Current step** — needed for two things: (1) so the loop runs steps
  4001–5000, not steps 1–5000 again, and (2) so `get_lr(step, ...)`
  (docs/04-pretraining.md, "Learning rate and warmup") computes the
  *correct* point in the warmup/schedule — resuming at step 0 would
  silently replay warmup from scratch on a partially-trained model.
- **`model_config`** — already saved; unchanged by this milestone.
- **Tokenizer** — already saved separately, once, as `tokenizer.json`
  next to `latest.pt` (the tokenizer persistence milestone). It doesn't
  change step to step, so it's saved once and reused across every
  resume — nothing new needed here.
- **Nothing else.** `train_config` (batch size, `max_steps`, etc.) comes
  from the same `--config` file passed to `scripts/train.py` on every
  run, resume included — it isn't part of the checkpoint.

## What "resume training" means

Rebuild the exact same model and optimizer, load the saved weights and
optimizer state into them, and continue the step loop starting at
`step + 1` — running the *remaining* steps up to `max_steps`, with the
learning-rate schedule and optimizer momentum picking up exactly where
they left off. Not: retrain from scratch; not: only reload the weights
and start the optimizer fresh (see above for why that's broken); not: run
another full `max_steps` steps on top of the ones already done.

## Worked example: interrupted at step 4000, resuming from step 4000

Config: `max_steps=5000`, `checkpoint_every=1000` (a new config field —
see below). Training writes `latest.pt` after step 1000, 2000, 3000,
4000. At step 4237, the process is killed (power loss).

- **Without this milestone**: the only checkpoint on disk is from before
  training started (none) — the whole run is lost, restart at step 0.
- **With this milestone**: `latest.pt` holds the state as of step 4000
  (the last periodic save before the crash). Running
  `scripts/train.py --config configs/small.yaml --resume` loads that
  checkpoint, restores the model and optimizer to their step-4000 state,
  and continues the loop from step 4001 through 5000 — 237 steps of
  work (4001–4237) are redone, not all 4237.

## Periodic checkpointing

Write `latest.pt` every `checkpoint_every` steps (a new `TrainConfig`
field, same pattern as `eval_every` — explicit in `configs/*.yaml`, not
hardcoded), in addition to once more at the very end of the run
(`is_last_step`, already the case today). More frequent saving bounds
how much work a crash can lose, at the cost of the (cheap, for this
project's model sizes) time spent writing the file. `latest.pt` is
overwritten in place each time — one file, always the most recent state,
not a growing pile of step-numbered files. Keeping a single, explicit,
`torch.load`-able file matches this milestone's design goal ("keep the
checkpoint format explicit and easy to inspect") and reuses the existing
checkpoint path exactly as `train_model` already computes it.

## What `--resume` does

`scripts/train.py` gains a `--resume` flag. When passed:

1. Look for `latest.pt` in `config.checkpoint_dir` (the same path
   `train_model` already writes to — no new path to configure).
2. If it's missing, or malformed (missing `optimizer_state_dict` or
   `step` — e.g. a checkpoint saved before this milestone existed), or
   its `model_config` doesn't match the config passed in this run: fail
   immediately with a clear, specific error. This mirrors the tokenizer
   persistence milestone's precedent
   (`load_tokenizer_for_checkpoint`'s `FileNotFoundError` for a
   pre-migration checkpoint, `docs/01-tokenization.md`) — a silent
   fallback (e.g. quietly starting fresh, or resuming with a
   randomly-initialized optimizer) would hide a real problem rather than
   surface it.
3. Otherwise, rebuild the model and `AdamW` optimizer, load both state
   dicts, and start the training loop at `step + 1` instead of `0`.

When `--resume` is **not** passed, behavior is unchanged from today:
train from step 0, using whatever fresh model/optimizer `scripts/train.py`
already constructs.

## Reused, unchanged

- **Tokenizer persistence** — `tokenizer.json` is saved once (on the
  first `train_model` call for a run) and never touched by resume;
  `scripts/train.py` still trains the tokenizer up front exactly as
  today.
- **Device handling** — `resolve_device`/`--device` are untouched; a
  resumed run picks a device exactly the same way a fresh run does.
  (Checkpoints still load with `map_location="cpu"` first, same as
  `load_pretrained_model` — see `docs/device-support.md` — then move to
  the resolved device.)
- **Model/data pipeline** — no changes to `GPT`, `GPTConfig`,
  `TokenDataset`, or the tokenizer/data scripts beyond what's listed
  above.

## What this milestone does *not* do

No distributed/multi-GPU checkpointing (sharded state across processes),
no automatic periodic-eviction of old checkpoints, no step-numbered
checkpoint history (`ckpt-1000.pt`, `ckpt-2000.pt`, ...) — just one
`latest.pt` that's overwritten periodically and always resumable, which
is enough for this project's single-device training loop
(docs/04-pretraining.md, "Simplification note").

## Status: implemented and tested

- `TrainConfig` (`src/llm_from_scratch/train/loop.py`) gained a
  `checkpoint_every` field (`configs/small.yaml`: 250, matching
  `eval_every`; `configs/finetune.yaml`: 50).
- `train_model` now writes `latest.pt` every `checkpoint_every` steps (in
  addition to once more at the final step, as before), and the checkpoint
  dict gained two keys: `optimizer_state_dict` (`optimizer.state_dict()`)
  and `step` (`step + 1`, i.e. steps completed so far). `model_state_dict`
  and `model_config` are unchanged.
- `train_model` gained two optional parameters, `start_step: int = 0` and
  `optimizer_state_dict: dict | None = None` — when given, the loop runs
  `range(start_step, max_steps)` instead of `range(max_steps)`, and the
  freshly-constructed `AdamW` optimizer loads that state before the loop
  starts, so warmup/schedule position and AdamW momentum both continue
  from where a prior run left off rather than resetting.
- New `load_checkpoint_for_resume(checkpoint_dir, model_config) -> dict`
  (`src/llm_from_scratch/train/loop.py`, exported from
  `llm_from_scratch.train`): loads `checkpoint_dir/latest.pt`, raising
  `FileNotFoundError` if it doesn't exist, `ValueError` if it's missing
  `optimizer_state_dict`/`step` (a pre-milestone checkpoint), or
  `ValueError` if its `model_config` doesn't match the one passed in —
  no silent fallback in any of these cases, per this doc's design.
- `scripts/train.py` gained `--resume` (a flag, no argument): when given,
  it calls `load_checkpoint_for_resume(train_config.checkpoint_dir,
  model_config)`, loads the returned `model_state_dict` into the
  freshly-constructed `GPT`, and passes `start_step`/`optimizer_state_dict`
  through to `train_model`; without `--resume`, behavior is unchanged
  (fresh model, `start_step=0`, no optimizer state).
- Tokenizer persistence and device handling are untouched, exactly as
  planned: `scripts/train.py` still trains/saves the tokenizer the same
  way every run, and `--device`/`resolve_device` work identically whether
  or not `--resume` is passed.

Tests added in `tests/test_train.py`: `checkpoint_every` written into the
two inline YAML fixtures and `_tiny_train_config`;
`test_checkpoint_written_periodically_not_only_at_end` (spies on
`torch.save` to confirm `latest.pt` is written at steps 3, 6, 9, and the
final step 10, not just once at the end);
`test_load_checkpoint_for_resume_missing_file_raises`;
`test_load_checkpoint_for_resume_rejects_pre_milestone_checkpoint` (a
checkpoint with only `model_state_dict`/`model_config` must raise
`ValueError`, not resume with an assumed step 0);
`test_load_checkpoint_for_resume_rejects_mismatched_model_config`;
`test_resume_continues_from_saved_step_not_zero` (trains 4 steps, resumes
with `max_steps=10`, asserts exactly 6 more training steps run — not 10 —
and the final checkpoint's `step` is 10). Full suite: `69 passed`
(previous 64 + 5 new: the four `load_checkpoint_for_resume`/periodic-save
tests above plus `test_resume_continues_from_saved_step_not_zero`).

Manual end-to-end smoke test (the actual scenario this milestone exists
for): trained a tiny checkpoint (vocab_size 300, context_length 16,
n_layer 2, n_embd 32) with `scripts/train.py --device cpu`, `max_steps:
4000`, `checkpoint_every: 50`, on a synthetic repeating-sentence corpus.
Started the run in the background and sent `SIGKILL` the instant the log
showed `step 150/4000` (confirmed via `kill -0` that the process was
still running at the moment of the kill, i.e. a genuine interruption, not
a race against a run that had already finished). Inspected the surviving
`latest.pt` directly: `step` was exactly `150`, and
`optimizer_state_dict["state"]` was non-empty (real AdamW momentum, not a
fresh optimizer). Ran `scripts/train.py --device cpu --resume` (same
config, unmodified `max_steps: 4000`): it printed `Resuming from step
150/4000`, then continued logging `step 200/4000`, `step 250/4000`, ...
up to `step 4000/4000` — confirming steps 151–200 (not 1–200) ran next,
train loss continued its existing downward trend with no restart spike
(0.0010 at step 150 pre-crash → 0.0008 at step 200 post-resume, smoothly
continuing rather than jumping back up), and the final checkpoint's `step`
was 4000. Also verified the "no silent fallback" requirement directly:
`--resume` against an empty/nonexistent `checkpoint_dir` failed
immediately with the `FileNotFoundError` above; `--resume` against a
hand-crafted checkpoint containing only `model_state_dict`/`model_config`
(simulating a pre-milestone checkpoint) failed immediately with the
`ValueError` "is missing ['optimizer_state_dict', 'step']" message, never
silently starting fresh. Finally, ran `scripts/evaluate.py` and
`scripts/generate.py` against the fully-resumed checkpoint and confirmed
both load and run without error (the checkpoint's two new keys,
`optimizer_state_dict` and `step`, are simply ignored by
`load_pretrained_model`, which only ever reads `model_state_dict` and
`model_config`).
