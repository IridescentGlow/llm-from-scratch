# Atomic checkpoint writes (cross-cutting milestone)

Not a new stage — this hardens the checkpoint/resume milestone
(`docs/checkpoint-resume.md`), the same way tokenizer persistence, device
support, and EOS stopping extended earlier stages without becoming new
numbered stages.

## What can go wrong today

`train_model`'s `save_checkpoint()` (`src/llm_from_scratch/train/loop.py`)
calls `torch.save(checkpoint_dict, checkpoint_path)` where `checkpoint_path`
is `latest.pt` itself — the *live* file that `--resume`, `evaluate.py`, and
`generate.py` all read. `torch.save` isn't instantaneous: it serializes a
few hundred KB–MB of tensors and writes them out over some (small but
nonzero) span of time.

If the process is killed, the machine loses power, or a pre-emptible cloud
GPU is reclaimed **during** that write — not between writes, but mid-write —
`latest.pt` is left holding whatever bytes had made it to disk so far:
neither the old checkpoint (already overwritten) nor a complete new one.
The file exists, but it's garbage.

## Why this matters more than it sounds

The whole point of checkpointing is: *whatever else happens, there is
always a checkpoint on disk you can trust.* `docs/checkpoint-resume.md`
was built specifically so a crash loses at most `checkpoint_every` steps
of work, not the whole run. A truncated `latest.pt` breaks that guarantee
at its weakest point: not "we lost some progress" (acceptable, documented,
bounded) but "we lost *everything*, because the one file we were relying on
is now unreadable." `--resume` would fail with an opaque
`torch.load`/pickle deserialization error instead of the clean, documented
`FileNotFoundError`/`ValueError` cases `docs/checkpoint-resume.md` already
handles — there's currently no way to distinguish "no checkpoint yet" from
"checkpoint exists but is corrupted," and no way to recover the previous
good state, because it's already gone.

## What an atomic write means

**Atomic** here means: from any outside observer's point of view (including
a process that crashes and looks again later), the write either hasn't
happened yet, or has completely happened — there is no in-between state
visible on disk. `latest.pt` is either the old complete checkpoint or the
new complete checkpoint, never a partial one.

## How temp-file-plus-rename provides that guarantee

1. Write the new checkpoint to a *different* path in the same directory —
   e.g. `latest.pt.tmp` — with `torch.save`, exactly as today.
2. Once that write finishes completely, call `os.replace(tmp_path,
   checkpoint_path)` to atomically swap it into place as `latest.pt`.

`os.replace` (on Linux, backed by the `rename()` syscall on the same
filesystem) is atomic at the operating-system level: it changes what name
points to what file in one indivisible step. There is no instant in time
where `latest.pt` points to a half-written file. If the crash happens
during step 1 (writing the temp file), `latest.pt` is untouched — the old
checkpoint is still there, intact. If the crash happens after step 2, the
new checkpoint is fully in place. The only way to lose the *old* checkpoint
is for the *new* one to have fully, successfully replaced it first.

## The guarantee this gives us

At any moment — including immediately after a crash — `checkpoint_dir /
"latest.pt"` is one of exactly two things:

- the last checkpoint that was fully written before this save started, or
- the checkpoint this save just fully finished writing,

and never a partially-written file in between.

## Worked example: power loss during a checkpoint write

Training is at step 6000, `checkpoint_every=1000`, and the last full
checkpoint on disk is from step 6000... say step 5000, and step 6000's
save has just started. Power is lost 40% of the way through writing the
new checkpoint's tensors.

- **Without this milestone**: `torch.save` was writing straight to
  `latest.pt`. The old step-5000 checkpoint has already been overwritten
  with the first 40% of step 6000's bytes. `latest.pt` is now neither
  checkpoint — `torch.load` on it raises a deserialization error. Progress
  is not "lost back to step 5000"; the checkpoint file itself is unusable,
  and there's nothing to resume from at all.
- **With this milestone**: `torch.save` was writing to `latest.pt.tmp`.
  Power loss mid-write leaves `latest.pt.tmp` corrupted, but `latest.pt`
  itself was never touched — it's still the complete step-5000 checkpoint,
  exactly as `docs/checkpoint-resume.md` promises. `--resume` picks up
  from step 5000, loses 1000 steps of work (the same bounded loss the
  original milestone already accepts), and loads a checkpoint that is
  never corrupted.

## Why this doesn't touch the checkpoint format or resume logic

This milestone only changes *how* the bytes get from memory onto the
`latest.pt` path — not *what* those bytes are. `model_state_dict`,
`model_config`, `optimizer_state_dict`, and `step` are unchanged; the
"reject a pre-milestone checkpoint" and "reject a mismatched model_config"
validation in `load_checkpoint_for_resume` is unchanged; `--resume`'s
behavior once a checkpoint is loaded is unchanged. The only observable
difference is that `latest.pt` can no longer be left in a truncated state —
everything downstream keeps working exactly as documented.

## What this milestone does *not* do

No checkpoint history/versioning (still one `latest.pt`, no
`ckpt-1000.pt`-style files — that's an explicit non-goal of
`docs/checkpoint-resume.md` too), no fsync/durability guarantees beyond
what the filesystem already provides for a completed `rename()`, no
protection against disk-full or permissions errors during the temp-file
write itself (those still raise, same as `torch.save` does today — the
old `latest.pt` is simply left untouched in that case too, which is
already the desired outcome).

## Status: implemented and tested

`save_checkpoint()` in `src/llm_from_scratch/train/loop.py` now writes to
`checkpoint_path.with_suffix(".pt.tmp")` (i.e. `latest.pt.tmp`, in the same
directory as `latest.pt`, so `os.replace` stays on one filesystem and is
guaranteed atomic) via the existing `torch.save` call, then calls
`os.replace(tmp_path, checkpoint_path)` to publish it as `latest.pt`. No
changes to `load_checkpoint_for_resume`, `train_model`'s public signature,
or any checkpoint dict contents.

Tests added in `tests/test_train.py`:
`test_checkpoint_save_leaves_no_tmp_file_behind` (a normal successful run
ends with only `latest.pt` on disk, no lingering `.tmp` file);
`test_checkpoint_survives_interrupted_write` (monkeypatches `os.replace` to
raise on the second call, simulating a crash after the temp file is fully
written but before the rename that would publish it — asserts the
*previous* successful checkpoint, from step 3, is still on disk and loads
cleanly, not a half-written step-6 file). Full suite: `83 passed` (previous
81 + 2 new).

Manual end-to-end smoke test (a real interruption, not just the unit
tests): trained a tiny checkpoint (vocab_size 300, context_length 16,
n_layer 2, n_embd 32) on a varied synthetic corpus via `scripts/train.py
--device cpu`, `checkpoint_every: 50`. `SIGKILL`ed the process mid-run
(confirmed via `kill -0` that it was still alive at the moment of the
kill); the checkpoint directory afterward contained only a clean
`latest.pt` (577,291 bytes, `torch.load`-able, `step=7450`,
non-empty `optimizer_state_dict`) — no `latest.pt.tmp` left behind.
Resumed via `--resume`; a second interruption happened along the way (a
`timeout` wrapper killed that first resume attempt mid-run too) and
`latest.pt` again survived intact and resumable (`step=16250`) — two real
interruptions in a row, neither one corrupted the checkpoint. A final
`--resume` run continued cleanly from step 16250 through the configured
16400, logging steps 16300, 16350, 16400 (i.e. 16251–16400 ran, not
0–16400), with the final checkpoint's `step` equal to 16400. `latest.pt`
was the only checkpoint file present at every point in this test — no
`.tmp` artifact ever survived a successful save.
