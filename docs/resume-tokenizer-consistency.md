# Resume tokenizer consistency (cross-cutting milestone)

Not a new stage — this extends the checkpoint/resume milestone
(`docs/checkpoint-resume.md`) and reuses the tokenizer persistence
milestone (`docs/01-tokenization.md`), the same way earlier cross-cutting
milestones extended existing stages without becoming a new numbered one.

## What `--resume` currently does with the tokenizer

`scripts/train.py` trains a tokenizer from scratch on every single run,
`--resume` or not:

```python
tokenizer = BPETokenizer()
tokenizer.train(corpus, vocab_size=model_config.vocab_size)
tokenizer.add_eos_token()
```

This happens *before* the script even checks `args.resume`. So a resumed
run re-reads every `.txt` file in `data/raw/`, re-runs BPE training on
whatever text it finds there today, and hands the *freshly retrained*
tokenizer to `train_model` — completely independent of the
`tokenizer.json` that was saved next to the checkpoint being resumed.

## Why retraining is unnecessary

A tokenizer doesn't change step to step. `docs/checkpoint-resume.md`
already says this plainly: *"Tokenizer — already saved separately, once,
as `tokenizer.json` next to `latest.pt` ... It doesn't change step to
step, so it's saved once and reused across every resume — nothing new
needed here."* The file already exists, on disk, next to the checkpoint
being resumed. Loading it is one `BPETokenizer.load()` call, using the
same `load_tokenizer_for_checkpoint` helper `scripts/evaluate.py`,
`scripts/finetune.py`, and `scripts/generate.py` already call. Retraining
is strictly more work for a result that should already be sitting on
disk.

## Why it's unsafe to assume the raw corpus is unchanged

`BPETokenizer.train()` learns its merges (and therefore its entire
id-to-string mapping) purely from the frequency statistics of whatever
text it's given. Nothing about `--resume` guarantees that text is the
same text the original run trained on:

- Someone might add, edit, or remove files in `data/raw/` between the
  original run and a later `--resume` — for entirely unrelated reasons
  (new training data staged for the *next* run, a typo fix, a cleanup).
- The checkpoint might be resumed on a different machine, or after a
  `git pull`, where `data/raw/` doesn't match what it was when training
  started.
- Even the *exact same files*, re-globbed, aren't guaranteed to be read
  back in the same order (see below) unless something enforces it.

Nothing currently *checks* any of this before retraining — it just
happens silently, every time.

## How a changed corpus, file ordering, or training conditions produce a different id mapping

BPE training is deterministic *given* a fixed corpus string and
`vocab_size` — but "fixed corpus string" is doing a lot of work in that
sentence:

- **Changed corpus text.** `BPETokenizer.train()` counts adjacent byte
  pairs and merges whichever is most frequent, repeatedly. Add, remove,
  or edit even one file's content and pair frequencies shift — a merge
  that used to win by one occurrence might now lose to a different pair,
  changing every merge id from that point forward (merge order determines
  ids: `256 + i` for the `i`-th merge). One changed sentence can cascade
  into a completely different vocabulary.
- **Changed file ordering.** `scripts/train.py` builds the corpus as
  `"\n".join(f.read_text() for f in text_files)`, where `text_files =
  sorted(raw_path.glob("*.txt"))`. This is deterministic *today* because
  of the explicit `sorted()` — but it depends on filenames sorting the
  same way, which breaks the moment a file is renamed, or a new file
  lands with a name that sorts earlier than existing ones. Even though
  every file's *content* is unchanged, concatenating them in a different
  order changes adjacency at the file boundaries, and — if
  pre-tokenization chunk boundaries shift as a result — can change pair
  counts, which again cascades into different merge ids.
- **Changed training conditions.** `vocab_size` comes from
  `model_config.vocab_size`, read from whatever `--config` file is passed
  at resume time. Point `--resume` at a different (or edited) config —
  even by accident — and `train()` runs with a different target
  `vocab_size`, producing a different number of merges and, again, a
  different id mapping, entirely independent of the corpus.

Any one of these silently produces a tokenizer whose id-to-string mapping
differs from the one the checkpoint's embedding table was actually
trained under — with no error, no warning, and no visible symptom beyond
training quietly proceeding on now-mismatched data.

## Why loading the persisted tokenizer gives the same guarantee already used elsewhere

`docs/01-tokenization.md`'s tokenizer persistence milestone exists to
solve exactly this problem for evaluation, fine-tuning, and generation:
each of `scripts/evaluate.py`, `scripts/finetune.py`, and
`scripts/generate.py` calls `load_tokenizer_for_checkpoint` instead of
retraining, specifically so id 4171 keeps meaning the same string
everywhere that checkpoint's weights are ever used again. A resumed
training run is not conceptually different from those three cases — it's
still "using this checkpoint's weights again," just to keep training
them rather than to run inference. There's no reason resume should be the
one caller of a checkpoint that still retrains a tokenizer from a corpus;
it should get the identical guarantee the other three already have, via
the identical helper.

## How a resumed run should use the tokenizer saved alongside the checkpoint

On `--resume`, `scripts/train.py` should call
`load_tokenizer_for_checkpoint(checkpoint_path)` — the same helper
already imported by the other three scripts — pointed at
`checkpoint_dir/latest.pt`, and use the tokenizer it returns instead of
calling `BPETokenizer().train(corpus, ...)`. No new tokenizer logic is
needed; this is entirely a matter of calling an existing function instead
of the existing `train()` call, conditioned on `args.resume`.

A **fresh** (non-resumed) run is unaffected: it still has no prior
checkpoint to load a tokenizer from, so it still trains one from
`data/raw/`, exactly as today.

## What should happen if the checkpoint has no tokenizer file

Nothing new — `load_tokenizer_for_checkpoint` already raises
`FileNotFoundError` with an explicit explanation when `tokenizer.json` is
missing (a checkpoint that predates tokenizer persistence). `--resume`
should let that propagate rather than falling back to retraining, for the
same reason `scripts/evaluate.py`/`scripts/finetune.py`/
`scripts/generate.py` already do: a retrained fallback's correctness
would depend on an unverifiable assumption (the raw corpus being
byte-identical to what pretraining used), which is precisely the failure
mode this milestone exists to close. An old, pre-tokenizer-persistence
checkpoint already can't be resumed today either, for the same reason
`load_checkpoint_for_resume` rejects it (missing
`optimizer_state_dict`/`step`) — this doesn't change that; it just
extends the same "refuse, don't reconstruct" policy to the tokenizer
specifically.

## Worked example: checkpoint made with tokenizer A, resume accidentally reconstructing tokenizer B

1. **Original run.** `data/raw/story.txt` contains a corpus where, among
   other merges, byte pair `(t, h)` happens to be the single most
   frequent pair, so it's the very first merge learned: merge 0 → new
   token id `256`, meaning `"th"`. Training runs for 4,000 of
   `max_steps=5,000`, checkpointing periodically. The embedding table's
   row 256 spends those 4,000 steps being nudged into a vector that means
   "this position is the start of a `th`-initial fragment." `tokenizer.json`
   saved next to `latest.pt` records this exact mapping — call it
   **tokenizer A**, where id 256 = `"th"`.
2. **The process is killed** (crash, pre-emption, whatever) at step 4,000,
   with `latest.pt` and `tokenizer.json` both intact on disk.
3. **Before resuming**, someone drops a second file,
   `data/raw/appendix.txt`, into the same directory — unrelated
   preparation for a *future* run, or just a file that happened to be
   staged there. Nobody intends this to affect the run in progress.
4. **`scripts/train.py --resume` is run.** Today, it re-globs
   `data/raw/*.txt`, now finds *two* files, concatenates them, and
   retrains BPE from scratch on the combined text. The added file's
   content shifts pair frequencies enough that this time `(t, h)` isn't
   the most frequent pair — say `(i, n)` is, so merge 0 now produces id
   `256` meaning `"in"` instead. Call this **tokenizer B**.
5. **Training resumes** using tokenizer B to encode data, feeding id `256`
   into the model every time `"in"` appears in the training data — but
   the model's embedding row 256 still encodes what it learned under
   tokenizer A: "start of a `th`-initial fragment." The two meanings have
   nothing to do with each other. Training proceeds, loss numbers still
   move, nothing crashes — but from step 4,001 onward, the model is
   learning against a corrupted signal, exactly the way
   `docs/01-tokenization.md`'s fine-tuning worked example describes for
   id `4171`.

With this milestone: step 4 above instead calls
`load_tokenizer_for_checkpoint`, which loads tokenizer A straight from
`tokenizer.json` — untouched by the new file in `data/raw/` — and
training resumes using the exact same id-to-string mapping the first
4,000 steps were trained under.

## What this milestone does *not* do

No change to how tokenizers are trained (`BPETokenizer.train()` itself is
untouched), no change to checkpoint contents (`model_state_dict`,
`model_config`, `optimizer_state_dict`, `step` stay exactly as
`docs/checkpoint-resume.md` defined them), and no change to
`scripts/evaluate.py`, `scripts/finetune.py`, or `scripts/generate.py` —
they already do this correctly and are out of scope here. Fresh (non-
resumed) training is unaffected: it still trains and saves a tokenizer
exactly as it does today.

## Status: implemented and tested

`scripts/train.py` now branches on `args.resume` right where the
tokenizer used to be unconditionally trained: when `--resume` is passed,
it calls `load_tokenizer_for_checkpoint(checkpoint_dir / "latest.pt")` —
the same helper already used by `scripts/evaluate.py`,
`scripts/finetune.py`, and `scripts/generate.py` — instead of calling
`BPETokenizer().train(corpus, ...)`. A fresh (non-resumed) run is
unchanged: it still trains and saves a new tokenizer exactly as before.
Either way, `model_config.vocab_size` is set from the resulting
tokenizer's `vocab_size` (loaded or freshly trained), and the raw corpus
is still read and encoded with whichever tokenizer was obtained — a
resumed run still needs `token_ids` for training data, it just gets them
from the *persisted* tokenizer's mapping, not a retrained one. No changes
to `BPETokenizer.train()`, checkpoint contents, or
`scripts/evaluate.py`/`scripts/finetune.py`/`scripts/generate.py`.

Tests added in `tests/test_train_resume_tokenizer.py` (5 new, loading
`scripts/train.py` directly via `importlib` since `scripts/` has no
`__init__.py`):
`test_fresh_training_creates_a_tokenizer`,
`test_resume_loads_the_saved_tokenizer_not_a_retrained_one` (changes
`data/raw/`'s content between the original run and `--resume`, asserts
the resumed `tokenizer.json` is byte-for-byte the same merges/vocab/EOS id
as before the resume, not retrained on the new text),
`test_changing_raw_corpus_does_not_affect_resumed_training_data` (goes
one step further: asserts the actual `tokens.bin` written during resume
matches the *original* tokenizer's encoding of the new corpus text, not a
retrained tokenizer's encoding),
`test_resume_with_no_saved_tokenizer_fails_clearly` (a checkpoint with no
`tokenizer.json` next to it raises `FileNotFoundError` on `--resume`, not
a silent retrain), and
`test_resumed_training_still_reaches_expected_step` (resume still
continues the step counter correctly, unaffected by this change). Full
suite: `104 passed` (previous 99 + 5 new).

Manual end-to-end smoke test (not just unit tests): trained a tiny
checkpoint (`vocab_size 300`, `context_length 16`, `n_layer 2`, `n_embd
32`) via `scripts/train.py --device cpu` on a synthetic corpus,
`max_steps: 100`, `checkpoint_every: 50` — confirmed `tokenizer.json` was
written. Replaced the raw corpus file with completely different text
(different vocabulary, no word overlap with the original), then ran
`scripts/train.py --device cpu --resume` with `max_steps: 150`. Output
showed `Resuming with tokenizer loaded from .../tokenizer.json` and
**no** `Training tokenizer on ... characters...` line (confirming
retraining was actually skipped, not just harmlessly repeated), then
`Resuming from step 100/150` through `step 150/150`. `diff` against a
copy of `tokenizer.json` taken before the resume showed **zero
difference** — the tokenizer was untouched by the changed corpus, exactly
as designed. Separately, removed `tokenizer.json` from the checkpoint
directory and re-ran `--resume`: it failed immediately with the same
explicit `FileNotFoundError` `load_tokenizer_for_checkpoint` already
raises elsewhere, not a silent retrain.
