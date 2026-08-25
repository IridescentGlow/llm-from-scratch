# Token cache / corpus reuse (cross-cutting milestone)

Not a new stage — this extends Stage 2's data pipeline and Stage 4's
pretraining loop, the same way checkpoint/resume and tokenizer performance
extended earlier stages without becoming a new numbered one. See
`docs/checkpoint-resume.md` and `docs/tokenizer-performance.md` for that
precedent.

## What `tokens.bin` is today

`tokens.bin` (`data/processed/tokens.bin`) is the raw pretraining corpus,
already tokenized into integer ids, written to disk as a flat binary file
of `uint16`s (`src/llm_from_scratch/data/tokens.py`, `write_token_ids`).
`load_token_ids` memory-maps it back for `TokenDataset` to slice into
training windows. It exists so the model doesn't re-tokenize text on every
batch — the whole corpus is encoded once per script run, then reused for
every step of that run.

## Why every run re-reads and re-encodes the whole corpus

`scripts/train.py` and `scripts/evaluate.py` both do the same thing on
every invocation, `--resume` included:

```python
text_files = sorted(raw_path.glob("*.txt"))
corpus = "\n".join(f.read_text() for f in text_files)
...
token_ids = tokenizer.encode(corpus)          # re-encodes, every run
write_token_ids(token_ids, tokens_path)       # re-writes tokens.bin, every run
```

Nothing checks whether `tokens.bin` already holds the correct result of
encoding this exact corpus with this exact tokenizer. Every run pays the
full cost of `tokenizer.encode(corpus)` again, even a `--resume` run that
is, by definition, continuing with the *same* tokenizer and (usually) the
*same* corpus it already encoded once before.

## Why this becomes expensive on a large corpus

`docs/tokenizer-performance.md` already made `encode()` efficient — chunked,
not quadratic — but "efficient" still means work proportional to corpus
size, `O(corpus length)`. At the project's stated "a few hundred MB" target
scale, that's still tens of seconds to minutes of pure-Python regex
splitting and merge-lookup work, spent identically on every single run.
For `scripts/train.py --resume`, this cost is paid *before* a single
training step runs — a crash-and-resume cycle that should pick up in
seconds instead re-pays a large fixed tax first, every time. For
`scripts/evaluate.py`, run repeatedly during development to check a
checkpoint, the exact same re-encode happens every time with an identical
result.

## Why `--resume` in particular should never re-encode

`docs/resume-tokenizer-consistency.md` already established that `--resume`
must load the checkpoint's *exact* persisted tokenizer, not retrain one —
otherwise token ids could silently mean something different than what the
embedding table learned. That milestone fixed *which tokenizer* resume
uses. It didn't fix the fact that resume still re-runs `encode()` from
scratch with that tokenizer, every time, even though a resumed run's whole
premise is "continue what was already running" — the corpus and tokenizer
are, in the ordinary case, identical to the ones that already produced a
`tokens.bin` earlier in this same run's lineage. Re-encoding on resume is
pure waste in the common case, and the risk this milestone has to guard
against is the *uncommon* case: the corpus or tokenizer having actually
changed since that `tokens.bin` was written, in which case reusing it
would silently train on the wrong ids — precisely the failure mode
`docs/01-tokenization.md`'s tokenizer persistence section already warns
about, just from a new angle (a stale cache instead of a mismatched
tokenizer).

## What it means to cache tokenized data

Caching here means: before encoding, check whether a previously-encoded
`tokens.bin` already holds the exact result that encoding-now would
produce, and if so, load it instead of redoing the work. "Exact result"
depends on two things that must both still match reality:

1. **Which corpus** was encoded (its content, not just its file names —
   editing a `.txt` file in place, adding one, or reordering the glob must
   all count as "changed").
2. **Which tokenizer** did the encoding (its merges/vocab — retraining BPE
   even on the *same* corpus text is deterministic and would reproduce the
   same tokenizer, but a different `vocab_size` target, or genuinely
   different corpus content, produces a different tokenizer with different
   merges).

If either has changed since the cache was written, the cache is stale and
must not be reused.

## Determining whether a cached token file is still valid

Three checks, in order of cost — cheapest first, only running the next one
if the previous one passes:

1. **Do `tokens.bin` and its metadata sidecar both exist?** If either is
   missing, there's no cache to validate — this is the "no cache" case,
   not an error.
2. **Does the metadata's recorded `corpus_hash` match a hash of the
   corpus text this run just read, and does its `tokenizer_hash` match a
   hash of the tokenizer this run is about to encode with?** Both are
   cheap `sha256` digests over data already in memory (the corpus string
   is already fully loaded to build it in the first place; the tokenizer's
   serialized form is small — merges + vocab, not the corpus). A mismatch
   in either means "stale" — the corpus changed, or the tokenizer changed
   (different training run, different `vocab_size`, or a genuinely
   different corpus that happened to train the same-shaped tokenizer by
   coincidence — the tokenizer hash alone can't distinguish that, which is
   why corpus and tokenizer are both checked, not one or the other).
3. **Does the metadata's recorded `tokens_hash` match a hash of the
   `tokens.bin` file currently on disk?** This one is not about the
   *inputs* going stale — it catches `tokens.bin` itself having been
   overwritten by something else since the metadata was last written. This
   matters concretely in this project: `scripts/finetune.py` writes its
   own, unrelated token stream (instruction/response examples, not the raw
   pretraining corpus) to the *same path*,
   `data/processed/tokens.bin` (see "Known interaction with
   `scripts/finetune.py`" below), without updating the pretraining cache's
   metadata. Without this check, a corpus/tokenizer hash match alone could
   claim a stale-by-a-different-mechanism cache is valid, even though the
   bytes on disk are no longer the pretraining corpus's tokens at all.

Any failure at any step — missing files, an unreadable/corrupt metadata
file, a JSON parse error, a hash mismatch — is treated identically: **the
cache is not used, and the corpus is encoded fresh, exactly as happens
today.** This is a deliberately different policy than the "fail loudly,
refuse to proceed" precedent `docs/01-tokenization.md` and
`docs/checkpoint-resume.md` set for a missing/mismatched tokenizer or
checkpoint. Those refuse because there is no safe fallback — reconstructing
a lost tokenizer or optimizer state from nothing risks silently training on
the wrong meaning. A token cache has no such risk: it is a pure derived
artifact of the corpus and tokenizer, both of which are still fully
available every time. Rebuilding it is always correct, just slower. So a
broken cache degrades to "as slow as today," never to "wrong," and never
needs to stop the run.

## What metadata is stored, and why

A small JSON sidecar, `data/processed/tokens.meta.json`, next to
`tokens.bin`:

```json
{
  "corpus_hash": "3f9a1c...",
  "tokenizer_hash": "8b2e77...",
  "tokens_hash": "d4a610...",
  "num_tokens": 1048576
}
```

- `corpus_hash` — `sha256` of the exact corpus string
  (`"\n".join(f.read_text() for f in text_files)`) that produced this
  cache. Hashing full content, not file names/sizes/mtimes, is what makes
  "the corpus changed" mean *any* change to what will actually be
  encoded — content edited in place, a file added or removed, or the glob
  order changing — with nothing to accidentally miss.
- `tokenizer_hash` — `sha256` of the tokenizer's own serialized form
  (the same `merges`/`vocab`/`special_tokens`/`pretokenize` data
  `BPETokenizer.save()` writes, hashed rather than written twice).
  Together with `corpus_hash`, this is what "tokenizer identity" and
  "tokenization settings" (`vocab_size` target, pre-tokenization mode)
  reduce to — both are already fully captured by what merges the
  tokenizer actually learned, so there's nothing separate to track.
- `tokens_hash` — `sha256` of `tokens.bin` itself, as it sits on disk right
  now. Not part of deciding "should I re-encode" (that's `corpus_hash` +
  `tokenizer_hash`) — it's an integrity check on the file this metadata
  claims to describe, catching the case above where something else wrote
  different bytes to the same path.
- `num_tokens` — the token count, for a cheap sanity print/log and a first,
  even-cheaper-than-hashing sanity check (file size ==
  `num_tokens * 2` bytes, since tokens are `uint16`) before bothering with
  the full `tokens_hash` scan.

No tokenizer `vocab_size` field, no separate "pretokenize mode" field, no
corpus file list — all of that is already implied by `corpus_hash` and
`tokenizer_hash` matching, so storing it again would just be a second,
redundant way for the metadata to be self-inconsistent.

## Desired behavior by scenario

| Scenario | Behavior |
|---|---|
| Fresh run, no cache files present | Encode corpus, write `tokens.bin` + `tokens.meta.json`. Same as today, plus the metadata write. |
| Fresh run, valid cache (same corpus, same tokenizer) | Skip `encode()` entirely; `load_token_ids` the existing `tokens.bin`. Only possible because a fresh run's newly-*trained* tokenizer is deterministic for unchanged corpus + `vocab_size` — see below. |
| Corpus changed since cache was written | `corpus_hash` mismatch → treat as no cache → re-encode, overwrite `tokens.bin` + metadata. Never reuse. |
| Tokenizer changed (different `vocab_size`, retrained on different data, or a genuinely different tokenizer object) | `tokenizer_hash` mismatch → same as above → re-encode, overwrite. |
| Cache metadata missing, or `tokens.bin` missing, or metadata unreadable/corrupt | Treated as "no cache" (see policy above) → re-encode, (re)write both files. No error, no crash. |
| `--resume` | Loads the checkpoint's persisted tokenizer (unchanged, per `docs/resume-tokenizer-consistency.md`), then checks the cache the same way a fresh run does. In the ordinary case — corpus and tokenizer both unchanged since the run that produced this checkpoint — the cache is valid, and `encode()` is skipped entirely: resume goes straight to loading `tokens.bin` and continuing training. If the corpus *was* edited between crash and resume, `corpus_hash` mismatches and resume correctly falls back to re-encoding with the checkpoint's tokenizer (still the right tokenizer — just re-run against new text), rather than silently training on old ids. |

Fresh (non-`--resume`) runs still retrain a brand-new `BPETokenizer` from
the current corpus, exactly as today — this milestone does not change
*when* a tokenizer gets trained, only whether `encode()` and the
`tokens.bin` write get skipped once that tokenizer (freshly trained or
loaded from a checkpoint) turns out to match what's already cached. BPE
training is deterministic given the same corpus text and `vocab_size`
(`docs/01-tokenization.md`), so a repeated fresh run on unchanged data
retrains an identical tokenizer and its hash matches the cache — the
saving shows up on the `encode()` step, which is the part that scales with
full corpus size on every call, not on tokenizer training (already
addressed separately by `docs/tokenizer-performance.md`).

## Worked example

1. `scripts/train.py --config configs/small.yaml` runs for the first time.
   `data/raw/story.txt` contains `"the cat sat. the cat ran."`. BPE trains
   a tokenizer, `tokenizer.encode(corpus)` produces
   `[116, 104, 256, ...]` (some real id sequence). `write_token_ids` writes
   `data/processed/tokens.bin`. The cache writer computes
   `corpus_hash = sha256("the cat sat. the cat ran.")`,
   `tokenizer_hash = sha256(<tokenizer's serialized merges+vocab>)`,
   `tokens_hash = sha256(<tokens.bin's bytes>)`, `num_tokens = 12`, and
   writes `data/processed/tokens.meta.json`.
2. Training crashes at step 4000 of 5000 (the checkpoint/resume scenario
   `docs/checkpoint-resume.md` already covers). `data/raw/story.txt` is
   untouched.
3. `scripts/train.py --config configs/small.yaml --resume` runs.
   `load_tokenizer_for_checkpoint` loads the exact tokenizer from step 1
   back from `tokenizer.json`. The script reads `story.txt` again (still
   needed to compute `corpus_hash` and as a fallback source if the cache
   *is* stale) and hashes it: identical to step 1's `corpus_hash`. It
   hashes the loaded tokenizer: identical to step 1's `tokenizer_hash`.
   Both match `tokens.meta.json`. `tokens_hash` is checked against the
   `tokens.bin` currently on disk: still matches (nothing else wrote to
   it). The cache is valid — `encode()` is skipped, `tokens.bin` is
   memory-mapped directly, and training resumes from step 4001 in
   seconds, not after another full corpus encode.
4. Someone edits `story.txt`, adding a new sentence, then runs
   `scripts/train.py --config configs/small.yaml --resume` again.
   `corpus_hash` of the new file content no longer matches
   `tokens.meta.json`'s recorded value. The cache is rejected;
   `tokenizer.encode()` runs again (using the checkpoint's *unchanged*
   tokenizer — still correct per `docs/resume-tokenizer-consistency.md`,
   just against new text), and `tokens.bin` + `tokens.meta.json` are
   overwritten with the new result.

## Where the cache lives, and why

`data/processed/`, next to `tokens.bin` itself — not
`checkpoint_dir` (next to `tokenizer.json`/`latest.pt`).

Reasoning: the cache's validity is entirely determined by the
`corpus_hash`/`tokenizer_hash` pair recorded in its own metadata, not by
which checkpoint directory happens to be asking. Multiple checkpoint
directories (different hyperparameter runs, or `scripts/evaluate.py`
invocations against different checkpoints) can share one raw corpus and
one tokenizer state, and today they already share one `tokens.bin` at
`data/processed/` — `scripts/train.py` and `scripts/evaluate.py` both
write to that same path. Keeping the cache there means every one of those
runs can hit the same cache instead of each checkpoint directory carrying
its own redundant copy of what is, when the hash matches, identical data.
Putting it under `checkpoint_dir` instead would mean a fresh checkpoint
directory always starts cache-cold even when an identical corpus +
tokenizer combination was already encoded for some *other* checkpoint
minutes earlier — solving the resume case only, not the "re-run
`evaluate.py` a few times while iterating" case that motivated this
milestone just as much.

## Known interaction with `scripts/finetune.py`

`scripts/finetune.py` already writes its own token stream — encoded
instruction/response examples, via `build_masked_token_ids`, not the raw
pretraining corpus — to the *same path*, `data/processed/tokens.bin`
(plus `loss_mask.bin`, unrelated to this milestone). This is pre-existing
behavior, not something this milestone introduces or is in scope to fix.
Without the `tokens_hash` integrity check described above, this would be a
real hazard for a cache: `scripts/train.py`, run again after a
`scripts/finetune.py` run, would see a `corpus_hash`/`tokenizer_hash` match
against its own stale `tokens.meta.json` (finetune doesn't touch that
file) while the actual `tokens.bin` bytes are finetune's, not the
pretraining corpus's. The `tokens_hash` check exists specifically to catch
this: it detects that `tokens.bin` no longer contains what the metadata
claims, and falls back to re-encoding correctly. `scripts/finetune.py`
itself is not modified by this milestone.

## What stays unchanged

- `TokenDataset` — untouched. It only ever reads whatever `load_token_ids`
  hands it; caching happens entirely before that point.
- Train/validation splitting (`train_val_split`) — untouched, still called
  on whatever token array is loaded, cached or freshly encoded.
- Tokenizer persistence format (`tokenizer.json`) — untouched. The cache
  hashes a tokenizer's serialized form; it doesn't change what gets
  written for the tokenizer itself.
- EOS / special-token behavior — untouched. `add_eos_token()` runs before
  the tokenizer is hashed (fresh runs) or was already baked into the
  loaded tokenizer (resume), so EOS is part of what `tokenizer_hash`
  already captures, with no separate handling needed.
- No database, no cache-eviction policy, no multi-entry cache — one
  `tokens.bin` + one `tokens.meta.json`, overwritten in place, exactly
  mirroring `docs/checkpoint-resume.md`'s "one `latest.pt`, not a growing
  pile of files" precedent.

## What this milestone does *not* do

No change to model architecture, optimizer behavior, generation, or
fine-tuning behavior. No change to `scripts/finetune.py`'s own
`tokens.bin`/`loss_mask.bin` write path (see "Known interaction" above —
flagged, not fixed, here). No caching of the fine-tuning token stream. No
new CLI flag — caching is always-on and self-invalidating, the same way
tokenizer persistence and checkpoint/resume are always-on with no opt-out
flag.

## Status: implemented and tested

`src/llm_from_scratch/data/cache.py` adds `CacheMetadata` (`corpus_hash`,
`tokenizer_hash`, `tokens_hash`, `num_tokens`) and
`ensure_token_cache(corpus, tokenizer, tokens_path, meta_path) -> bool`,
matching the design above exactly: hashes the corpus (`sha256` of the
corpus string) and the tokenizer (`sha256` of the exact bytes
`BPETokenizer.save()` would write — computed by actually calling `save()`
to a temp file and hashing that, rather than duplicating the
merges/vocab/special_tokens/pretokenize dict-building logic a second time,
so this can never silently drift out of sync with the real persistence
format); checks a fast file-size pre-check
(`tokens.bin`'s size == `num_tokens * 2` bytes, since tokens are `uint16`)
before the full `tokens_hash` scan; on any mismatch or any problem loading
the metadata (missing file, corrupt JSON, missing keys), re-encodes and
overwrites both files — never raises, matching the "always safely
rebuildable" policy above. Returns `True` when a valid cache was reused
(no `encode()` call happened) and `False` when it (re-)encoded, so callers
can log which happened.

`scripts/train.py` and `scripts/evaluate.py` both call
`ensure_token_cache` in place of their old unconditional
`tokenizer.encode(corpus)` + `write_token_ids(...)` — for both fresh and
`--resume` runs alike, exactly as designed. `scripts/train.py` also prints
`"Reusing cached tokens.bin"` or `"Encoded N characters"` so a run's
console output makes which path was taken visible. `scripts/finetune.py`
is untouched (see "Known interaction" above — still flagged, not fixed,
here).

Tests added in `tests/test_cache.py` (10 new): first-run cache creation,
valid-cache reuse (`encode()` monkeypatched to raise if called, proving it
genuinely isn't invoked on a hit), unchanged corpus+tokenizer producing
identical token ids across two calls, a changed corpus invalidating the
cache, a changed tokenizer (different `vocab_size`, hence different
merges) invalidating the cache, missing metadata / corrupt metadata /
missing `tokens.bin` all rebuilding cleanly with no exception raised, and
a dedicated test for the `scripts/finetune.py` collision scenario —
`tokens.bin` overwritten by an unrelated token stream with the metadata
left untouched is detected via the `tokens_hash` check and correctly
re-encoded rather than silently trusted. `tests/test_train_resume_cache.py`
(2 new) tests `scripts/train.py` end to end: `--resume` with an unchanged
corpus reuses the cache with `BPETokenizer.encode` monkeypatched to raise
if called (proving resume genuinely skips encoding, not just that the
output happens to match), and `--resume` with a changed corpus produces a
different `tokens.bin` (proving the stale-cache case is never silently
served). Full suite: `146 passed` (previous 135 + 11 net new: 10 in
`test_cache.py` + 2 in `test_train_resume_cache.py`, minus 1 — no existing
tests were removed, this is exactly 12 new test functions across two new
files).

Manual end-to-end smoke test (not just unit tests): on a real
`scripts/train.py --config ... --device cpu` run against a ~17,600-
character synthetic corpus — (1) first run printed `"Encoded 17,600
characters"` and created `tokens.bin` + `tokens.meta.json`; (2) a second
fresh (non-`--resume`) run against the unchanged corpus printed `"Reusing
cached tokens.bin"`, confirming repeated fresh runs benefit too, not just
resume; (3) `--resume` (raised `max_steps`) printed `"Reusing cached
tokens.bin"` and `"Resuming from step 20/40"`, confirming resume skips
encoding entirely on an unchanged cache; (4) appending new text to the
raw corpus then running `--resume` again printed `"Encoded 19,850
characters"` (not a silent reuse), and `tokens.bin`'s content hash
(`md5sum`) changed, confirming the stale cache was correctly rejected and
rebuilt with the checkpoint's original (unchanged) tokenizer against the
new text. Separately ran `scripts/evaluate.py` against the same
checkpoint/config right after a `train.py` run — it reused the identical
cache `train.py` had just written (`tokens.bin` byte-for-byte unchanged
before/after `evaluate.py`), confirming the two scripts share one cache
as designed.
