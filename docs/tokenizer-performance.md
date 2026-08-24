# Tokenizer performance — regex pre-tokenization + efficient BPE

Not a new stage, and not a new tokenizer — this is a performance and
scale-correctness milestone on top of Stage 1 (`docs/01-tokenization.md`),
the same way device support or checkpoint/resume extended earlier stages
without becoming a new numbered one.

## Why the current tokenizer only worked comfortably on toy corpora

Every stage of this project has been smoke-tested on tiny synthetic
corpora — hundreds of characters, occasionally a few KB. `CLAUDE.md`'s own
stated default scale is "a modest public text corpus (a few hundred MB)."
Nobody has actually pointed `scripts/train.py` at anything close to that
size yet. When you do, `BPETokenizer.train()` and `.encode()`
(`src/llm_from_scratch/tokenizer/bpe.py`) become the bottleneck, not the
model or the training loop — training a tokenizer on real-sized text would
take somewhere between hours and days, not minutes, in pure Python.

## Where the current performance cost comes from

`train()`'s main loop, once per merge:

```python
for i in range(num_merges):
    pair_counts = self._count_pairs(ids)   # scans the WHOLE token sequence
    best_pair = max(pair_counts, key=pair_counts.get)
    ids = self._merge(ids, best_pair, new_id)  # scans the WHOLE token sequence again
```

`ids` starts as one Python list holding every byte of the entire corpus.
`_count_pairs` walks that whole list to build a fresh `Counter` of every
adjacent pair, every single merge. `_merge` then walks the whole list
again to apply the winning merge. Neither step remembers anything from the
previous merge — every merge starts from zero information about what
changed last time.

`encode()` has the same shape, just scoped to one call's input text
instead of the whole corpus: repeatedly re-scan the current token list for
pair frequencies, apply the best learned merge, repeat.

## Why repeatedly rescanning the whole sequence gets expensive fast

Let `N` be the corpus length (in bytes/tokens) and `M` be the number of
merges (`vocab_size - 256`). Each merge costs `O(N)` — one counting pass,
one merge pass. `M` merges means `O(M × N)` total work for training.

That product is the problem. `N` grows with corpus size — "a few hundred
MB" is on the order of 10⁸ characters. `M` grows with vocabulary size — a
real vocabulary is thousands of merges, not `280 - 256 = 24` like this
project's test fixtures use. `M × N` for realistic numbers
(`M ≈ 10,000`, `N ≈ 10⁸`) is roughly `10¹²` operations — computationally
infeasible in pure Python regardless of how tightly the loop is written.
This isn't a constant-factor problem you can fix by writing faster Python;
it's the *shape* of the algorithm (linear rescan × number of merges) that
has to change.

There's a second, independent reason full rescans are wasteful, not just
slow: natural text repeats itself enormously. The word "the" might appear
50,000 times in a corpus. Scanning the raw byte stream re-derives "the"'s
pair statistics 50,000 separate times, even though every occurrence
contributes the *identical* pair information. That redundant work is pure
waste — it doesn't buy the algorithm anything it didn't already know from
the first occurrence.

## What regex pre-tokenization is, and why modern BPE tokenizers use it

**Pre-tokenization** means splitting the raw text into rough chunks
*before* BPE ever runs — typically on whitespace and punctuation
boundaries, using a regex — and then running BPE merges independently
*inside* each chunk. GPT-2's tokenizer (and its descendants, including
`tiktoken`) does exactly this: a regex splits text into pieces like
`" the"`, `","`, `" running"` (the leading space usually stays attached to
the following word — that's deliberate, not a bug, since " the" and "the"
genuinely behave differently as tokens in real text), and merges are
learned and applied only within each piece.

Two reasons this matters:

1. **Merge quality.** Without pre-tokenization, nothing stops BPE from
   merging across what a person would consider a word or punctuation
   boundary — e.g. the trailing letter of one word merging with the
   leading space of the next, purely because that byte pair happened to
   be frequent. Those merges are usually not meaningful units; they're
   just frequency-driven noise. Pre-tokenization bounds merging to chunks
   that already respect natural boundaries, so the vocabulary spends its
   budget on real sub-word pieces instead of boundary artifacts.
2. **Performance** (the point of this milestone). Chunking makes the
   "count once, reuse many times" trick above actually work: instead of
   treating the corpus as one giant sequence, pre-tokenization naturally
   groups it into a large number of *repeated, identical, short* chunks
   (every occurrence of "the" is the exact same chunk). That structure is
   what an efficient implementation exploits — see below.

## How pre-tokenization limits merges to reasonable chunks

Concretely: split the corpus with a regex into chunks (roughly: one chunk
per word, one per run of punctuation, one per run of whitespace). Encode
each chunk to bytes independently. BPE's counting and merging then only
ever look *within* a chunk — a candidate pair from the end of one chunk
and the start of the next is never considered, because they're never
adjacent in the same list. This is a hard constraint, not a preference:
learned merges can never cross a chunk boundary, by construction.

## How an efficient implementation avoids unnecessary full-corpus rescans

The actual algorithmic fix — the reason this milestone changes complexity,
not just adds regex splitting — is to stop treating the corpus as one flat
sequence at all:

1. Pre-tokenize the corpus into chunks (the regex above).
2. Count **unique** chunks and how many times each occurs — a
   `dict[chunk_text, count]` — instead of keeping every occurrence
   separately. "the" appearing 50,000 times becomes one entry with
   count 50,000, not 50,000 separate list entries.
3. Represent each *unique* chunk as its own small list of current token
   ids (starting as bytes). Pair counts are computed once per unique
   chunk and multiplied by that chunk's corpus-wide occurrence count —
   so a pair inside "the" is counted with weight 50,000 from a single
   O(chunk length) scan, not 50,000 separate scans.
4. When a pair is merged, only the (typically small number of) unique
   chunks that actually contain that pair need to be touched — update
   just their token lists and just the pair counts that changed as a
   result (the pair that was consumed, and the one or two new pairs
   created at the merge point). Chunks that never contained the merged
   pair are untouched and don't need to be revisited.

This turns training's cost from "rescan everything, every merge" into
"do a bounded amount of work proportional to how much actually changed,
every merge." The total cost across all of training becomes roughly
proportional to the total size of the *unique* vocabulary of chunks in
the corpus (which is bounded and comparatively small for natural language,
regardless of how large the raw corpus is) plus a bounded amount of
touch-up work per merge — not `O(M × N)` against the raw corpus size.

`encode()` gets the same chunk-scoping: split the input text into chunks
with the same regex, apply learned merges within each chunk independently
(same lowest-learned-merge-first logic it already uses), then concatenate
ids across chunks. This bounds each merge-search to one chunk's length
instead of the whole input's length, which matters most for encoding long
documents.

## The intended improvement — and what this isn't

This is not going to be `tiktoken`. `tiktoken` is implemented in Rust,
uses a specialized fast-path merge data structure (a priority queue over
pairs plus a doubly-linked-list representation for O(log N) merge
updates), and ships a fixed, pre-trained vocabulary rather than training
one per run. None of that is in scope here — the point of this project is
to see BPE working, not to reimplement a production tokenizer.

What *is* in scope: change the algorithm's shape from "rescan the entire
corpus on every merge" to "do bounded work proportional to what a merge
actually touches, exploiting natural-language repetition." That's the
difference between "physically cannot finish on a few hundred MB" and
"finishes in a reasonable amount of time on a single machine" — which is
the actual bar this milestone needs to clear, not tiktoken's bar.

No NumPy, no vectorization is introduced for its own sake. The data here
(chunks, pairs, counts) is exactly the kind of small-object, dictionary-
and-list-shaped data Python's built-ins are already good at; the win comes
from *not doing redundant work*, not from a faster inner loop over the
same redundant work.

## Worked example: pre-tokenized chunks, BPE operating inside them

Corpus: `"the cat sat. the cat ran."`

**Pre-tokenization** (rough regex: split on runs of letters, runs of
punctuation, and whitespace, keeping a leading space attached to the
following word) gives chunks:

```
"the", " cat", " sat", ".", " the", " cat", " ran", "."
```

**Count unique chunks:**

```
"the":   1
" cat":  2
" sat":  1
".":     2
" the":  1
" ran":  1
```

(`"the"` at the very start and `" the"` later are different chunks — one
has no leading space, since it's the first word in the corpus, the other
does. That's expected; it's exactly why real tokenizers often end up with
both a `"the"` token and a `" the"` token.)

**BPE now runs per unique chunk**, weighted by count. Take `" cat"`
(count 2, bytes `[0x20, c, a, t]`): its adjacent pairs
`(space,c)`, `(c,a)`, `(a,t)` each get +2 toward the corpus-wide pair
counts, from a single scan of this one 4-byte chunk — not two separate
scans. If `(c,a)` turns out to be globally the most frequent pair across
all chunks, it merges into a new token `ca` **only inside chunks that
contain it** — here, just `" cat"` (both its occurrences, via the count),
leaving `"the"`, `" sat"`, `"."`, `" the"`, `" ran"` completely untouched
and unrescanned. `" cat"` becomes `[space, "ca", t]`; pair counts are
updated by removing `(c,a)` and `(a,t)`, adding `(space,"ca")` and
`("ca",t)` — a handful of updates, not a full recount.

Notice what pre-tokenization already prevented: nothing ever proposed
merging the `.` at the end of `"sat."`'s chunk with the space that starts
the next chunk, `" the"` — they were never adjacent in the same list to
begin with.

## The tradeoff: this changes tokenizer behavior, not just speed

Pre-tokenization is not a pure performance change — it changes *which*
merges get learned, because merges can no longer cross chunk boundaries.
A tokenizer trained with pre-tokenization on the same corpus, at the same
`vocab_size`, will generally learn a **different** set of merges (and
therefore different token ids for different strings) than the current
global-byte-stream tokenizer does today. This is expected and, per the
reasoning above, usually an improvement in merge quality — but it is a
real behavior change, not a no-op speedup.

## Backward compatibility: existing tokenizer files stay loadable

This is a hard constraint for this milestone: **a `tokenizer.json`
produced by today's code must still load and must still `encode()`/
`decode()` exactly as it did before** — no silently different token ids
for an existing saved tokenizer. Two facts make this achievable without a
breaking format change:

- The on-disk *shape* of `tokenizer.json` (`merges`, `vocab`,
  `special_tokens`) doesn't need to change. Pre-tokenization changes how
  merges get *learned* and *applied during encode()*, not the data
  structures that represent them once learned.
- What does need one small, additive piece of metadata: whether a given
  tokenizer's merges were learned with chunk boundaries in effect or not.
  A tokenizer trained by today's code learned some merges under
  unrestricted, whole-corpus adjacency — including, potentially, pairs
  that happened to span what a chunk boundary would now be. Encoding new
  text with a *chunk-scoped* encode() against *that* tokenizer's merges
  could fail to find/apply a merge it should apply, changing its output.
  So `tokenizer.json` gains one new, optional field recording which mode
  produced this tokenizer. A file with the field absent (every tokenizer
  saved before this milestone) is treated as "pre-tokenization off" and
  uses the exact current global algorithm at both train and encode time —
  byte-for-byte the same behavior as today. A tokenizer trained by the
  new code sets the field, and uses the new chunk-scoped algorithm. Old
  files keep working exactly as before; only newly-trained tokenizers
  opt into the new behavior. `vocab`/`merges` ids for an old file are
  never recomputed or reassigned on load — `load()` still just reads them
  back verbatim, as it does today.
- Everything downstream of the tokenizer — EOS handling
  (`add_eos_token`/`eos_token_id`, `docs/eos-generation-stopping.md`),
  `save()`/`load()`, tokenizer persistence
  (`docs/01-tokenization.md`) — is unaffected: EOS is metadata layered on
  top of whatever vocabulary training produced, regardless of which
  algorithm produced it.

## What this milestone does *not* do

No change to model architecture, training loop, evaluation, or
generation — this is entirely inside `BPETokenizer`. No new tokenizer
format version, no re-encoding or migration of existing `tokenizer.json`
files. No deletion of `build_corpus()` or any other unrelated cleanup
identified in the prior audit — those are separate, later milestones. No
NumPy/vectorization introduced without a specific reason tied to the
complexity argument above. No claim of `tiktoken`-level performance — see
"The intended improvement" above.

## Status: implemented and tested

`src/llm_from_scratch/tokenizer/bpe.py` adds `_PRETOKENIZE_PATTERN` (the
GPT-2-style regex, using the third-party `regex` package for `\p{L}`/
`\p{N}` Unicode-category support) and `BPETokenizer._pretokenize(text)`.
`train()` now: pre-tokenizes the corpus into chunks, counts unique chunks
weighted by occurrence (`Counter`), and tracks `pair_counts` (`Pair ->
weighted count`) alongside `pair_chunks` (`Pair -> set of unique-chunk keys
containing it`). Each merge only touches the chunks in `pair_chunks[best_pair]`
— removing their old pair contributions, applying the merge to just that
chunk's short token list, and re-adding new pair contributions — never
rescanning chunks the merge didn't touch. `train()` always sets
`self.pretokenize = True`. `encode()` now checks `self.pretokenize`: when
`True`, it pre-tokenizes the input the same way and applies learned merges
within each chunk independently (`_apply_merges`, extracted from the
original loop, unchanged in logic); when `False`, it runs the exact
original whole-text algorithm. `save()`/`load()` gained a `pretokenize`
field; `load()` defaults it to `False` when the key is absent (every
`tokenizer.json` saved before this milestone), so a legacy file's
`encode()`/`decode()` behavior is unchanged, byte-for-byte, from before
this milestone. No changes to `decode()`, `add_eos_token()`,
`eos_token_id`, or the on-disk `merges`/`vocab`/`special_tokens` shape.

Tests added in `tests/test_tokenizer.py` (9 new):
`test_pretokenize_splits_into_word_and_punctuation_chunks`,
`test_pretokenize_handles_empty_string`,
`test_newly_trained_tokenizer_is_pretokenized_by_default`,
`test_merges_do_not_cross_chunk_boundaries` (trains on `"cat cat cat cat
cat"`, asserts the pair spanning `'t'` + the following chunk's leading
`' '` is structurally absent from `merges`, regardless of vocab_size or
tie-breaking), `test_correctness_against_known_small_example` (hand-
verified: `"aaaa"` with one merge allowed produces exactly `{(a,a): 256}`
and `encode("aaaa") == [256, 256]`),
`test_save_load_preserves_pretokenize_flag`,
`test_pretokenize_flag_controls_whether_boundary_merge_applies` (same
merges/vocab, differing only in the `pretokenize` flag, demonstrating a
boundary-crossing merge applies under legacy/global mode and not under
pre-tokenized mode), `test_legacy_tokenizer_without_pretokenize_key_still_loads_and_works`
(hand-edits a saved file to remove the `pretokenize` key entirely,
confirms `load()` defaults to `False` and encode/decode round-trips
correctly), and `test_efficient_training_is_meaningfully_faster_than_naive_rescanning`
(times the new algorithm against a from-scratch reference re-implementation
of the exact pre-milestone whole-corpus-rescan algorithm on a ~28,000-
character repetitive corpus; asserts the new algorithm is at least 3x
faster and completes in under 5 seconds). Two pre-existing tests
(`test_train_produces_merges_and_vocab`, `test_add_eos_token_assigns_next_unused_id`)
were adjusted to assert internal consistency (`vocab_size == 256 +
len(merges)`, EOS id equals whatever `vocab_size` training actually
produced) instead of a hardcoded merge count — pre-tokenization bounds
merges to within chunks, so a tiny multi-word test corpus that previously
relied on cross-word merges no longer reaches the same exact count (17
merges instead of 24, at `vocab_size=280`), which is the documented
tradeoff, not a bug. Full suite: `99 passed` (previous 90 + 9 new).

Manual end-to-end verification (not just unit tests): trained on a 5,000,037-
character synthetic corpus (repeated random sentences from a 30-word
vocabulary) — `train()` completed in 0.73s, producing 156 merges
(vocab_size target 2000; pre-tokenization capped actual merges to what the
corpus's limited unique-word vocabulary supports, as expected).
Encode/decode round-tripped correctly, and a save+load round-trip produced
identical `encode()` output. Repeated at 50,000,007 characters (50 MB, a
meaningful fraction of the project's stated "a few hundred MB" target
scale, using a 40-word vocabulary): `train()` completed in 6.13s (210
merges), and `encode()` of a 1,000,000-character slice completed in 4.23s.
For comparison, the old whole-corpus-rescan algorithm's complexity
(`O(num_merges × corpus_length)`) would put a 50,000,000-character corpus
at roughly `1744 × 50,000,000 ≈ 87 billion` operations — not attempted
here, since it would not complete in any practical amount of time; the
9th unit test above demonstrates the same algorithmic gap directly, at a
scale (~28,000 characters) small enough for both algorithms to actually
finish inside the test suite's runtime.
