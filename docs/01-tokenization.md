# 01 — Tokenization

## What it is
Models don't read text. They read numbers. Tokenization is the step that
chops text into small chunks ("tokens" — often sub-word pieces, not full
words) and maps each chunk to an integer ID via a fixed vocabulary. Those
integers are what actually gets fed into the model.

## Why it matters
The vocabulary size and splitting strategy shape everything downstream:
model input size, how well it handles rare or made-up words, and how much
text fits in a given context window. Get this wrong and nothing built on
top of it works well, no matter how good the model architecture is.

## The one key idea
A tokenizer is a lossless(-ish) compression scheme with a fixed dictionary.
"Byte Pair Encoding" (BPE) builds that dictionary by starting from
individual bytes and repeatedly merging the most frequent adjacent pair
into a new token — so common words collapse into single tokens, and rare
words fall back to smaller, still-recognizable pieces. Nothing is ever
"unknown" — worst case, a word just splits down to raw bytes.

## Why not just split on words or characters?

**Word-level** ("split on spaces"): vocabulary explodes (every inflection —
`run`, `runs`, `running` — is a separate entry), and any word not seen
during training becomes an unrepresentable "unknown" token. Bad for rare
words, typos, code, or other languages.

**Character-level**: vocabulary stays tiny, and nothing is ever unknown —
but sequences get very long (one token per character), so the model has to
work much harder to relate distant characters into meaning. A 10-word
sentence might be 60 tokens.

BPE sits between these: common whole words become one token, rare words
degrade gracefully into a handful of sub-word pieces instead of exploding
the vocabulary or blowing up sequence length.

## How BPE training works, step by step

1. **Start at the byte level.** Every input string starts as its raw bytes
   (0–255). This is the base vocabulary — 256 tokens, and it can represent
   *any* text in any language, since all text is bytes underneath.
2. **Count adjacent pairs.** Look at every pair of adjacent tokens across
   the whole training corpus and count how often each pair occurs.
3. **Merge the most frequent pair.** Take the single most common pair,
   e.g. `(t, h)`, and mint it as a new token, `th`. Add it to the
   vocabulary. Everywhere `t` is immediately followed by `h` in the corpus,
   replace the two tokens with the one new merged token.
4. **Repeat.** Re-count pairs (now `th` can pair with other things, e.g.
   `(th, e)` → `the`), merge the new most-frequent pair, and repeat for a
   fixed number of merges. Each merge grows the vocabulary by exactly one
   token and shrinks the encoded corpus a little.
5. **Stop when you hit a target vocabulary size.** 256 base bytes + N
   merges = vocab size. This project's config (`configs/small.yaml`) sets
   the target size — more merges means fewer, longer tokens per word, and
   a bigger lookup table.

The output of training is just two things: the ordered list of merges
(which pair got merged, in what order) and the resulting vocabulary
(token → integer ID). That's the whole "model" — no neural net involved
yet.

## Worked example

Say the entire training corpus is the string `"low lower lowest"`. Start
byte-level (using characters here for readability — real bytes work
identically):

```
l o w _ l o w e r _ l o w e s t
```

Count adjacent pairs: `(l,o)` appears 3 times — the most frequent. Merge
it into `lo`:

```
lo w _ lo w e r _ lo w e s t
```

Now `(lo,w)` is most frequent (3 times). Merge into `low`:

```
low _ low e r _ low e s t
```

Now `(low,_)`? No — `_` isn't shared enough to matter yet; next most
frequent adjacent pair might be `(e,r)` or similar, depending on tie-
breaking. The process just keeps going: merge the most frequent pair,
one at a time, until the vocabulary hits its target size. Notice `low`
became a single token after just two merges, because it was common —
that's the whole point.

## Encoding and decoding

- **Encode** (text → token IDs): apply the learned merges to new text, in
  the same order they were learned. Start from bytes, merge whichever
  learned pair appears, repeat until no more learned merges apply. Then
  look up each resulting token's ID.
- **Decode** (token IDs → text): reverse lookup each ID back to its token
  string (or bytes), concatenate. Because we started from bytes and never
  discarded anything, this round-trips exactly — decode(encode(x)) == x.

## Simplification note

Real GPT-style tokenizers (e.g. GPT-2/GPT-4's `tiktoken`) add pre-tokenization
rules — regex splitting on whitespace/punctuation before BPE runs — so
merges never cross word boundaries in weird ways, and they use a fixed
vocab trained once on a huge corpus, shipped with the model. We're
skipping the regex pre-splitting step for now and training on our own
(small) corpus, to keep the core merge algorithm visible. We may add
pre-tokenization later if merge quality suffers.

## What we build here

A small BPE tokenizer trained on our own corpus (not reusing GPT's), so
you see every step: byte-level start, merge-frequency counting, and the
encode/decode round trip. Surface: `.train(corpus)`, `.encode(text)`,
`.decode(ids)`.

## Tokenizer persistence

### Why a model must keep the exact tokenizer it was trained with

A trained model doesn't actually know words, or bytes, or characters. It
only knows *integers* — during pretraining, every weight in the token
embedding table (`GPT.token_embedding`, Stage 3) is nudged into place
under the assumption "row 4171 of this table represents whatever string
token id 4171 decodes to." That assumption is set once, at the start of
training, by whichever tokenizer produced the training data. If you ever
feed that same model a *different* tokenizer — one that assigns id 4171 to
a different string — the model doesn't notice or error out. It just keeps
using the embedding it learned for the old meaning of 4171, applied to a
now-wrong new meaning. The weights are still numbers; nothing crashes.
Only the output quietly stops making sense.

So "the tokenizer" isn't a disposable preprocessing detail — it's part of
the trained artifact, exactly as much as the weights are. A checkpoint and
the tokenizer that produced its training data are one unit. Split them
apart, and the checkpoint's weights become uninterpretable.

### Why retraining BPE on a different corpus silently scrambles token IDs

`BPETokenizer.train()` (this file) learns merges by counting adjacent byte
pairs *in whatever corpus you give it* and merging the most frequent pair
first, repeating until `vocab_size` is reached. The merges it learns — and
therefore which string ends up at which integer id — depend entirely on
the frequency statistics of that specific corpus. Train it on Shakespeare
and id 300 might be `"the "`. Train it on the same *size* vocabulary but a
different corpus — say, a folder of 30 instruction/response examples — and
id 300 might land on `"tion"`, or `"Inst"`, or nothing resembling the
first corpus's token 300 at all. `vocab_size` matching across two runs
guarantees nothing about the *ids matching* — it only guarantees the two
vocabularies are the same *size*.

This isn't a hypothetical risk in this project — it's the actual current
behavior of `scripts/finetune.py`, which retrains a fresh `BPETokenizer`
on the fine-tuning corpus, not the corpus the checkpoint was pretrained
on. The pretrained embedding table's row 4171 still means whatever it
meant during pretraining; the fine-tuning tokenizer's id 4171 almost
certainly means something else entirely. Fine-tuning proceeds without any
error — loss goes down, a checkpoint gets saved — but the model is being
trained on inputs that don't mean what its own embedding table thinks they
mean. `scripts/evaluate.py` and `scripts/generate.py` happen to retrain on
the *same* raw corpus the checkpoint was pretrained on, which — because
BPE's merge order is deterministic for a fixed corpus and `vocab_size` —
reproduces the identical vocabulary today, but only as long as
`data/raw/` never changes between training and evaluating. That's a
fragile coincidence to depend on, not a guarantee.

### Worked example: why id 4171 must keep meaning the same thing

Say pretraining ran BPE on a corpus and merge #3915 happened to produce
the token `"ing"` at id `4171`. During pretraining, the model's embedding
table row 4171 is nudged, over thousands of steps, into a vector that
captures "this position is usually followed by a verb-continuing
context," and the LM head's row 4171 learns to output a high score for
"ing" whenever the preceding context suggests a continuing verb.

Now fine-tune that checkpoint with a tokenizer retrained from scratch on a
small instruction dataset. Suppose *that* corpus's own frequency
statistics happen to produce a completely different token — say `"Inst"`
(the start of the word "Instruction," which appears at the start of every
example) — at id `4171` this time, purely because it was one of the more
frequent byte-pairs in that smaller corpus. Every time the fine-tuning
data contains `"Inst"`, it's encoded as id 4171 and fed into the model.
The model has no idea "Inst" is now being represented by 4171 — its
embedding row 4171 still encodes "verb-continuing context," learned from
"ing." The model is being asked to process `"Inst"` through the lens it
built for `"ing"`. Training still runs, loss still moves, nothing crashes
— but the signal is corrupted from the first token onward.

Tokenizer persistence exists so id 4171 keeps meaning the same string
everywhere that checkpoint's weights are ever used again — evaluation,
fine-tuning, and generation alike.

### What tokenizer persistence means

Saving a tokenizer means writing down, on disk, everything needed to
reconstruct the *exact* mapping between strings and integers that was
used to produce a given checkpoint's training data — so it can be loaded
back later instead of re-derived (and possibly re-derived differently) by
running `train()` again.

Three things fully define a trained `BPETokenizer`:

1. **The vocabulary** — every token id's underlying bytes
   (`self.vocab: dict[int, bytes]`). This is what `decode()` needs.
2. **The merge order** — which byte pair became which new token id, in
   the exact order merges were learned (`self.merges: dict[Pair, int]`).
   This is what `encode()` needs: it must re-apply merges in the same
   order they were learned, or a piece of text can encode to a different
   sequence of ids than it did at training time.
3. **Special-token metadata** — none exist yet in this project (no
   `<eos>`/`<pad>`/`<unk>` — see the simplification note above), but the
   save format should have a place for them so a future addition (e.g. an
   end-of-text token for Stage 7-style generation) doesn't require a
   second, incompatible file format. An empty dict is fine for now.

Both `vocab` and `merges` are fully determined by each other plus the
base 256 byte vocabulary (`vocab[256 + i]` is always
`vocab[merges_in_order[i][0]] + vocab[merges_in_order[i][1]]`), so in
principle only the merge order needs to be saved and the vocab could be
rebuilt from it — but saving both is simpler, harder to get wrong, and
avoids re-deriving anything at load time.

### How `save()` and `load()` should work

```python
tokenizer.save("checkpoints/latest/tokenizer.json")
...
tokenizer = BPETokenizer.load("checkpoints/latest/tokenizer.json")
```

- `save(path)`: write `merges` (as an ordered list of `[id_a, id_b, new_id]`
  triples — JSON object keys can't be tuples, so the pair has to be
  flattened) and `vocab` (as `{id: hex-or-base64-encoded bytes}`, since
  JSON strings can't hold arbitrary raw bytes) to a JSON file. Plain JSON,
  not `pickle` — a tokenizer file should be safe to open in a text editor
  and safe to load without executing arbitrary code (unlike the
  checkpoint's current `torch.load(..., weights_only=False)`, which this
  intentionally avoids repeating).
- `load(path)`: read that JSON back, reconstruct `self.vocab` and
  `self.merges` exactly, and return a `BPETokenizer` with no `train()`
  call involved. `encode()`/`decode()` behave identically to the instance
  that was saved, because the same on-disk data is what defines them.

This is a plain save/load pair, not a new "format version" system —
scope stays exactly this list, with the special-token slot as the only
forward-looking addition.

### How training should save the tokenizer

`train_model` (Stage 4) currently saves only `model_state_dict` and
`model_config` to `checkpoint_dir/latest.pt`. Once BPE training happens in
`scripts/train.py` (it already does, today, once per run — this doesn't
change), the resulting tokenizer should be saved *next to* the checkpoint
it trained the data for, e.g. `checkpoint_dir/tokenizer.json`, not
recomputed from `data/raw_path` by every later script. A checkpoint
directory becomes a single self-contained unit: weights + config +
tokenizer, all produced by one training run, all needed together to use
the model correctly again.

### How evaluation, fine-tuning, and generation should load it

`scripts/evaluate.py`, `scripts/finetune.py`, and `scripts/generate.py`
currently each re-run `BPETokenizer().train(corpus, vocab_size=...)` on
some corpus (the pretraining corpus for the first two, and — incorrectly,
per the worked example above — the fine-tuning corpus for
`scripts/finetune.py`). All three should instead call
`BPETokenizer.load(checkpoint_dir / "tokenizer.json")` and never call
`train()` again after the original pretraining run. This removes the
`--config`-only-for-the-tokenizer plumbing each script currently needs
(reading `data.raw_path`, re-globbing `.txt` files, re-encoding the whole
corpus just to get a tokenizer object) and removes the specific bug this
milestone exists to fix: fine-tuning will use the exact tokenizer
pretraining used, because it's the *same object*, loaded from disk, not a
new one trained from different text.

### Migration and compatibility with existing checkpoints

Checkpoints already produced by this project (Stages 4–7's manual smoke
tests) were saved *before* tokenizer persistence existed — no
`tokenizer.json` sits next to them. **Resolved: refuse, don't
reconstruct.** `load_tokenizer_for_checkpoint` raises `FileNotFoundError`
with an explicit explanation when a checkpoint has no `tokenizer.json`,
rather than silently falling back to retraining one from a corpus. A
retrained fallback would only be trustworthy if the original raw corpus
were provably byte-identical to what was used at pretraining time — an
assumption this project has no way to verify, and getting it wrong is
exactly the silent-corruption failure mode this whole milestone exists to
close. Old checkpoints (all of which only ever came from tiny smoke tests
on scratch corpora, per the manual-run notes in Stages 4–7's status
sections) aren't migrated — regenerate them with the current
`scripts/train.py`, which now saves `tokenizer.json` automatically.

## Status: implemented and tested

`src/llm_from_scratch/tokenizer/bpe.py` adds `BPETokenizer.save(path)` and
`BPETokenizer.load(path)` (classmethod), matching the design above exactly
— plain JSON, both `merges` and `vocab` written, a reserved (currently
empty) `special_tokens` slot. `train_model` (Stage 4,
`src/llm_from_scratch/train/loop.py`) takes an optional `tokenizer`
argument and, when given one, saves it as `tokenizer.json` next to
`latest.pt` in `checkpoint_dir`. `load_tokenizer_for_checkpoint`
(`src/llm_from_scratch/finetune/checkpoint.py`, alongside Stage 6's
`load_pretrained_model`) loads that file given a checkpoint path, raising
`FileNotFoundError` for a checkpoint that predates persistence (see
"Migration" above) instead of retraining.

`scripts/train.py` now passes its trained tokenizer into `train_model`.
`scripts/evaluate.py`, `scripts/finetune.py`, and `scripts/generate.py` all
call `load_tokenizer_for_checkpoint` instead of retraining BPE from a
corpus. `scripts/generate.py` no longer needs `--config` at all, since the
tokenizer no longer requires the original raw corpus to reconstruct.
`scripts/finetune.py`'s tokenizer bug (it previously retrained BPE on the
*fine-tuning* corpus instead of the pretraining corpus — see the worked
example above) is fixed: it now loads the pretrained checkpoint's exact
tokenizer and passes it through to `train_model`, so the fine-tuned
checkpoint's `tokenizer.json` is byte-identical to the pretrained one.

Tests: `tests/test_tokenizer.py` adds save/load round-trip coverage
(merges/vocab equality, and that encode/decode behavior is identical
after a save+load cycle). `tests/test_train.py` adds coverage that
`train_model` writes `tokenizer.json` when given a tokenizer and writes
nothing when not. `tests/test_finetune.py` adds coverage for
`load_tokenizer_for_checkpoint`, both the success path and the explicit
`FileNotFoundError` for a checkpoint with no saved tokenizer. Full suite:
`56 passed` (7 tokenizer + 10 data + 9 model + 11 train + 8 eval + 7
finetune + 4 generate).

Manual end-to-end verification: pretrained a tiny checkpoint
(`vocab_size=300`, `context_length=16`, 60 steps, train_loss 4.24 → 2.39),
confirmed `tokenizer.json` was written next to `latest.pt`; ran
`scripts/evaluate.py` and `scripts/generate.py` (the latter with no
`--config` flag) successfully against it with no retraining; ran
`scripts/finetune.py` against it on a 6-example instruction set and
confirmed the fine-tuned checkpoint's `tokenizer.json` is byte-for-byte
identical to the pretrained one (`diff` reports no difference) — direct
confirmation the fine-tuning tokenizer bug is fixed. Separately confirmed
that pointing `scripts/generate.py` at a checkpoint directory with no
`tokenizer.json` fails immediately with the explicit migration error
above, rather than silently retraining.

`src/llm_from_scratch/tokenizer/bpe.py` implements `BPETokenizer` with
`.train(corpus, vocab_size)`, `.encode(text)`, `.decode(ids)`, matching the
algorithm above exactly (no regex pre-tokenization yet — see the
simplification note). Tests in `tests/test_tokenizer.py` cover training,
encoding, and lossless round-trip on both ASCII and non-ASCII text.
