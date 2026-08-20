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

## Status: implemented and tested

`src/llm_from_scratch/tokenizer/bpe.py` implements `BPETokenizer` with
`.train(corpus, vocab_size)`, `.encode(text)`, `.decode(ids)`, matching the
algorithm above exactly (no regex pre-tokenization yet — see the
simplification note). Tests in `tests/test_tokenizer.py` cover training,
encoding, and lossless round-trip on both ASCII and non-ASCII text.
