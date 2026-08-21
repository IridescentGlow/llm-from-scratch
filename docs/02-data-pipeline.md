# 02 — Data Pipeline

## What it is
The process that turns raw tokenized text into fixed-size `(input, target)`
pairs the model actually trains on — sliding a context-length window over
the token stream, where the target is just the input shifted by one token.

## Why it matters
This is where "predict the next token" becomes a concrete training signal.
Batch size, context length, and how you sample from the corpus directly
affect training stability and how much of the data the model actually sees.

## The one key idea
Language modeling only needs one label: the next token. You don't need
annotated data — the text labels itself. The pipeline's job is just
windowing + batching + shuffling that efficiently.

## From text to training signal

Stage 1 turned text into a single long list of token IDs. Stage 4
(pretraining) needs the model to practice one skill over and over: given
some tokens, predict the next one. Training a neural net means showing it
many `(input, correct answer)` examples. This stage manufactures those
examples out of the plain token stream — no human labeling required.

## Windowing: input and target are the same slice, offset by one

Pick a `context_length` (how many tokens the model looks at once — set in
`configs/small.yaml`). For any starting position `i` in the token stream:

- `input`  = tokens `[i : i + context_length]`
- `target` = tokens `[i+1 : i + context_length + 1]`

`target` is `input` shifted one position to the right. At every position
in the window, the token immediately after it is the "correct answer" for
that position. One window gives the model `context_length` training
examples at once (predict token 2 from token 1, predict token 3 from
tokens 1–2, etc.) — that's why decoder-only training is efficient.

## Worked example

Token stream (using letters as stand-ins for token IDs):
`[A, B, C, D, E, F, G]`, with `context_length = 3`.

Starting at position 0:
```
input:  [A, B, C]
target: [B, C, D]
```
Read as three examples layered on top of each other: "after A, predict B",
"after A,B predict C", "after A,B,C predict D". Slide the window forward
to get the next training example. Sliding by exactly `context_length` (no
overlap) uses each token as an input exactly once; sliding by a smaller
stride reuses tokens across multiple windows, giving more training
examples from the same corpus at the cost of redundancy.

## Batching

The model processes many windows at once, not one at a time — this is
what makes training fast on a GPU. Stack `batch_size` windows into one
tensor of shape `(batch_size, context_length)` for inputs, and the
matching shape for targets. `configs/small.yaml` sets `batch_size`.

## Handling corpora too big for memory

A "few hundred MB" corpus (this project's target scale) can still
be too large to tokenize and hold as one Python list comfortably, and
production corpora are much bigger than that. The fix: pre-tokenize once
into a flat binary file of token IDs on disk (fixed-width integers), then
read windows from it with `numpy.memmap` — the OS pages data in from disk
on demand instead of loading the whole file into RAM. Windowing logic
doesn't change; only where the token stream physically lives does.

## Train/validation split

We hold out a slice of the corpus (`train_split` in `configs/small.yaml`,
e.g. 0.9) that the model never trains on, so Stage 5 (evaluation) can
measure performance on text the model hasn't memorized. Split is done once
on the raw token stream, by position — the first 90% of tokens are train,
the rest are validation.

## Simplification note

Real training pipelines shuffle window start positions randomly each
epoch, use multiple parallel data-loading workers, and often pack multiple
short documents into one window with document-boundary masking. We're
sampling random start positions from a single memmapped array with no
multi-worker loading — simple, but slower to saturate a fast GPU. Good
enough for a small model on modest data.

## What we build here
A `Dataset`/`DataLoader` pair that reads token IDs, yields
`(input_ids, target_ids)` batches of shape `(batch_size, context_length)`,
and handles corpus files too large to fit in memory at once (via
`numpy.memmap`).

## Status: implemented and tested

`src/llm_from_scratch/data/tokens.py` handles on-disk token storage
(`write_token_ids`, `load_token_ids` via `numpy.memmap`, `train_val_split`).
`src/llm_from_scratch/data/dataset.py` implements `TokenDataset` (windowing
+ shift-by-one) and `get_dataloader` (batching, `shuffle=True` for random
sampling). Both work identically over a plain array or a memmap. Tests in
`tests/test_data.py` cover input/target alignment, batch shapes, train/val
split correctness, and memmap round-trip/windowing.
