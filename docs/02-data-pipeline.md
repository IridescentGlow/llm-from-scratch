# 02 — Data Pipeline

## What it is
The process that turns raw tokenized text into fixed-size (input, target)
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

## What we build here
A `Dataset`/`DataLoader` pair that reads token IDs, yields
`(input_ids, target_ids)` batches of shape `(batch_size, context_length)`,
and handles corpus files too large to fit in memory at once.

## Status: not started
