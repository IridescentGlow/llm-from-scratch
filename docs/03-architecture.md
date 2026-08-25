# 03 — Architecture

## What it is
The transformer itself: token embeddings + positional information, stacked
blocks of self-attention and feed-forward layers, and a final projection
back to vocabulary-sized logits (a score per possible next token).

## Why it matters
This is the actual "brain." Self-attention is what lets the model weigh
every earlier token when predicting the next one, instead of only looking
at a fixed nearby window — that's the core idea that made transformers
outperform older architectures (RNNs).

## The one key idea
Self-attention answers, for every token: "which earlier tokens should I
pay attention to, and how much?" It computes this with three learned
projections (query, key, value) and a softmax-weighted sum — everything
else in the architecture (multi-head, layer norm, residual connections)
exists to make that one operation trainable at depth.

## The big picture: token IDs in, next-token scores out

Stage 2 hands the model a batch of `(batch_size, context_length)` integer
token IDs. The model's job is: for every position in that sequence,
output a score for every possible next token (a "logit" — an
un-normalized number, higher means "more likely"). Shape in:
`(batch_size, context_length)`. Shape out:
`(batch_size, context_length, vocab_size)`. Everything between those two
shapes is the architecture.

The flow, top to bottom:

```
token IDs
  → token embedding + positional embedding   (numbers → vectors)
  → N × [ self-attention → feed-forward ]      (the "thinking")
  → final layer norm
  → LM head (linear projection to vocab_size)  (vectors → logits)
```

## Embeddings: turning IDs into vectors the model can reason over

A token ID is just an arbitrary integer index — `id=42` isn't "closer" to
`id=43` in any meaningful sense. The **token embedding** is a lookup table
(one learned vector per vocabulary entry, length `n_embd`) that gives each
token a position in a continuous space, where distance and direction can
mean something the model learns during training (e.g. related words end
up near each other).

## Positional information: where in the sequence is this token?

Attention (below) has no built-in sense of order — it treats the sequence
as a set unless told otherwise. A **positional embedding** is a second
lookup table (one learned vector per position, `0` to `context_length-1`)
added element-wise to the token embedding. After this, each vector encodes
both *what* the token is and *where* it sits in the sequence.

## Self-attention: the core operation

For every token, self-attention builds three vectors from its embedding
via three learned linear layers:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain?" (one per token, including this one)
- **Value (V)** — "what do I actually offer, if attended to?"

Compare this token's query against every earlier token's key (dot
product) to get a relevance score per pair, turn those scores into
weights that sum to 1 (**softmax**), then take the weighted sum of every
earlier token's value vector. That weighted sum is the output — a new
vector for this token, blended from whichever earlier tokens mattered
most.

"Masked": a token may only attend to itself and earlier tokens, never
later ones — otherwise the model could "cheat" by seeing the very token
it's supposed to predict. This is enforced by setting scores for future
positions to `-infinity` before the softmax, so their weight becomes 0.

"Multi-head": instead of one Q/K/V computation, split `n_embd` into
`n_head` smaller chunks and run attention independently on each, then
concatenate the results. Each head can learn to track a different kind of
relationship (e.g. one head tracks "the subject of this verb," another
tracks "the noun this adjective modifies"). `configs/small.yaml` sets
`n_head`.

## Feed-forward layer: per-token processing

After attention mixes information *across* tokens, the feed-forward layer
(a small 2-layer MLP: expand, non-linearity, project back down) processes
*each token's vector independently* — no cross-token mixing here. If
attention is "gather relevant context," the feed-forward layer is "now
think about what that context means" for this token alone.

## Residual connections: don't erase, add

Each sub-layer (attention, feed-forward) doesn't replace its input — it
computes a delta and adds it: `output = input + sublayer(input)`. This
matters because with many stacked layers (`n_layer` of them), gradients
during training would otherwise shrink to nothing on the way back through
the network (vanishing gradients). The residual path gives gradients a
direct shortcut back to every earlier layer.

## Layer normalization: keep the numbers well-behaved

Before each sub-layer, **layer norm** rescales a token's vector to have
mean 0 and variance 1 (then applies a small learned scale/shift). Without
this, values can drift to very large or very small magnitudes as they
pass through many layers, making training unstable. This project uses
"pre-norm" (normalize *before* attention/feed-forward, not after) —
the modern standard, more stable to train than the original 2017 paper's
"post-norm."

## The LM head: vectors back to vocabulary scores

After the last transformer block and a final layer norm, one more linear
layer projects each token's `n_embd`-length vector to a `vocab_size`-length
vector of logits — one score per possible next token. Feed those logits
through a softmax to get a probability distribution, and the highest-
probability token is the model's best guess for "what comes next."

## Worked example: shapes through the model

Config: `n_embd=384`, `n_head=6`, `context_length=256`, `vocab_size=8000`,
`batch_size=32` (this project's `configs/small.yaml`).

```
input token IDs:            (32, 256)
after token+pos embedding:  (32, 256, 384)
after each transformer block: (32, 256, 384)   <- shape unchanged, content refined
after final layer norm:     (32, 256, 384)
after LM head:               (32, 256, 8000)
```

Notice the sequence shape `(32, 256, 384)` never changes across the N
transformer blocks — each block refines the vectors in place, adding more
context-aware information each time, without changing their shape. Only
the very last step (the LM head) changes the last dimension, from
`n_embd` to `vocab_size`.

## Simplification note

Real production models (GPT-3/4 class) use optimizations we skip here:
rotary or learned-relative positional encodings instead of a plain
lookup table, FlashAttention-style fused attention kernels for speed,
grouped-query attention to cut memory, and often much deeper/wider
stacks. We use plain learned absolute positional embeddings and a
straightforward (slower, but fully visible) attention implementation —
the point here is to see every operation, not to be fast.

## What we build here
A minimal GPT-style decoder: embedding layer → N transformer blocks
(masked multi-head attention + MLP, with residuals and layer norm) →
output head. Every piece built by hand, no `nn.TransformerEncoder` shortcut.

## Status: implemented and tested

`src/llm_from_scratch/model/gpt.py` implements `GPTConfig`,
`CausalSelfAttention`, `FeedForward`, `Block`, and `GPT` (embeddings → N
`Block`s → final layer norm → LM head), matching the doc's flow exactly.
`GPT.forward(idx, targets=None)` returns `(logits, loss)`, with loss
computed via cross-entropy when `targets` is given. `GPT.generate(idx,
max_new_tokens)` greedily extends a sequence (argmax over next-token
logits each step), truncating context to `context_length` as needed.
Tests in `tests/test_model.py` cover construction, forward-pass logits
shape, loss computation, causal masking (a later token can't affect an
earlier position's output), and generation.

Update (weight tying + GPT-style initialization milestone): `GPT` now has
no separate `lm_head` parameter — the final projection to vocab-sized
logits reuses `token_embedding.weight` directly, and every
`nn.Linear`/`nn.Embedding` weight is explicitly initialized (`N(0, 0.02²)`,
with extra `1 / sqrt(2 * n_layer)` scaling on the two projections that
write onto the residual stream) instead of relying on PyTorch's library
defaults. No shape, flow, or forward-pass logic described above changed.
See `docs/weight-tying-initialization.md` for the full design, worked
examples, and legacy-checkpoint policy.
