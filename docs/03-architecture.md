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

## What we build here
A minimal GPT-style decoder: embedding layer → N transformer blocks
(masked multi-head attention + MLP, with residuals and layer norm) →
output head. Every piece built by hand, no `nn.TransformerEncoder` shortcut.

## Status: not started
