# 06 — Fine-tuning

## What it is
Taking the pretrained model — which only knows how to continue text — and
adapting it to follow instructions, using a smaller dataset of
(instruction, response) pairs.

## Why it matters
A raw pretrained model completes text plausibly; it doesn't "answer
questions" or "follow instructions" on its own. Fine-tuning is what turns
a text-continuation engine into something that behaves like an assistant.

## The one key idea
Fine-tuning is the same training loop as pretraining (same loss, same
optimizer mechanics) — the only real differences are: much less data,
a much lower learning rate, and data formatted as instruction/response
pairs instead of raw text. It's a continuation of stage 4, not a new
technique.

## What we build here
A supervised instruction-tuning pass on top of the pretrained checkpoint,
using a small hand-built or public instruction dataset, with before/after
generation comparisons to make the effect visible.

## Status: not started
