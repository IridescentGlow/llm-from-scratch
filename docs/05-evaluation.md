# 05 — Evaluation

## What it is
Measuring whether the trained model is actually good — beyond watching
training loss decrease, which only tells you it fits the training data.

## Why it matters
A model can have great training loss and still be useless: overfit,
repetitive, incoherent past a few tokens, or simply memorizing. Evaluation
is what tells you whether to move to fine-tuning or go back and fix
something upstream.

## The one key idea
Always evaluate on held-out text the model never trained on. The gap
between train loss and validation loss tells you if you're overfitting;
qualitative generation samples tell you if it's coherent at all — numbers
alone can look fine while the output reads like nonsense.

## What we build here
A validation-loss/perplexity check on held-out data, plus a small script
that generates sample text at checkpoints so you can eyeball quality
alongside the numbers.

## Status: not started
