# 04 — Pretraining Loop

## What it is
The actual training process: feed batches through the model, compare
predicted next-token distribution to the real next token (cross-entropy
loss), backpropagate, update weights, repeat — for many steps.

## Why it matters
This is where "architecture" becomes "a model that knows things." A
correct architecture with a broken training loop (bad learning rate,
no gradient clipping, wrong loss) produces nothing useful. This stage is
mostly about getting the mechanics right and watching loss actually fall.

## The one key idea
The loss is just: "how surprised was the model by the real next token?"
Lower surprise, on average, across the whole corpus = better model. Every
training trick (learning rate warmup, gradient clipping, weight decay)
exists to make that number go down smoothly instead of exploding or
stalling.

## What we build here
A plain PyTorch training loop with checkpointing, logging (loss/step,
tokens/sec), gradient clipping, and a simple LR schedule — no external
training framework, so every step is visible.

## Status: not started
