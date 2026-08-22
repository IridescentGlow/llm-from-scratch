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

## Pretraining vs. fine-tuning

| | Pretraining (Stage 4) | Fine-tuning (Stage 6) |
|---|---|---|
| Data | hundreds of MB of raw text | a few dozen–thousand instruction/response pairs |
| Goal | learn general language patterns | learn a specific behavior/format |
| Starting weights | random | pretrained checkpoint |
| Learning rate | higher | much lower |
| Steps | thousands+ | far fewer |

Everything else — the model, the forward pass, the cross-entropy loss, the
optimizer, gradient clipping — is identical. Fine-tuning reuses Stage 4's
`train_model` unchanged, just starting from loaded weights instead of
random ones, on different data, with different hyperparameters.

## Adapting a pretrained model to a smaller dataset

A freshly-initialized model (Stage 4's starting point) knows nothing:
random weights, random predictions. Loading a pretrained checkpoint
instead means every weight already encodes grammar, common phrasing, and
factual associations learned from a large corpus. Continuing to train
those same weights on a small, specialized dataset nudges them toward the
new behavior *without* throwing away what was already learned — as long
as the nudge is gentle (see learning rate, below). This only works because
the architecture doesn't change: same `GPTConfig`, same shapes, same
`GPT` class from Stage 3. Fine-tuning just means: build a `GPT` from the
checkpoint's saved config, load the checkpoint's saved weights into it,
then keep training.

## Instruction/response training, in simple terms

An instruction-tuning example is a pair: an instruction (what to do) and
a response (what a good answer looks like). To train on it with a
plain next-token-prediction loop — the only kind of loop this project
has — the pair is flattened into one piece of text with a fixed template,
then treated exactly like pretraining text: tokenize it, slide a
context window over it, and predict each next token from the ones
before it.

```
Instruction: Translate 'good morning' to French.
Response: Bonjour.
```

Every instruction/response pair uses this same two-line template, so the
model can learn the *shape* of the format itself (it starts with
"Instruction:", the response follows "Response:") — not just the specific
answers.

## Why the pretrained weights are valuable starting knowledge

Without pretraining, a model would have to learn English, French,
spelling, and grammar *and* the instruction-following format all from a
tiny dataset — nowhere near enough data for that. Because the pretrained
model already knows language in general, fine-tuning only has to teach it
one narrower thing: *given this template, produce a response like this*.
That's a much smaller amount of new information, which is exactly why
fine-tuning works with orders of magnitude less data than pretraining.

## Why fine-tuning uses a much smaller learning rate

A large learning rate makes large weight updates. On a small dataset,
large updates can overwrite the general knowledge the pretrained weights
encode in just a few steps — the model "forgets" what it knew before it's
had a chance to build on it (this failure mode is called **catastrophic
forgetting**). A much smaller learning rate makes each update a gentle
nudge: enough to shift behavior toward the new format, not enough to
erase what pretraining built. As a rule of thumb, fine-tuning learning
rates are commonly 5–20x smaller than the pretraining rate.

## Overfitting on a small fine-tuning dataset

A fine-tuning set might be a few dozen to a few thousand examples — tiny
compared to the model's parameter count. A model that large can simply
memorize a small dataset word-for-word if trained too long, rather than
learning the general pattern. The symptom is the same as in Stage 5:
training loss keeps dropping while validation loss (on held-out
instruction/response pairs, not seen during fine-tuning) stalls or rises.
The fix is also the same idea as always: watch validation loss, and stop
once it stops improving, rather than training for a fixed large number of
steps by default.

## Worked example: building the target

Say the corpus (after formatting) contains:

```
Instruction: Translate 'good morning' to French.
Response: Bonjour.
```

Tokenized, this is just a sequence of token ids, e.g. (illustrative, not
real ids): `[15, 372, 9, 4, 88, 2, 501, 9, 4, 12, 3]`. Exactly like Stage
2's sliding window: the **input** is the sequence, and the **target** is
the same sequence shifted one position to the left — at every position,
the target is "the next actual token." No special masking, no separate
"instruction" and "response" token streams — it's the same
input/target-shift-by-one construction Stage 2 already built, applied to
instruction-formatted text instead of raw prose.

## Simplification note

Production instruction-tuning usually **masks the loss on the prompt
tokens** — the model only gets penalized for mispredicting the response,
not the instruction text, since the instruction is given, not generated.
That requires a per-example dataset (not a single sliding window over
concatenated text) and a masked loss. We skip that here to keep reusing
Stage 2's `TokenDataset` and Stage 4's `train_model` completely unchanged;
the model still learns to produce reasonable responses, just from a
slightly noisier signal (it's also "graded" on how well it reproduces the
instruction text, not only the response). This is a real difference from
production instruction-tuning, not a hidden bug.

## What we build here

A supervised instruction-tuning pass on top of a pretrained checkpoint,
using a small hand-built instruction dataset, reusing Stage 1's tokenizer,
Stage 2's data pipeline, Stage 3's model, and Stage 4's training loop
completely unchanged — only new code is instruction-pair formatting and
the script that wires it together — with before/after loss and generation
comparisons to make the effect visible. No RLHF, no preference
optimization, no LoRA/adapters, no distributed training — plain full-model
fine-tuning, same as pretraining.

## Status: implemented and tested

New code, reusing Stages 1–5 unchanged:

- `src/llm_from_scratch/finetune/data.py` — `InstructionExample`,
  `format_example` (the fixed template above), `build_corpus` (joins
  formatted examples into one text blob), `load_examples_jsonl` (reads a
  `{"instruction", "response"}`-per-line JSONL file).
- `src/llm_from_scratch/finetune/checkpoint.py` — `load_pretrained_model
  (checkpoint_path) -> GPT`: rebuilds a `GPT` from a Stage 4 checkpoint's
  saved `model_config` + `model_state_dict`.
- `scripts/finetune.py` — loads the pretrained checkpoint, formats
  `data/finetune/examples.jsonl` into text, runs it through the exact
  Stage 1 tokenizer / Stage 2 `TokenDataset` / Stage 4 `train_model`
  pipeline unchanged (only the learning rate, step count, and checkpoint
  directory differ, via `configs/finetune.yaml`), and prints val loss +
  perplexity + a short generation sample before and after fine-tuning.
- `configs/finetune.yaml` — `train:` (small `max_steps`, low
  `learning_rate`, separate `checkpoint_dir: checkpoints/finetuned/` so
  the pretrained checkpoint is never overwritten) and `data:` (path to the
  instruction JSONL, `train_split`). No `model:` section — the
  architecture always comes from the loaded checkpoint, not duplicated in
  YAML.
- `data/finetune/examples.jsonl` — 30 small hand-written
  instruction/response pairs used for fine-tuning.

Tests in `tests/test_finetune.py` (5, all passing) cover: the fixed
template's exact output, multi-example corpus joining and ordering, JSONL
loading, that `load_pretrained_model` restores weights and config exactly
from a saved checkpoint, and an end-to-end check that fine-tuning (via
Stage 4's `train_model`) lowers both validation loss and perplexity (via
Stage 5's `evaluate_model`) on a tiny instruction corpus.

Full suite: `44 passed` (5 tokenizer + 10 data + 7 model + 9 train + 8
eval + 5 finetune).

Manual end-to-end smoke test: pretrained a tiny checkpoint on a
hand-written corpus (`scripts/train.py`, train_loss 4.83 → 3.43 over 60
steps), then ran `scripts/finetune.py` against it with a small instruction
set. Confirmed: the pretrained checkpoint loaded correctly; fine-tuning
batches were produced and losses computed; gradients/backprop and
optimizer updates ran (train_loss 4.43 → 3.84, val_loss 4.50 → 4.10,
perplexity 90.42 → 60.44 over 80 steps); a separate fine-tuned checkpoint
was saved to a different directory than the pretrained one, with the same
architecture config but different (confirmed changed) weights. Generated
text before/after was not coherent at this toy scale (tiny model, ~400
characters of pretraining text, 8 fine-tuning examples) — expected; the
point of the smoke test is verifying the training mechanics work, not
generation quality, which needs a much larger pretraining run to assess
meaningfully.

Former limitation — this one *was* a real bug, not just an inconvenience,
and is now fixed (tokenizer persistence milestone): `scripts/finetune.py`
used to retrain a fresh tokenizer on the *fine-tuning* corpus (not the
pretraining corpus), at the checkpoint's `vocab_size`. Because BPE merges
depend on corpus frequency statistics, a tokenizer retrained on a small,
differently-distributed fine-tuning corpus assigns different strings to
the same ids than the tokenizer that produced the pretraining data —
silently invalidating the pretrained embedding table for every token
(see docs/01-tokenization.md, "Why retraining BPE on a different corpus
silently scrambles token IDs" for a worked example). `scripts/finetune.py`
now calls `load_tokenizer_for_checkpoint` and uses the checkpoint's exact
pretraining-time tokenizer — confirmed by a manual smoke test showing the
fine-tuned checkpoint's `tokenizer.json` is byte-identical to the
pretrained one's.

As documented above, this implementation does not mask the loss on
instruction tokens (production instruction-tuning usually does) — it
trains on the whole formatted instruction+response text as one
continuous sequence, reusing Stage 2's `TokenDataset` unchanged.

Update (EOS / generation stopping milestone): fine-tuning data is no longer
built by concatenating raw formatted text and tokenizing once.
`build_token_ids` (`src/llm_from_scratch/finetune/data.py`) encodes each
example separately and appends the checkpoint's EOS token id after each
one, so the model learns to stop right after a complete response, not just
at the very end of the whole corpus. This requires the pretrained
checkpoint's tokenizer to have an EOS token; fine-tuning against a
checkpoint that predates EOS now fails immediately with a clear error
instead of silently fine-tuning without any stopping signal. See
docs/eos-generation-stopping.md.
