# Milestone — Response-Only Loss Masking for Fine-Tuning

## What the current fine-tuning loss does

Stage 6 (`docs/06-finetuning.md`) formats each instruction/response pair
into one string, tokenizes it, and treats it exactly like pretraining
text: slide a fixed-size window over the token stream, and at every
position predict "the next actual token" (Stage 2's shift-by-one
`TokenDataset`). Every position in every window contributes equally to
the cross-entropy loss — including the `Instruction: ...` and
`Response: ` tokens, not just the actual answer.

## Why that penalizes the model for the wrong thing

The instruction text (and the literal `Instruction:` / `Response:`
labels) is *given* to the model, not something it needs to learn to
produce — at inference time, the instruction is always supplied by
whoever's asking. Grading the model on how well it reproduces text it
was handed, rather than only on the response it has to generate itself,
adds noise to the gradient signal: some of every update is spent nudging
the model to predict already-known template/instruction tokens, which
doesn't teach it to be a better assistant. It also means a long,
information-dense instruction can dominate the loss for a given example
over a short response, even though the response is the part that
actually matters.

## Why response tokens should carry the loss

The point of fine-tuning is to teach the model to *generate a good
response* to a given instruction. The only tokens the model actually
needs to produce well are the response tokens (and the signal to stop —
see EOS, below). Concentrating every gradient update on exactly those
tokens focuses the whole fine-tuning budget — already tiny compared to
pretraining — on the one skill being taught.

## What a loss mask is

A loss mask is a same-length companion array to the target tokens: a
boolean flag per position saying "does this target token count toward
the loss?" It doesn't remove any tokens from the model's input — the
model still reads the full instruction as context, exactly as before.
It only changes which *target* positions the cross-entropy loss and
backward pass look at.

Concretely, this project already has a mechanism for exactly this:
`F.cross_entropy`'s `ignore_index` parameter (PyTorch default: `-100`)
skips any target position whose value equals that sentinel, contributing
zero loss and zero gradient. So "masking" a position means: don't set a
flag anywhere the model has to check — just set that position's *target
id* to `-100` before it ever reaches the loss. `GPT.forward`
(`src/llm_from_scratch/model/gpt.py`) already calls
`F.cross_entropy(logits.view(...), targets.view(...))` with no
`ignore_index` argument, which means it's already using the `-100`
default. **No change to the model or the loss call is needed at all** —
masking is entirely a matter of what target ids get built upstream, in
the fine-tuning data pipeline.

## How the mask should distinguish prompt vs. response vs. EOS

For one formatted example:

```
Instruction: {instruction}
Response: {response}
```

Split it into two parts:
- **prompt** = `"Instruction: {instruction}\nResponse: "` (everything up
  to and including the `Response: ` label) → masked out (`-100`), not
  part of the loss.
- **response** = `"{response}\n"` plus the EOS token appended after it →
  included in the loss normally.

EOS is deliberately *not* masked: the model has to be graded on
predicting "stop here" the same way it's graded on the words of the
response, or it never learns to stop (see
`docs/eos-generation-stopping.md`, which this milestone builds on
without changing).

## Worked example

Take the same illustrative example as `docs/06-finetuning.md`:

```
Instruction: Translate 'good morning' to French.
Response: Bonjour.
```

Split into prompt / response:

```
prompt   = "Instruction: Translate 'good morning' to French.\nResponse: "
response = "Bonjour.\n"
```

Illustrative token ids (not real ids, just to show the shape — same
numbers `docs/06-finetuning.md` already used):

| token id | 15 | 372 | 9 | 4 | 88 | 2 | 501 | 9 | 4 | 12 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| part | prompt | prompt | prompt | prompt | prompt | prompt | prompt | prompt | response | response | EOS |

(the first 8 ids are `Instruction: Translate 'good morning' to
French.\nResponse: `; the last 3 are `Bonjour.` + `\n` + EOS)

Stage 2's shift-by-one windowing turns this into (input, target) pairs.
The **mask applies to the target**, at the position of the token being
*predicted*, not the token being read:

With 11 tokens total there are 10 (input, target) pairs — the last one
(input id `12`, target id `3`) predicts EOS:

| input position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| input id | 15 | 372 | 9 | 4 | 88 | 2 | 501 | 9 | 4 | 12 |
| target id | 372 | 9 | 4 | 88 | 2 | 501 | 9 | 4 | 12 | 3 |
| target id as passed to the loss | -100 | -100 | -100 | -100 | -100 | -100 | -100 | -100 | 12 | 3 |
| contributes to loss? | no | no | no | no | no | no | no | no | **yes** | **yes** |

Only the last two positions — predicting `Bonjour.` + `\n` (id `12`) and
then EOS (id `3`) — contribute to cross-entropy and backprop. The other
eight positions still run through the model exactly as before (the model
still reads and attends to the whole instruction), they just don't
generate a gradient.

## Where this plugs into the existing pipeline

- `src/llm_from_scratch/finetune/data.py`'s `build_token_ids` currently
  returns one flat `list[int]` per fine-tuning corpus (concatenating all
  examples + EOS). A new `build_masked_token_ids` will return that same
  token id list *plus* a parallel `list[bool]` ("does this position's
  token count as a loss target") built the way the worked example above
  shows — by encoding the prompt prefix separately to find the
  prompt/response split point, per example.
- `src/llm_from_scratch/data/dataset.py`'s `TokenDataset` gets an
  optional `loss_mask` array (same length as `token_ids`, default
  `None` — pretraining and evaluation are untouched). When given, the
  target tensor it returns has `-100` written at every masked position
  before being handed to the model — the same shift-by-one windowing as
  today, just with that one extra step.
- `GPT.forward` and `train_model`/`estimate_loss`/`evaluate_model`
  require **no code changes** — they already just consume whatever
  `(input_ids, target_ids)` the dataset hands them, and
  `F.cross_entropy`'s default `ignore_index=-100` already does the
  skipping.
- `scripts/finetune.py` switches from `build_token_ids` to
  `build_masked_token_ids`, persists the mask alongside the token ids,
  and passes it to both the train and validation `TokenDataset`s (val
  loss should also reflect "how well does it predict the response,"
  since that's the actual objective now).

## Why this doesn't change the model architecture

Nothing about `GPT` changes — same layers, same shapes, same forward
pass, same `F.cross_entropy` call. The mask only changes *which target
positions supervise training*, which is a property of the training
*data*, not the model. This is the same spirit as the whole fine-tuning
stage: reuse Stage 3's `GPT` and Stage 4's `train_model` completely
unchanged, and only add new code where the actual new idea lives.

## Why this preserves pretrained general-language behavior better

Every gradient update fine-tuning applies is a small nudge away from the
pretrained weights (see `docs/06-finetuning.md`, "Why fine-tuning uses a
much smaller learning rate," on catastrophic forgetting). Right now, part
of that nudging budget is spent on "predict this instruction text you
were just given" — a skill the pretrained model, being a competent
language model, already mostly has, so this wastes some of the (already
tiny) fine-tuning signal reinforcing something not in question. Spending
100% of the gradient budget on "produce this response, then stop" is a
narrower, more targeted signal, which means fewer, more precise nudges
are needed to reach the desired behavior — less total movement away from
the general-language knowledge pretraining built.

## Known limitation: the prompt/response split point

The split between "prompt" and "response" tokens is found by encoding
the prompt prefix alone and comparing it against the encoding of the
full formatted example — the split index is `len(encode(prefix))`. BPE
merges are chunk-local (`docs/tokenizer-performance.md`) and this
project's pretokenizer splits on word/whitespace boundaries, so in
practice the prefix's tokens come out identical whether encoded alone or
as the start of the full string. This isn't mathematically guaranteed
for every possible tokenizer/string combination, though — it's a
practical assumption, not a proof. Out of scope for this milestone to
fully close; noted here so it isn't a silent unstated assumption.

## Known limitation: a window entirely inside one long prompt

`TokenDataset` slides a fixed-size `context_length` window over the
concatenated token stream (see "Explicitly out of scope," below, on why
padding isn't used instead). If a single example's prompt is longer than
`context_length`, a window can land entirely inside that prompt, with
every target position masked. `F.cross_entropy` returns `nan` for a batch
with nothing left to average over -- and because losses are summed across
batches (`evaluate_model`) or used directly for backprop (`train_model`),
one `nan` batch silently poisons the whole run. This isn't guarded
against here: it doesn't occur with this project's real configs
(`context_length: 256` against short hand-written instructions), and
guarding against a scenario the current data can't produce would be
premature. If it ever becomes a real problem, the fix would be a
contained one -- skip batches with no unmasked targets -- not a
redesign of the windowing itself.

## Explicitly out of scope for this milestone

- Padding (examples still get concatenated into one token stream and
  windowed, exactly like today — no per-example fixed-length batches).
- Any change to pretraining, evaluation of pretrained checkpoints, or
  generation.
- Chat templates, multi-turn formatting, RLHF/preference optimization.
