# 07 — Generation / Inference

## What it is
Using a trained model to produce new text, one token at a time, instead of
training it. Every prior stage was about building and improving the model;
this stage is about actually *using* it.

## Why it matters
A checkpoint on disk is useless until something turns it into text a person
can read. Generation is the payoff for Stages 1–6 — tokenizer, data
pipeline, architecture, pretraining, evaluation, fine-tuning all exist so
that this stage can produce coherent output.

## The one key idea
Generation is repeated single-step prediction: predict one next token,
append it to the sequence, then predict again — now with one more token of
context than before. There's no new model capability here, just the
existing forward pass, called in a loop, with no target and no gradient.

## Training vs. generation

| | Training (Stage 4) | Generation (Stage 7) |
|---|---|---|
| Input | a full batch of (input, target) windows | one prompt, growing by one token per step |
| Direction | forward pass, then backward pass | forward pass only |
| Gradients | computed and applied (optimizer step) | never computed |
| Output used | loss (a number) | logits → one chosen token id |
| Runs | thousands of steps over a dataset | `max_new_tokens` steps over one sequence |

The model itself doesn't change between these two modes — same `GPT.forward`,
same weights. What changes is what's done with the output: training turns
logits into a loss and a weight update; generation turns logits into the
next token.

## From prompt to token IDs

A prompt is just a string, e.g. `"Once upon a"`. Before the model can see
it, Stage 1's tokenizer (`BPETokenizer.encode`) turns it into a list of
token ids — the same encoding used at training time, since the model only
knows the vocabulary it was trained on. Those ids are wrapped into a tensor
of shape `(1, seq_len)` — batch size 1, one prompt.

## One generation step

1. **Forward pass**: feed the current sequence of token ids into
   `GPT.forward`. It returns `logits` of shape `(1, seq_len, vocab_size)` —
   for every position, a score per vocabulary token for "how likely is this
   token next."
2. **Take the last position**: only the logits at the *last* position matter
   — that's the model's prediction for the token that comes after everything
   seen so far. Shape `(1, vocab_size)`.
3. **Pick one token id** from those `vocab_size` scores (see decoding
   strategies below).
4. **Append** that token id to the sequence.
5. **Repeat**, now with a sequence one token longer, until `max_new_tokens`
   tokens have been generated.

This is the entire algorithm. Everything else in this stage (decoding
strategy, checkpoint loading, text decoding) is plumbing around this loop.

## Greedy decoding vs. temperature-based sampling

**Greedy decoding**: always pick the single highest-scoring token
(`argmax` over the logits). This is what `GPT.generate` already does (see
`src/llm_from_scratch/model/gpt.py`). It's deterministic — the same prompt
always produces the same output — and tends to be repetitive, since the
model always takes its single "safest" guess.

**Temperature-based sampling**: turn the logits into a probability
distribution with `softmax`, then *sample* a token from that distribution
instead of always taking the top one. This introduces controlled
randomness — the same prompt can produce different completions.

## What `temperature` actually changes

Before sampling, logits are divided by a `temperature` value, then passed
through `softmax`. Temperature reshapes how "peaked" or "flat" that
probability distribution is:

- `temperature = 1.0` — logits unchanged; sample from the model's
  probabilities as-is.
- `temperature < 1.0` (e.g. `0.5`) — sharpens the distribution: the
  already-likely tokens become even more likely, unlikely tokens become
  even less likely. Output gets closer to greedy decoding, more focused,
  less varied. `temperature → 0` converges to greedy decoding exactly.
- `temperature > 1.0` (e.g. `1.5`) — flattens the distribution: less-likely
  tokens get a real chance of being picked. Output gets more varied, more
  surprising, and at high enough values, less coherent.

Temperature doesn't change *what* the model predicts (the underlying
logits) — it changes how confidently the sampling step commits to the
model's top guess versus giving other options a chance.

## Why `max_new_tokens` exists

The generation loop has no natural stopping point of its own — nothing
tells it "the sentence is done," unless it's given an explicit end-of-
sequence (EOS) token to watch for (see docs/eos-generation-stopping.md,
added after this stage). `max_new_tokens` is a hard cap either way:
generate at most this many tokens, then stop regardless. It's a simple,
explicit substitute for "know when to stop" that still applies even when
EOS is available, in case the model never produces it.

## Why the model's context length matters during generation

`GPTConfig.context_length` is the longest sequence the model was trained to
handle — its positional embeddings only go up to that length
(`src/llm_from_scratch/model/gpt.py`, `GPT.forward` raises if `seq_len` >
`context_length`). As generation appends tokens, the sequence keeps
growing. Once it would exceed `context_length`, the model can no longer see
the whole thing at once. The existing `GPT.generate` already handles this
by feeding only the *last* `context_length` tokens into each forward pass
(`idx_cond = idx[:, -self.config.context_length:]`) — older tokens silently
fall out of view. This means very long generations gradually "forget" the
beginning of the prompt.

## Loading a checkpoint for inference

Same idea as Stage 5 and Stage 6: a checkpoint saved by `train_model`
contains `model_config` (a `GPTConfig`) and `model_state_dict` (the trained
weights). Loading it for generation means: rebuild a `GPT` from the saved
config, load the saved weights into it, and call `.eval()` — no optimizer,
no loss, no training data needed. Stage 6 already has exactly this logic in
`src/llm_from_scratch/finetune/checkpoint.py`'s `load_pretrained_model`;
Stage 7 reuses that function rather than duplicating it.

## From token IDs back to text

The reverse of encoding: once generation produces a final list of token
ids (prompt ids + newly generated ids), Stage 1's tokenizer
(`BPETokenizer.decode`) turns that list back into a string. Same
tokenizer instance used to encode the prompt — encoding and decoding must
use the same learned vocabulary and merges, or the ids won't mean anything.

## Why `torch.no_grad()` is used during inference

Every tensor operation inside `GPT.forward` normally also builds up a
computation graph, so that `loss.backward()` can later compute gradients
from it — that bookkeeping is what makes training possible. Generation
never calls `.backward()`, so building that graph is pure waste: it costs
extra memory and compute for something never used. `torch.no_grad()` tells
PyTorch to skip building the graph entirely. `GPT.generate` is already
decorated with `@torch.no_grad()` for exactly this reason.

## Worked example: 2 steps of generation

Say the prompt is `"Once upon a"`, and (for illustration) it tokenizes to
ids `[15, 372, 9]`. `context_length` is large enough that this all fits.

**Step 1:**
- Forward pass on `[15, 372, 9]` → logits at the last position, shape
  `(vocab_size,)`.
- Suppose (illustrative numbers) the logits, after softmax, give `"time"`
  the highest probability, say `0.41`.
- Greedy: pick `"time"` → id `88`. Sequence becomes `[15, 372, 9, 88]`.

**Step 2:**
- Forward pass on `[15, 372, 9, 88]` → logits at the *new* last position
  (the model now has one more token of context than in step 1, so this
  prediction can differ from what it would have guessed after `"a"` alone).
- Suppose `","` now has the highest probability, `0.33`.
- Greedy: pick `","` → id `4`. Sequence becomes `[15, 372, 9, 88, 4]`.

After `max_new_tokens = 2`, generation stops. Decoding
`[15, 372, 9, 88, 4]` back to text gives `"Once upon a time,"`. With
temperature sampling instead of greedy, step 1 or step 2 could instead have
picked a lower-probability token (e.g. `"night"` instead of `"time"`),
producing a different continuation from the same prompt.

## Simplification note

Production inference systems typically add: an end-of-text token so
generation can stop itself instead of relying only on a fixed
`max_new_tokens` (implemented — see docs/eos-generation-stopping.md, added
after this stage), top-k / top-p (nucleus) sampling to restrict sampling to
only the most plausible tokens (avoiding the very long tail temperature
alone can still pick from), repetition penalties, beam search, and batched
KV-caching for speed. We implement greedy decoding, temperature sampling,
and EOS-based stopping — enough to see generation work and to see what
temperature does — and skip the rest to keep the loop easy to read end to
end.

## What we build here

A `generate` function that loads a checkpoint (reusing Stage 6's
`load_pretrained_model`), encodes a prompt with Stage 1's tokenizer,
runs the predict-append loop with a choice of greedy or temperature
sampling, decodes the result back to text, and a filled-in
`scripts/generate.py` that wires a `--checkpoint`, `--prompt`,
`--max-new-tokens`, and `--temperature` CLI around it. No new model code
is planned — `GPT.forward` and the context-window handling in
`GPT.generate` are reused; the addition is a sampling-capable variant of
the generation loop plus the script.

## Status: implemented and tested

`GPT.generate` (Stage 3, `src/llm_from_scratch/model/gpt.py`) was extended
in place rather than replaced: it now takes an optional `temperature`
parameter. `temperature <= 0` (the default) keeps the original greedy
`argmax` behavior exactly; `temperature > 0` divides logits by temperature,
softmaxes, and samples with `torch.multinomial`. Context-length truncation
and `@torch.no_grad()` are unchanged from Stage 3.

New code:

- `src/llm_from_scratch/generate/inference.py` — `generate(model,
  tokenizer, prompt, max_new_tokens, temperature=0.0, device="cpu") -> str`:
  encodes the prompt (Stage 1's `BPETokenizer.encode`), calls
  `GPT.generate`, decodes the result (`BPETokenizer.decode`). No new model
  or checkpoint logic — reuses Stage 6's `load_pretrained_model` for
  loading and Stage 3's `GPT.generate` for the loop.
- `scripts/generate.py` — CLI with `--checkpoint`, `--prompt`,
  `--max-new-tokens` (default 50), `--temperature` (default 0.0, greedy).

Update (tokenizer persistence milestone): the tokenizer-retraining
limitation above is resolved. `scripts/generate.py` now calls
`load_tokenizer_for_checkpoint` (see docs/01-tokenization.md, "Tokenizer
persistence") to load the exact tokenizer saved next to the checkpoint by
`train_model`, instead of retraining one from `data.raw_path`. This also
means `--config` is no longer needed by this script at all — it was only
ever there to locate the raw corpus for retraining. Pointing it at a
checkpoint saved before this milestone (no `tokenizer.json` present) now
fails immediately with an explicit error rather than silently retraining
a possibly-different tokenizer.

Tests: `tests/test_model.py` adds `test_generate_greedy_is_deterministic`
(same prompt, same output, twice) and
`test_generate_temperature_sampling_produces_valid_token_ids` (sampled
output still has valid ids and shape); existing
`test_generate_extends_sequence_with_valid_token_ids` and
`test_generate_respects_context_length_when_prompt_exceeds_it` continue to
cover `max_new_tokens` and context-length truncation. `tests/test_generate.py`
(new) covers the `generate()` function: successful decode to a string,
`max_new_tokens` respected (checked at the `GPT.generate` call boundary,
not by re-encoding decoded text — see note below), long-prompt truncation,
and reuse of `load_pretrained_model` for checkpoint loading. Full suite:
`50 passed` (5 tokenizer + 10 data + 9 model + 9 train + 8 eval + 5
finetune + 5 generate).

**Bug found and fixed while implementing this stage**: `BPETokenizer.decode`
(Stage 1) called `bytes.decode("utf-8")` with no error handling. This never
mattered before Stage 7, because every prior stage only ever decoded ids
produced by `encode(real_text)`, which always round-trips to valid UTF-8.
Generation is the first stage to decode a model's own *output* ids — and an
early-training or randomly-initialized model can legally emit a raw byte
token (one of the base 256) that isn't valid UTF-8 on its own. Fixed with
`errors="replace"` in `BPETokenizer.decode`; this doesn't change behavior
for any valid input (encode/decode round-trips are unaffected), it only
stops decode from crashing on generated output.

Manual CLI smoke test: trained a tiny checkpoint (`vocab_size=300`,
`context_length=16`, `n_layer=2`, `n_embd=32`, 45,184 params) on a small
repeated hand-written corpus for 60 steps (train_loss 5.61 → 5.31), then
ran `scripts/generate.py` against it with prompt `"The fox"`. Both greedy
(`--temperature 0.0`, default) and temperature sampling (`--temperature
0.8`) ran end to end, produced 30 new tokens each, and decoded without
crashing (some `�` replacement characters appeared, from the
`errors="replace"` fix above — expected at this undertrained scale).
Greedy output was not coherent English — expected, since 60 steps on ~2.7KB
of repeated text is far too little for a small model to learn much; the
point of the smoke test is verifying the generation mechanics (checkpoint
loading, prompt encoding, the predict-append loop, context truncation,
decoding) work correctly end to end, not output quality, which needs a
much larger pretraining run to assess meaningfully (same caveat as Stage
6's fine-tuning smoke test).

Next: none — Stage 7 implemented and tested. No further stages planned in
`docs/00-roadmap.md` as of this session.

Update (EOS / generation stopping milestone): `GPT.generate` gained an
optional `eos_token_id` parameter — generation now stops the moment that id
is produced, before `max_new_tokens` if it happens sooner. `generate()`
(`src/llm_from_scratch/generate/inference.py`) passes the tokenizer's
`eos_token_id` through automatically; a tokenizer with no EOS (any
checkpoint predating this milestone) passes `None`, reproducing the exact
prior fixed-length behavior with no error. See
docs/eos-generation-stopping.md for the full design and the "Simplification
note" above, which is now partially resolved (the end-of-text token
described there as production-only now exists here too).
