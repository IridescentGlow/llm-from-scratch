# EOS / natural generation stopping (cross-cutting milestone)

Not a new stage — this extends Stage 1 (tokenizer), Stage 6 (fine-tuning),
and Stage 7 (generation), the same way the tokenizer persistence, device
support, and checkpoint/resume milestones extended existing stages without
becoming a new numbered one. See `docs/01-tokenization.md`,
`docs/06-finetuning.md`, and `docs/07-generation.md` for the code this
builds on.

## What an EOS token is

**EOS** stands for "end of sequence." It's a single reserved token id whose
only job is to mean "stop — this response is finished," the same way a
period means "sentence's over" to a human reader, except the model has to
*produce* this one on purpose, as an actual output choice, not just imply it.
It doesn't correspond to any letter, byte, or word. It's a marker glued onto
the vocabulary purely so the model has something concrete to predict when
it's done talking.

## Why the model currently always generates exactly `max_new_tokens`

`GPT.generate` (`src/llm_from_scratch/model/gpt.py`) is a fixed-count loop:
predict one token, append it, repeat, `max_new_tokens` times, then stop —
full stop, no matter what was actually generated. Nothing in the loop ever
asks "is this a good place to stop?" It doesn't know what "done" means,
because nothing has ever told it. Every previous generation, however
sensible the answer, keeps going until the counter runs out, often trailing
into a repeated phrase, a fabricated new "Instruction:" block, or just
padding-like noise (see the manual smoke test in `docs/07-generation.md`,
"Status").

## Why an explicit stopping token makes responses usable

For a raw text-completion model, running out of tokens is fine — there was
never a "right length" to hit. But for an instruction-following response,
there *is* a right length: exactly as long as the answer, then done. Without
a way to signal that, every generated response is either cut off mid-thought
(if `max_new_tokens` is too small) or padded out with garbage past its
natural end (if it's too large) — and there's no single safe setting that
works for both a one-word answer and a three-paragraph one. An explicit
stopping token lets the model choose the right length itself, per response.

## How special tokens differ from normal learned BPE tokens

Every ordinary token in this project's vocabulary — the 256 base bytes plus
every merge `BPETokenizer.train()` learns — exists because it showed up in
real training text. `encode()` can always produce it, `decode()` always
turns it back into the same bytes it came from, and it means "one of these
input characters."

EOS is different in every one of these ways:

- It's **not learned from data** — no corpus, no merge counting, no
  frequency statistics. It's just the next unused integer, assigned once,
  by code, not by training.
- It **never appears in real text.** `encode()` on ordinary text can never
  produce the EOS id, because it isn't the result of any merge — there's no
  byte pair that maps to it.
- It **has no bytes to decode.** `decode()` can't turn it back into UTF-8
  text the way every other token can, because it doesn't stand for any.
  `decode()` needs a special case: skip it (or render nothing) rather than
  trying to look up bytes that don't exist for it.
- Its meaning is **assigned, not discovered** — "id N means stop" is a
  decision this project makes once, not a pattern found in a corpus.

## How the EOS token should be represented and persisted

`BPETokenizer.save()`/`load()` (`src/llm_from_scratch/tokenizer/bpe.py`)
already writes a `special_tokens` field to `tokenizer.json` — reserved,
and always empty, since the tokenizer persistence milestone anticipated
exactly this (`docs/01-tokenization.md`, "The vocabulary... special-token
metadata"). This milestone is the first to actually put something in it:
`{"eos": <id>}`. No new file format, no second save/load path — the exact
same `tokenizer.json` a checkpoint directory already carries just gains one
populated key it was always designed to hold. `load_tokenizer_for_checkpoint`
(`src/llm_from_scratch/finetune/checkpoint.py`) needs no changes to keep
loading it correctly.

The id itself: the next unused integer after the tokenizer's trained BPE
vocabulary — i.e. `vocab_size` at the moment EOS is added (256 base bytes +
however many merges `train()` learned). Assigned once, right after
`train()` finishes and before the tokenizer is saved, never recomputed.

## Why the EOS token must have a stable id

This is the exact same reasoning as full tokenizer persistence
(`docs/01-tokenization.md`, "Why a model must keep the exact tokenizer it
was trained with"): the model's embedding table has one row per vocabulary
id, and during training that row's meaning is *whatever id happens to point
to it*. If EOS were reassigned a different id between training runs, the
model's actual learned "stop" row would silently correspond to a different,
wrong id — the model would keep confidently emitting its old EOS id, which
no longer means "stop" to anything reading the tokenizer, and never emit
the new one. The id must be fixed once, at the tokenizer's creation time, and
saved — exactly like every other token id already is.

This also means EOS has to be decided **at pretraining time**, not
fine-tuning time. `GPTConfig.vocab_size` (Stage 3) fixes the size of the
model's embedding table and output layer the moment the model is built —
there's no row reserved for a token that didn't exist yet. Adding EOS after
a model is already pretrained would mean resizing weight matrices that
already have learned values in every existing row, which is a real
architecture change and out of scope for this milestone (see "Design
constraints" below). So: `scripts/train.py` trains BPE, then adds EOS,
*then* builds `GPTConfig(vocab_size=tokenizer.vocab_size, ...)` — one extra
row exists in the embedding table and output layer from the very first
pretraining step onward, even though nothing in the pretraining corpus ever
produces that id. Fine-tuning inherits this tokenizer (and its EOS id)
completely unchanged, the same way it already inherits everything else via
`load_tokenizer_for_checkpoint`.

## How fine-tuning examples should use EOS

Today, `build_corpus` (`src/llm_from_scratch/finetune/data.py`) joins
formatted examples into one text blob, which is then tokenized as a single
string. EOS can't be inserted that way — it isn't a string, it has no bytes,
so it can't be written into a text corpus and picked up by `encode()`.

Instead, each example needs to be **encoded separately**, with the EOS id
appended to its own token id list, and *those* token id lists concatenated
— not the raw text. So the fine-tuning data pipeline changes from
"build one big string, tokenize once" to "tokenize each example, append EOS,
then concatenate ids." Everything downstream (`write_token_ids`,
`train_val_split`, `TokenDataset`) is untouched — it already just consumes a
flat array of token ids, and doesn't care how they were assembled.

If the checkpoint's tokenizer has no EOS id (see "Compatibility" below),
this step should fail clearly rather than silently skip appending EOS —
producing fine-tuning data that's supposed to teach "stop here" but doesn't,
with no error, would be a silent correctness bug, exactly the kind this
project's other milestones have refused to allow.

## How generation detects EOS and stops

`GPT.generate`'s loop already picks one `next_token` per step. Add an
optional `eos_token_id` parameter: after picking `next_token`, if it equals
`eos_token_id`, stop the loop immediately — the same way the loop already
stops when the step counter runs out, just possibly sooner. `generate()`
(`src/llm_from_scratch/generate/inference.py`) passes the tokenizer's EOS id
(if it has one) straight through. `decode()` already needs to skip
non-text token ids (see "special tokens" above), so the EOS id at the end
of the generated sequence is simply dropped, not rendered as `�` or
anything else.

## What happens if EOS never appears

Nothing goes wrong — generation just runs the full `max_new_tokens` and
stops the way it always has. An undertrained model, or a prompt that
doesn't resemble an instruction/response shape, may never predict EOS as
the highest-scoring (or sampled) token. This is expected, not an error
condition, and `max_new_tokens` existing as a hard cap regardless of EOS is
exactly what makes that safe (see next section).

## How this interacts with `max_new_tokens`

`max_new_tokens` stays a **hard safety cap**, unconditionally, even with EOS
enabled — generation stops at whichever comes first: EOS is produced, or
`max_new_tokens` tokens have been generated. This matters because nothing
guarantees a model — especially a small, lightly-trained one — ever learns
to predict EOS reliably; without the cap still in force, a model that never
predicts EOS would generate forever. EOS is a way to stop *early*, never a
replacement for the existing limit.

## How existing tokenizer persistence makes this possible

None of this needs a new save format, a new checkpoint field, or a new
loading path. `tokenizer.json`'s `special_tokens` slot already exists
specifically so an addition like this wouldn't need one
(`docs/01-tokenization.md`). The only genuinely new thing on disk is one
populated key (`{"eos": <id>}`) inside a structure that was already being
written and read.

## Worked example

Fine-tuning example, after formatting (`docs/06-finetuning.md`'s template):

```
Instruction: Translate 'good morning' to French.
Response: Bonjour.
```

Suppose this encodes (illustrative ids) to
`[15, 372, 9, 4, 88, 2, 501, 9, 4, 12, 3]`, and the tokenizer's EOS id is
`300` (one past its trained BPE vocab). The fine-tuning token stream for
this example becomes:

```
[15, 372, 9, 4, 88, 2, 501, 9, 4, 12, 3, 300]
                                          ^^^ EOS appended
```

Fine-tuning trains on this exactly like any other next-token sequence — the
model sees `...9, 4, 12, 3` and learns that `300` (EOS) is a good next
prediction right there, at the natural end of the response, alongside
whatever else it learns from the surrounding examples.

At generation time, prompting with `"Instruction: Translate 'good
afternoon' to French.\nResponse:"` and generating step by step: the model
predicts `Bon`, `jour`, `.`, then — if it has learned the pattern — `300`
(EOS). The moment `300` is predicted, the loop stops, even though
`max_new_tokens` (say, 50) was never reached. Decoding the result skips id
`300` (it has no bytes), producing exactly `"Bonjour."` with nothing trailing
after it.

## Compatibility implications for existing checkpoints/tokenizers

Any `tokenizer.json` saved before this milestone has an empty
`special_tokens` dict — no `"eos"` key at all. Two different situations,
handled two different ways, per the design constraint that we neither
silently invent an id nor pretend everything still works uniformly:

- **Generation** (`scripts/generate.py`) against a pre-EOS checkpoint: no
  error. `tokenizer.eos_token_id` is simply absent, so `GPT.generate` gets
  no `eos_token_id` to compare against, and behaves exactly as it does
  today — generates the full `max_new_tokens`, nothing more, nothing less.
  This isn't a silent correctness bug the way tokenizer-corpus mismatch was
  (`docs/01-tokenization.md`) — there's no wrong output being produced,
  just a capability (early stopping) that isn't available, because the
  metadata to support it doesn't exist. Documented as legacy behavior, not
  an error.
- **Fine-tuning** (`scripts/finetune.py`) against a pre-EOS checkpoint: this
  *should* fail clearly. Fine-tuning's entire point, for this milestone, is
  to teach the model when to stop — silently fine-tuning without EOS
  wouldn't be "graceful degradation," it would be a fine-tuning run that
  looks like it's doing its job but isn't. So the token-id-building step
  raises a clear error (e.g. "this checkpoint's tokenizer has no EOS token;
  retrain pretraining with the current `scripts/train.py` to get one")
  rather than quietly building a corpus with no stopping signal at all.

Existing pretrained checkpoints (all from tiny smoke tests, per the other
milestones' status notes) aren't migrated — same precedent as tokenizer
persistence and checkpoint/resume: regenerate them with the current
`scripts/train.py`, which will add EOS as part of building the tokenizer.

## Design constraints (restated, to confirm before implementing)

- Reuse `tokenizer.json`'s existing `special_tokens` field — no second
  format.
- Exactly one special token type (EOS) — no padding token, no unk token,
  no BOS, unless a real need shows up.
- Normal BPE behavior for ordinary text is untouched — `encode()`/`decode()`
  on real text round-trip exactly as before; EOS never interferes because
  it's never reachable from real byte sequences.
- Greedy and temperature sampling both keep working exactly as today — EOS
  detection is a check *after* a token is already chosen (by either
  strategy), not a change to how a token gets chosen.
- `max_new_tokens` remains a hard cap regardless of EOS.
- No silent invention of an EOS id for a tokenizer that doesn't have one —
  generation degrades gracefully (documented above); fine-tuning fails
  clearly instead.
- No padding tokens, no attention masks for batching, no chat templates —
  out of scope, not touched by this milestone.

## What this milestone does *not* do

No batched generation with padding, no chat-style multi-turn templates, no
BOS/pad/unk tokens, no retroactive EOS support for checkpoints whose
embedding table was already sized without it (that would require resizing
trained weights — a real architecture change, not this milestone's scope),
no top-k/top-p sampling changes (`docs/07-generation.md`'s simplification
note still applies for everything except the stopping condition).

## Status: implemented and tested

Implemented exactly as planned above, with the confirmed compatibility
policy: pre-EOS checkpoints generate normally (no early stop, no error);
pre-EOS checkpoints fail loudly at fine-tuning time; no silent migration or
invented EOS ids anywhere.

- **`src/llm_from_scratch/tokenizer/bpe.py`**: `BPETokenizer` gained a real
  `self.special_tokens: dict[str, int]` attribute (previously written to
  disk as always `{}` and never read back); `add_eos_token() -> int`
  assigns `id = self.vocab_size` (base vocab + special tokens so far),
  records it as `special_tokens["eos"]`, and raises `ValueError` if called
  twice; `vocab_size` now counts `len(vocab) + len(special_tokens)`, so a
  tokenizer's reported size always includes any special tokens added; a new
  `eos_token_id` property returns `special_tokens.get("eos")` (`None` if
  never added). `save()`/`load()` round-trip `special_tokens` for real now
  (`load()` defaults to `{}` for a legacy file with the key present but
  empty, or absent entirely). `decode()` skips any id in
  `special_tokens.values()` instead of trying to look up bytes for it.
- **`scripts/train.py`**: after `tokenizer.train(...)`, calls
  `tokenizer.add_eos_token()` and reassigns
  `model_config.vocab_size = tokenizer.vocab_size` before building the
  `GPT` — so the embedding table and output layer are sized to include the
  EOS row from the very first pretraining step. No config flag: every
  pretraining run through the current script always adds EOS, keeping this
  milestone to exactly one special token type with no extra configuration
  surface.
- **`src/llm_from_scratch/finetune/data.py`**: new `build_token_ids(examples,
  tokenizer) -> list[int]` encodes each formatted example separately and
  appends `tokenizer.eos_token_id` to that example's own ids before
  concatenating across examples — raises `ValueError` ("no EOS token") if
  the tokenizer has none. `scripts/finetune.py` uses this instead of
  `build_corpus` + a single whole-corpus `encode()` call (`build_corpus` and
  `format_example` are unchanged and still used internally).
- **`src/llm_from_scratch/model/gpt.py`**: `GPT.generate` gained an optional
  `eos_token_id: int | None = None` parameter; after choosing `next_token`
  each step (greedy or temperature-sampled, unchanged), it breaks the loop
  immediately if that token equals `eos_token_id`. `max_new_tokens` is
  still the loop's cap either way — unaffected when `eos_token_id` is
  `None` or never produced.
- **`src/llm_from_scratch/generate/inference.py`**: `generate()` passes
  `tokenizer.eos_token_id` through to `model.generate()` — `None` for a
  tokenizer with no EOS, which reproduces the exact pre-milestone behavior.
  `scripts/finetune.py`'s own before/after sample generations (via
  `model.generate` directly, not the `generate()` wrapper) also now pass
  `eos_token_id=tokenizer.eos_token_id`.

Tests added: `tests/test_tokenizer.py` — `add_eos_token` assigns the next
unused id and refuses a second call; a tokenizer with no EOS reports
`eos_token_id is None`; `decode()` skips an EOS id; save/load round-trips
EOS; loading a legacy tokenizer.json (populated by an EOS-less
`BPETokenizer.save()`) yields `eos_token_id is None`, not an error.
`tests/test_model.py` — `GPT.generate` stops immediately once
`eos_token_id` is produced (verified via greedy determinism: the first
token a prompt produces with no `eos_token_id` given is reused as a
stand-in EOS id to force an early, verifiable stop) and still runs the full
`max_new_tokens` when `eos_token_id` is never produced. `tests/test_generate.py`
— `generate()` forwards the tokenizer's `eos_token_id` to `GPT.generate`
(and forwards `None` unchanged for a legacy, EOS-less tokenizer).
`tests/test_finetune.py` — `build_token_ids` appends EOS after each
example's own ids (not just once at the very end) and raises a clear
`ValueError` mentioning "no EOS token" when the tokenizer has none. Full
suite: `81 passed` (previous 69 + 12 new: 6 tokenizer + 2 model + 2 generate
+ 2 finetune).

Manual end-to-end smoke test: pretrained a tiny checkpoint (`vocab_size:
300` requested, `301` actual after EOS, `context_length=16`, `n_layer=2`,
`n_embd=32`) via `scripts/train.py --device cpu` on a small repeating
corpus; confirmed the saved tokenizer's `eos_token_id == 300` and the saved
checkpoint's embedding table has exactly `301` rows (the reserved EOS row).
Fine-tuned that checkpoint via `scripts/finetune.py` on ~90 repeated
instruction/response examples (1,500 steps, enough for this toy model to
actually learn the pattern: val loss `13.15 -> 0.46`); confirmed the
generated token stream really does contain example-ending EOS ids by
inspecting `build_token_ids`'s output directly, and confirmed
`scripts/generate.py`/direct `model.generate()` calls against the
fine-tuned checkpoint stop **before** `max_new_tokens` (`60`) is reached,
in both greedy and `--temperature 0.8` sampling, e.g. `"Instruction: Who is
lazy?\nResponse: The lazy dog.\n"` generated in `16` tokens with
`out[-1] == eos_token_id`, not `60`. Separately verified the two
compatibility paths directly: (1) hand-built a legacy tokenizer.json (real
pretrained tokenizer with `special_tokens` cleared to `{}`, simulating a
pre-milestone file) and ran `scripts/generate.py` against it — completed
successfully, no error, ran the full requested `max_new_tokens`, exactly
pre-milestone behavior; (2) ran `scripts/finetune.py` against that same
legacy checkpoint and confirmed it failed immediately with
`ValueError: This checkpoint's tokenizer has no EOS token, so fine-tuning
can't teach the model when to stop ...`, not a silent no-op or an invented
id.

Next: none — this milestone is complete. No further stages or milestones
planned as of this session.
