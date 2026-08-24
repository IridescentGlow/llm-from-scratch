# Reproducibility / seeding (cross-cutting milestone)

Not a new stage — this adds an opt-in `--seed` flag to the existing
pretraining, fine-tuning, and generation scripts, the same way device
support and checkpoint/resume extended earlier stages without becoming a
new numbered stage.

## What a random seed is

Every "random" number a computer generates is actually produced by a
deterministic formula, starting from some internal state. A **random seed**
is the starting value for that state. Give the same seed to the same
sequence of operations, and you get the exact same sequence of "random"
numbers back, every time. That's the whole trick: it turns "unpredictable"
into "reproducible, but still looks random."

## Why this project currently produces different runs from the same starting conditions

Nothing in `scripts/train.py`, `scripts/finetune.py`, or
`scripts/generate.py` sets a seed today. Every one of those "random"
number generators (RNGs) is left at whatever state PyTorch/NumPy/Python
happened to initialize it to when the process started — which differs
run to run. So two runs of `python scripts/train.py --config
configs/small.yaml` on the identical corpus produce different model
weights, different training trajectories, and different generated text,
even though every input file was the same.

## The three important randomness sources here

- **Model initialization** — `GPT.__init__`
  (`src/llm_from_scratch/model/gpt.py`) builds `nn.Embedding`, `nn.Linear`,
  and `nn.LayerNorm` layers with no custom weight-init code, so PyTorch's
  default init draws from the global RNG. This is where two runs first
  diverge — before a single training step has happened.
- **Dropout** — during training (not `.eval()` mode), `CausalSelfAttention`
  and `FeedForward` (`src/llm_from_scratch/model/gpt.py`) randomly zero out
  some activations on every forward pass (`configs/small.yaml` sets
  `dropout: 0.1`). Each forward pass draws fresh randomness, so even two
  runs that started from identical weights would immediately diverge once
  training began.
- **DataLoader shuffling** — `get_dataloader(..., shuffle=True)`
  (`src/llm_from_scratch/data/dataset.py`), the default `train_model` uses
  for the training set, draws a new random batch order from the global RNG
  every epoch. Validation already uses `shuffle=False` and is unaffected.

## Why generation with temperature > 0 is intentionally random

`GPT.generate` (`src/llm_from_scratch/model/gpt.py`) has two modes.
`temperature <= 0` (the default) always picks the single highest-probability
next token — already fully deterministic, seed or no seed. `temperature > 0`
samples from the probability distribution instead of always taking the top
pick, which is the entire point of a temperature setting: it's what gives
generated text variety across repeated runs of the same prompt. Seeding
doesn't remove that randomness — it makes it *reproducible* randomness,
so the same seed always samples the same "random" choices.

## What `--seed` guarantees in this project

Given an explicit `--seed N`, and rerun on the same device (e.g. always
CPU, or always the same GPU) with the same PyTorch/library versions: model
initialization, dropout, and training-batch order will be identical run to
run, so two `--seed 42` training runs on the same data produce identical
final weights. Seeded generation with `temperature > 0` will sample the
same tokens each time. That's the scope of the guarantee — see the caveats
below for what it does *not* cover.

## Why `--seed` defaults to `None`

`--seed` is opt-in, not on by default. Omitting it keeps today's existing
behavior exactly as-is (genuinely random init/dropout/shuffling each run),
which is often what you actually want while exploring — e.g. training the
same config several times to see the range of outcomes, not the same
outcome five times. Reproducibility is something you ask for when you
need it (debugging, comparing a code change against a baseline, writing
up a result), not a constraint applied to every run whether wanted or not.

## Why reproducibility is generally strongest when rerunning on the same device type/software environment

A seed fixes the *starting state* of an RNG, but the exact sequence of
numbers that RNG produces after that can depend on the hardware and
software actually running the computation — different PyTorch versions,
different CPU vs. GPU kernels, even different thread counts can produce
tiny floating-point differences that compound over many training steps.
Same seed, same machine, same library versions is the strongest guarantee;
same seed on a different GPU model or a different PyTorch version is not
promised to match bit-for-bit, even though it'll be close and will use the
same *logical* sequence of random choices.

## CUDA/MPS nondeterminism caveats in simple terms

Some GPU operations (certain reductions and atomic-add-based ops) are
allowed to add numbers back together in a different order between runs,
purely for speed — and floating-point addition isn't perfectly
order-independent, so this can produce tiny numerical differences even
with an identical seed on identical hardware. Removing this entirely is
possible (`torch.use_deterministic_algorithms(True)`, plus extra
environment configuration) but costs real performance and can make some
operations raise an error if they have no deterministic implementation.
This milestone does not turn that on — see "What this milestone does not
do" below. `--seed` still narrows GPU runs down to "same logical random
choices, tiny possible floating-point drift," which is a large
improvement over today's "completely unseeded."

## Why resume is not promising exact RNG-stream continuation

`--resume` (`docs/checkpoint-resume.md`) restores model weights, optimizer
state, and step count — but not the exact internal RNG state at the moment
of interruption, which was never saved to the checkpoint and isn't in
scope here. So a training run interrupted at step 4000 and resumed will
not draw the exact same dropout/shuffle sequence a single uninterrupted
run would have from step 4000 onward. This doesn't break anything resume
already promises (loss continues smoothly, no restart spike) — it only
means "interrupt-and-resume with `--seed 42`" isn't bit-identical to
"train straight through with `--seed 42`." Promising true continuation
would mean saving and restoring full RNG state in the checkpoint, which is
a bigger, separate feature this milestone deliberately doesn't take on.

## Worked example: same seed vs. no seed

Two runs of `python scripts/train.py --config configs/small.yaml --seed 42`
on the same corpus: identical initial weights (checked before step 0),
identical train_loss at every logged step, identical final checkpoint.

Two runs of `python scripts/train.py --config configs/small.yaml` (no
`--seed`): different initial weights, and the train_loss values diverge
from step 1 onward — not necessarily wildly different in magnitude, but
not numerically identical either.

## What this milestone does *not* do

No `torch.use_deterministic_algorithms(True)` or CUDA determinism
environment variables (real performance cost, out of scope — see the CUDA
caveats above). No changes to `DataLoader` internals, model architecture,
optimizer, or generation logic — seeding only changes what RNG state
those already-existing random draws start from. No RNG-state saving in
checkpoints, so `--resume` continuations are not bit-reproducible (see
above). No seed for `scripts/evaluate.py` — evaluation is forward-only,
already uses `shuffle=False`, and runs in `.eval()` mode (dropout
disabled), so it's already fully deterministic given a fixed checkpoint
and no seed is needed.

## Status: implemented and tested

`src/llm_from_scratch/seed.py` has `set_seed(seed: int) -> None`, seeding
`random.seed`, `numpy.random.seed`, `torch.manual_seed`, and
`torch.cuda.manual_seed_all` (a no-op when no CUDA device is present, so
it's always safe to call). `scripts/train.py`, `scripts/finetune.py`, and
`scripts/generate.py` each gained `--seed` (default `None`); when given,
`set_seed(args.seed)` is called immediately after device resolution and
before any model or data construction, so the very first random draw
(weight init) is covered. `scripts/evaluate.py` is unchanged (already
fully deterministic, no seed needed). No changes to `DataLoader`, `GPT`,
`TrainConfig`, or any config file — seeding is entirely a
call-it-before-anything-else concern.

Tests added in `tests/test_seed.py` (7 new):
`test_same_seed_produces_identical_model_init`,
`test_different_seeds_produce_different_model_init`,
`test_same_seed_produces_identical_training_result` (two full tiny
training runs with `dropout=0.2`, asserting identical `train_losses` and
identical final parameters),
`test_different_seeds_produce_different_training_result`,
`test_greedy_generation_is_deterministic_regardless_of_seed`,
`test_seeded_stochastic_generation_is_reproducible`,
`test_different_seeds_can_produce_different_stochastic_generation`. Full
suite: `90 passed` (previous 83 + 7 new).

Manual end-to-end smoke test (real script runs, not just unit tests):
trained a tiny checkpoint (vocab_size 300, context_length 16, n_layer 2,
n_embd 32, dropout 0.1) on a varied synthetic corpus via
`scripts/train.py --device cpu`. Ran it twice with `--seed 42` against two
separate checkpoint directories: logged train/val loss at steps 10/20/30
were identical between the two runs, and a direct tensor-level diff of
both saved checkpoints' `model_state_dict`s confirmed every parameter
tensor was bit-identical (`torch.equal` true for every key). A third run
with `--seed 999` (same config, same data) diverged from step 10 onward
and its final `model_state_dict` was confirmed *not* equal to the
`--seed 42` runs'. Separately, ran `scripts/generate.py --temperature 1.0
--seed 7` against the same checkpoint twice and got byte-identical output
both times; ran `scripts/generate.py --temperature 0.0` (greedy, no
`--seed`) twice and got identical output both times (already deterministic
pre-milestone); ran `scripts/generate.py --temperature 1.0` twice with no
`--seed` and got two different outputs, confirming unseeded stochastic
generation still varies run to run exactly as before this milestone.
