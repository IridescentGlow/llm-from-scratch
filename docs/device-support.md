# Device support (cross-cutting milestone)

Not a new stage — this touches Stages 4–7 equally (pretraining, evaluation,
fine-tuning, generation), the same way the tokenizer persistence milestone
did. See `docs/01-tokenization.md` for that precedent.

## What this is

Every stage already computes on tensors and a model. A **device** is *where*
those tensors and that model's numbers physically live and get computed —
either the computer's CPU (regular processor cores) or a GPU (a separate
chip built for doing huge numbers of simple math operations, like matrix
multiplication, in parallel). This milestone lets you choose which one to
use from the command line, instead of always silently using the CPU.

## Why it matters

Nothing about the model changes. What changes is speed: the same training
run that takes minutes on a GPU can take hours or days on a CPU, because a
transformer forward/backward pass is almost entirely matrix multiplication —
exactly what a GPU is built to do many of at once. Without device support,
this project is stuck at CPU speed even on a machine with a GPU sitting
right there unused.

## CPU vs. GPU for LLM workloads

A CPU has a handful of powerful cores, good at doing one complex thing after
another quickly. A GPU has thousands of small, simple cores, good at doing
the *same* simple operation on thousands of numbers simultaneously. Training
a transformer is mostly: multiply this big matrix of weights by this big
matrix of activations, over and over. That's exactly the "same simple
operation, thousands of times" shape a GPU is built for — which is why GPUs
give large speedups for this specific kind of workload, not because they're
"faster computers" in general.

## CUDA in simple terms

**CUDA** is NVIDIA's software layer that lets PyTorch send tensor operations
to an NVIDIA GPU and get results back. When code says `.to("cuda")`, it
means "move this tensor/model's numbers onto the GPU, and do all further
math on it there, via CUDA." PyTorch also supports **MPS** (Apple's GPU
backend, for Apple Silicon Macs) the same way, via `.to("mps")`. If neither
is available, `.to("cpu")` is always valid — every operation in PyTorch that
runs on GPU is also implemented to run on CPU.

## Tensors and the model must be on the same device

PyTorch does not automatically move things around. If the model's weights
are on the GPU but the input tensor is still on the CPU, `model(input)`
raises an error — mismatched devices, not something PyTorch silently
reconciles. Whenever the device changes, *both* the model and every tensor
fed into it must be explicitly moved.

### Worked example

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x.device)  # cpu

if torch.cuda.is_available():
    x = x.to("cuda")
    print(x.device)  # cuda:0

model = torch.nn.Linear(3, 1)
model.to(x.device)      # move the model to wherever x now lives
y = model(x)             # works: both on the same device
```

If `model.to(x.device)` were skipped, `model(x)` would raise
`RuntimeError: Expected all tensors to be on the same device`.

## How this project's `device` parameters currently work

The device plumbing already exists, quietly, in four places — all of them
default to `"cpu"` and none of them are currently exposed to the user:

- `train_model(..., device: str = "cpu")` and its helper `estimate_loss`
  (`src/llm_from_scratch/train/loop.py`) — call `model.to(device)` once, and
  `.to(device)` on every batch's `input_ids`/`target_ids`.
- `evaluate_model(..., device: str = "cpu")`
  (`src/llm_from_scratch/eval/metrics.py`) — same pattern.
- `generate(..., device: str = "cpu")`
  (`src/llm_from_scratch/generate/inference.py`) — same pattern, on the
  prompt tensor.

`GPT` itself (`src/llm_from_scratch/model/gpt.py`) has no device logic of
its own — it's a plain `nn.Module`, and `positions = torch.arange(seq_len,
device=idx.device)` already follows whatever device the *input* is on. This
is why moving the model and its inputs is enough; nothing inside `GPT` needs
to change.

## Why the CLI scripts still effectively default to CPU

`scripts/train.py`, `scripts/evaluate.py`, `scripts/finetune.py`, and
`scripts/generate.py` call `train_model`, `evaluate_model`, and `generate`
without ever passing `device=`, so the `"cpu"` default is always what runs —
regardless of what hardware is available. The device *parameter* exists;
the *CLI option* to set it does not. There's also no automatic detection
anywhere — even a machine with a CUDA GPU currently trains on CPU unless a
caller of these functions (not currently any script) explicitly passes
`device="cuda"`.

One more spot this milestone touches: `load_pretrained_model` and
`torch.load(checkpoint_path, weights_only=False)`
(`src/llm_from_scratch/finetune/checkpoint.py`, used by
`scripts/evaluate.py`, `scripts/finetune.py`, `scripts/generate.py`) loads
a checkpoint's tensors onto whatever device they were *saved* from, by
default. A checkpoint saved on a CUDA machine can fail to load at all on a
CPU-only machine unless `map_location` is set. This isn't a new bug
introduced here, but device support is exactly the moment to fix it,
since it's the same "what device are we actually on" question.

## Target behavior

- **Automatic selection** (no `--device` given): try CUDA first, then MPS,
  then fall back to CPU — in that order, since CUDA and MPS are almost
  always faster than CPU when available.
- **Explicit selection** (`--device cpu` / `--device cuda` / `--device mps`
  on every relevant script): use exactly what was asked for.
- **Explicit request for an unavailable device**: fail immediately with a
  clear error (e.g. `--device cuda` on a machine with no CUDA GPU) — never
  silently fall back to something else. Silent fallback would hide a
  genuine problem (e.g. "why is this so slow" turning out to mean "it's
  been on CPU the whole time").
- **CPU always works**: every operation used here has a CPU implementation,
  training/inference correctness never depends on which device is used —
  only speed does. CPU must stay fully supported, both as the automatic
  fallback when no GPU is present and as an explicit choice (for debugging,
  small smoke tests, or machines without a GPU at all).

## What this milestone does *not* change

No change to the model's architecture, the training math, the loss, or any
existing function's behavior beyond *which device it runs on*. The `device`
parameters already on `train_model`, `estimate_loss`, `evaluate_model`, and
`generate` are reused as-is — this milestone adds device *selection* (a
shared helper + `--device` on each CLI script) on top of plumbing that
already exists, not a redesign.

## Simplification note

Production training setups also handle multi-GPU (data/model parallelism)
and mixed precision (fp16/bf16) to use GPU memory and bandwidth more
efficiently. This milestone only handles *single-device* selection — one
CPU, one GPU, or one MPS device — matching this project's existing
single-device training loop (see docs/04-pretraining.md's simplification
note).

## Status: implemented and tested

- `src/llm_from_scratch/device.py` — `resolve_device(requested: str | None =
  None) -> str`: `None` auto-detects (CUDA -> MPS -> CPU); an explicit
  `"cpu"`/`"cuda"`/`"mps"` is validated and returned as-is, or raises
  `RuntimeError` with a clear message if that device isn't available, or
  `ValueError` for an unrecognized device string. No changes to `GPT` or to
  any of the four existing `device` parameters (`train_model`,
  `estimate_loss`, `evaluate_model`, `generate`) -- this only decides what
  string gets passed to them.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/finetune.py`,
  `scripts/generate.py` all gained `--device {cpu,cuda,mps}` (default: not
  given, i.e. auto-detect), call `resolve_device`, print the resolved
  device, and pass it through to `train_model`/`evaluate_model`/`generate`.
  `scripts/finetune.py` also calls `model.to(device)` explicitly right
  after loading the checkpoint, and builds its reused `prompt_ids` tensor
  on `device`, since it runs generation both before and after
  `train_model` -- not just handled by one function call like the other
  three scripts.
- `load_pretrained_model` (`src/llm_from_scratch/finetune/checkpoint.py`)
  now loads checkpoints with `map_location="cpu"` always, so a checkpoint
  saved from a CUDA run still loads on a CPU-only machine; the caller's own
  `device=` argument to `train_model`/`evaluate_model`/`generate` moves the
  model to the actually-requested device afterward.

Tests in `tests/test_device.py` (8, all passing) cover: auto-detect falls
back to CPU when nothing else is available, auto-detect prefers CUDA over
MPS and MPS over CPU, explicit `"cpu"` always succeeds, explicit
`"cuda"`/`"mps"` succeed when available and raise a clear `RuntimeError`
when not, and an unrecognized device string raises `ValueError`. (All via
`monkeypatch` on `torch.cuda.is_available` / `torch.backends.mps.is_available`,
since the test machine itself is CPU-only.) Full suite: `64 passed`
(previous 56 + 8 new).

Manual end-to-end smoke test: trained a tiny checkpoint with
`scripts/train.py --device cpu` (vocab_size 300, context_length 16,
n_layer 2, n_embd 32, 45,184 params); ran `scripts/evaluate.py --device
cpu` and `scripts/generate.py` (no `--device`, auto-detect) against it
successfully, both printing `Using device: cpu`; ran `scripts/finetune.py
--device cpu` end to end (before/after val loss + generation, fine-tuned
checkpoint saved); ran `scripts/generate.py --device cuda` on this CPU-only
machine and confirmed it fails immediately with `RuntimeError: Requested
device 'cuda' but no CUDA GPU is available.` instead of silently falling
back to CPU -- direct confirmation of the "no silent fallback" requirement.
