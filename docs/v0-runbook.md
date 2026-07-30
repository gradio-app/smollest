# Running the v0 experiment

Concrete commands for the experiment described in [`PLAN.md`](../PLAN.md). Read the
pre-registered pass conditions there *before* looking at any numbers.

## Prerequisites

llama.cpp on `PATH` (`brew install llama.cpp`) providing `llama-imatrix`,
`llama-quantize`, `llama-perplexity`, and `llama-bench`. Record the build with
`llama-cli --version`; imatrix behaviour changes between releases, so results
from different builds are not comparable. The runner stores the build string in
every results row.

## 1. Get a high-precision base model

The source must be BF16/F16. An already-quantized GGUF cannot be used: requantizing
compounds error and destroys the comparison between variants.

```bash
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-BF16.gguf \
  --local-dir ~/.cache/smollest/models
```

## 2. Get a general calibration corpus

This is the baseline the trace corpus has to beat.

```bash
curl -sL -o ~/.cache/smollest/calibration_datav3.txt \
  https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt
```

It is ~69k estimated tokens, or ~134 chunks at `ctx=512`. **This caps `--chunks` for
every variant.** Exceeding it means the general variant silently consumes less
calibration data than the trace variant, and the experiment then measures corpus
size rather than corpus content. The default of `--chunks 128` sits just under it.

## 3. A general-code evaluation corpus (E3, for H3)

E3 checks that trace calibration has not wrecked general ability. It must be
disjoint from both the general calibration corpus and your traces — the runner
refuses to reuse the calibration text, but it cannot detect subtler overlap, so
prefer source from repositories that appear nowhere in your transcripts.

## 4. Stage A: pick the corpus recipe

Your transcripts are ~57% tool results and ~30% tool-call JSON, so which parts of
a trace to calibrate on is a real question rather than a detail. Stage A settles it
at one bit width instead of guessing.

```bash
smollest stage-a \
  --base-model ~/.cache/smollest/models/Qwen3.5-4B-BF16.gguf \
  --general-corpus ~/.cache/smollest/calibration_datav3.txt
```

Prints mean KLD per recipe (`prose` / `authored` / `full`) and names the winner.

## 5. Stage B: the actual comparison

```bash
smollest stage-b \
  --base-model ~/.cache/smollest/models/Qwen3.5-4B-BF16.gguf \
  --general-corpus ~/.cache/smollest/calibration_datav3.txt \
  --general-eval ~/.cache/smollest/general-code-eval.txt \
  --recipe full
```

Four variants (`none` / `general` / `traces` / `hybrid`) at two bit widths across
every available eval set, with the trace variant repeated over disjoint subsamples
to establish a noise floor.

```bash
smollest report
```

## Resuming

Every cell is keyed by a hash of its inputs and appended to
`~/.cache/smollest/experiment/results.json`. Re-running skips completed cells, so
an interrupted run resumes. Changing `--chunks`, `--ctx`, `--eval-chunks`, the base
model, or the llama.cpp build changes the keys and forces recomputation, which is
intended: those are not comparable across values.

Intermediate imatrices and quantized models are cached by name under
`~/.cache/smollest/experiment/`. Delete a file to force just that step.

## Sanity checks before trusting any number

Run these first. Each catches a broken rig that would otherwise produce a
plausible-looking table.

| Check | If it fails |
|---|---|
| `none` is worse than `general` on E1/E2 | the imatrix is not being applied — confirm `llama-quantize` logged `load_imatrix` |
| `Q4_K_M` beats `Q3_K_M` within every variant | the evaluation is broken |
| replicate sigma is small next to the `none`-vs-`general` gap | eval sets are too small to resolve anything; H1 is untestable at this budget |

## Disk

Reference logits dominate. llama.cpp scores only the second half of each window
and stores uint16 log-probabilities, so a set is roughly

```
(ctx/2) x chunks x vocab x 2 bytes
```

At `ctx=512`, `--eval-chunks 20`, and Qwen3's ~152k vocab that is ~1.5GB per eval
set. `evaluate.logits_bytes()` computes it. Quantized models add ~2.5GB each and
there is one per variant per bit width.
