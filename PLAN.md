# smollest v0: falsify trace-calibrated quantization

## Context

The README pitches: take a local coding model, calibrate quantization on your own agent
traces, get a smaller/faster model specialized to your workload. Two problems motivate
building the experiment before the product:

1. **The literature does not support the strong claim.** Domain-matched calibration is
   documented to improve quality *at a fixed bit width* ([Calibrating Beyond
   English](https://arxiv.org/pdf/2601.18306)). Nothing shows it lets you *spend fewer
   bits*. [COVERCAL](https://arxiv.org/html/2604.24008) argues domain representativeness
   is the wrong objective entirely — outlier-channel coverage is what matters, and
   domain-focused selection can underperform through redundant coverage. Unsloth found
   narrow calibration causes benchmark "cheating" and deliberately built *general* sets.
2. **The product claim has an obvious cheap alternative**: just download the smaller
   general quant. Any version of this that's worth building must beat that baseline.

So v0 is not a product. It is a measurement rig whose purpose is to kill the idea cheaply
if it deserves killing, and to produce the numbers the README currently lacks.

**Outcome:** a results table that either supports building `smollify`, or tells you to stop.

## Environment (verified)

| | |
|---|---|
| Hardware | Apple M5, 32GB, Metal working set 26.8GB, 612GB free disk |
| llama.cpp | b8140 via homebrew — `llama-imatrix`, `llama-quantize`, `llama-perplexity`, `llama-bench` all present |
| KLD support | `llama-perplexity --kl-divergence --kl-divergence-base FNAME` ✓ |
| imatrix merge | `llama-imatrix --in-file a.gguf --in-file b.gguf` ✓ (enables hybrid variant) |
| Traces | 1775 Claude Code JSONL, ~18M unique tokens after exact dedup (only 1.9% duplicate) |
| Trace dates | 1764 files in 2026-07, 11 in 2026-06 — **~1 month of history only** |
| Trace mix | 57% `tool_result`, 30% `tool_use` JSON, 8% user prose, 2% assistant text |
| Concentration | Top 8 project cwds dominate: icml-2026-repro-challenge, trackio, gradio worktrees |

Two consequences worth stating up front. First, "calibrating on your traces" mostly means
calibrating on **file contents and shell output you happened to read** — not on model
behavior. Second, with only one month of history a temporal split is weak; the
**held-out-project split is the real generalization test**.

## The claim, stated falsifiably

Pre-register these before running anything. σ = std of mean-KLD across the three
replicate subsamples (see Controls).

| | Hypothesis | Pass condition |
|---|---|---|
| **H1** | Trace calibration beats general calibration at matched size | `KLD(trace) < KLD(general)` by >2σ on **both** held-out eval sets, at **both** bit widths |
| **H2** | It's worth a whole quant level (the product claim) | `KLD(trace, Q3_K_M) ≤ KLD(general, Q4_K_M)` |
| **H3** | It doesn't wreck general ability | `KLD(trace)` on general-code eval ≤ 1.1 × `KLD(general)` on same |

Interpretation, decided in advance:

- **H1 fails** → idea is dead in this form. Stop. Pivot to the eval harness (see Hedge).
- **H1 passes, H2 fails** → real but marginal. Reframe honestly as "slightly better quants
  for your workload," not "hyper-specific model." Do not ship the current README's pitch.
- **H1 + H2 pass** → the pitch is roughly right. Build v1.
- **H3 fails** → gate any shipping product behind a general-calibration floor (hybrid).

**Honest limit of this metric:** KLD-vs-F16 on held-out traces measures *fidelity to the
unquantized model on your distribution*. It does not measure task success. A v0 pass is
necessary, not sufficient — agent-loop eval is v1. Say this in any writeup.

## v0 experiment design

**Base model:** a 4B at F16 (~8GB), e.g. Qwen3-4B. Fits Metal working set comfortably,
fast enough to iterate on methodology. Repeat the winning config on the 27B you actually
run only if the signal survives. Source must be F16/BF16 — the Qwen3.5 Q4_K_M files
already in `~/Library/Caches/llama.cpp` cannot be used (requantizing destroys the
comparison).

**Staged matrix** — stage A picks the corpus recipe cheaply, stage B runs the real test.

*Stage A — which corpus recipe calibrates best? (Q3_K_M only, where imatrix matters most)*

| Recipe | Contents |
|---|---|
| R1 | user prose + assistant text only (~10% of tokens) |
| R2 | R1 + `tool_use` JSON (model-authored, ~40%) |
| R3 | everything incl. `tool_result` (full input distribution, 100%) |

*Stage B — full comparison at Q3_K_M and Q4_K_M*

| Variant | imatrix |
|---|---|
| A | none |
| B | general calibration set (public, e.g. bartowski `calibration_datav3`) |
| C | traces, winning recipe from stage A |
| D | hybrid — `--in-file` merge of B and C |

*Eval sets* (disjoint from calibration, ~10k tokens each)

| | Set | Tests |
|---|---|---|
| E1 | held-out **projects** — traces from cwds excluded from calibration | generalization to new work |
| E2 | held-out **time** — latest traces, calibration from earlier | does it survive next week |
| E3 | general code, public (non-trace) | H3 regression check |

*Metrics per cell:* mean KLD vs F16 (primary), top-1 token agreement %, PPL, file size,
tok/s from `llama-bench`.

*Controls — these are what make the result trustworthy:*
- **Noise floor:** run variant C three times on disjoint random subsamples of the same
  calibration corpus. σ across replicates defines "exceeds noise." Without this the whole
  experiment is unfalsifiable.
- **Token budget held constant** across recipes and against the general set — otherwise
  you measure corpus size, not corpus content.
- **Never evaluate on a project present in calibration.** The top-8 concentration makes
  this trivially easy to get wrong and would inflate every number.

## Implementation

Thin modules in `smollest/`, shelling out to llama.cpp. No inference in Python, no new
runtime deps beyond `huggingface-hub` for the model download.

**`smollest/traces.py`** — the only nontrivial pure logic, and the only part worth unit tests.
```
iter_records(paths)              -> yields dicts from Claude Code JSONL
render_block(block, recipe)      -> str   # recipe ∈ {R1,R2,R3}; handles text/thinking/tool_use/tool_result
build_corpus(files, recipe, budget_tokens, seed) -> str
split_by_project(files)          -> (calib_files, heldout_files)
split_by_time(files, cutoff)     -> (calib_files, heldout_files)
dedup(blocks)                    -> list  # sha1 exact-match
```
Note: records are `{"type": "user"|"assistant", "message": {"content": [...]}}` with
`cwd`/`timestamp` at top level; `content` is sometimes a bare string. Skip `type` values
other than user/assistant (`queue-operation`, `system`, `attachment`, … are noise).

**`smollest/quantize.py`** — subprocess wrappers, each returning the output path.
```
compute_imatrix(model_f16, corpus_path, out, chunks=200, ctx=512)
merge_imatrix(inputs, out)                       # llama-imatrix --in-file a --in-file b
quantize(model_f16, qtype, out, imatrix=None)    # llama-quantize --imatrix ...
```

**`smollest/evaluate.py`**
```
save_base_logits(model_f16, eval_path, out, chunks)
kld(model_q, eval_path, base_logits, chunks) -> {mean_kld, top1_agree, ppl}
bench(model_q) -> {pp_tps, tg_tps}             # llama-bench
```
Parse llama.cpp stdout with regexes; keep raw stdout alongside parsed values so a parse
break never silently loses a 4-minute run.

**`smollest/experiment.py`** — matrix runner. Writes `results.json` (append-only, one row
per cell, keyed by config hash so reruns resume) plus a markdown table. Idempotent:
skip cells already in `results.json`.

**CLI:** `smollest experiment stage-a|stage-b` (add `[project.scripts]` back to
`pyproject.toml`).

**Tests** (`tests/test_traces.py`): recipe filtering, bare-string vs list content, dedup,
both splits, budget truncation. Mock the subprocess layer — do not run llama.cpp in tests.

## Delivery: plan doc + 4 stacked PRs

**Plan doc:** this file lands as `PLAN.md` on `main` and is pushed directly (consistent with
how you push here). Note it will trigger the `publish` workflow again, as will each PR merge.

**Stacked PRs.** GitHub's native stacked PRs hit public preview today (2026-07-30), via
`gh extension install github/gh-stack` — commands are `gs init <branch>`, `gs add <branch>`,
`gs push`, `gs submit`. `gh` 2.83.1 is installed and authed as `abidlabs` with `repo` scope,
but no extensions yet. I'll verify the real command surface with `gh stack --help` after
installing, since the feature is one day old and the public docs are thin. **Fallback if the
extension misbehaves:** plain git branches with `gh pr create --base <previous-branch>`,
which produces the same stack and is recognized by GitHub's stack map UI. Either way the
result is four PRs you can review and merge bottom-up.

| PR | Branch | Contents | Depends on |
|---|---|---|---|
| 1 | `v0-traces` | `smollest/traces.py` + `tests/test_traces.py` — extraction, 3 recipes, dedup, both splits. No llama.cpp dependency, fully unit-tested. | `main` |
| 2 | `v0-quantize` | `smollest/quantize.py` — imatrix / merge / quantize wrappers, subprocess mocked in tests | PR 1 |
| 3 | `v0-evaluate` | `smollest/evaluate.py` — base logits, KLD parsing, bench; stdout-parsing tests on captured fixtures | PR 2 |
| 4 | `v0-experiment` | `smollest/experiment.py`, `[project.scripts]` CLI, run instructions | PR 3 |

Each PR is independently reviewable and does nothing on import. Ordering keeps the only
nontrivial pure logic (trace parsing) in the first, smallest PR.

**Scope boundary:** these PRs deliver the *rig*, not the results. Running stages A and B is
hours of compute and follows once you've merged. Exception — before opening PR 4 I run one
real cell end to end on the 4B (Phase 0 + a single quantize/eval round trip) and put the
actual numbers in PR 4's description, so you aren't reviewing subprocess wrappers that have
never touched a real model.

## Execution phases

**Phase 0 — de-risk before committing compute (~30 min).** Three things can invalidate the
plan and all are cheap to check:
1. Download an F16 4B GGUF; confirm `llama-quantize` accepts it. Verify the chosen repo
   actually publishes F16/BF16 — if not, fall back to `convert_hf_to_gguf.py` from a
   cloned llama.cpp repo (needs no cmake, just the Python script).
2. **Measure the KLD logits file size with `--chunks 2`, then extrapolate.** Estimate is
   `n_tokens × n_vocab × 4B`; at Qwen3's ~152k vocab, 10k tokens ≈ 6GB per eval set.
   Fine on 612GB free, but confirm the dtype rather than trusting the arithmetic. If it's
   larger than expected, cut eval tokens or compute top-1 agreement from two plain runs.
3. Confirm `merge_imatrix` semantics — that `--in-file` averages rather than
   concatenates-and-overwrites. If it doesn't do what variant D needs, build the hybrid by
   concatenating corpora instead.

**Phase 1 — traces.py + tests.** Pure logic, no compute. Print corpus stats per recipe so
the token-budget control is visible.

**Phase 2 — quantize.py + evaluate.py**, validated on one throwaway cell end to end.

**Phase 3 — Stage A** (3 recipes, Q3_K_M, E1 only). Pick winner. ~1 hr.

**Phase 4 — Stage B** (4 variants × 2 bit widths × 3 eval sets + 3 replicates).
~14 quants, ~45 KLD runs, ~3 hrs on the 4B.

**Phase 5 — write up against the pre-registered criteria.** Include the negative result
prominently if that's what it is; that outcome is the point of the exercise.

## Verification

- `pytest` green on trace parsing; `ruff check --fix --select I && ruff format`.
- Sanity checks that catch a broken rig before you trust any number:
  - Variant A (no imatrix) must be **worse** than B on E1/E2. If not, the imatrix isn't
    being applied — check `llama-quantize` actually consumed `--imatrix`.
  - Q4_K_M must beat Q3_K_M within every variant. If not, the eval is broken.
  - KLD of F16 against its own logits must be ~0.
  - Replicate σ should be small relative to the A-vs-B gap; if σ swamps that known-real
    difference, the eval sets are too small to resolve anything and H1 is untestable at
    this budget.
- End-to-end: `smollest experiment stage-b` from clean produces `results.json` +
  markdown table with every cell populated.

## Risks

- **~1 month of trace history** means the temporal split is short-horizon and the drift /
  ratchet question the README raises cannot be answered yet. Don't claim it is.
- **4B may not transfer to 27B.** Bit allocation sensitivity differs with scale. A 4B pass
  licenses a 27B confirmation run, not a product claim.
- **Trace corpus is mostly file contents**, so this partly measures "calibrate on my
  repos' source" rather than "calibrate on my agent behavior." Stage A's recipe comparison
  is what separates those.
- **Metal tensor API is disabled** on this build (`has tensor = false`) — perf only, not
  correctness, but keep tok/s comparisons within one machine and one build.
- Homebrew llama.cpp upgrades mid-experiment would invalidate cross-cell comparisons. Pin:
  record `llama-cli --version` in every results row.

## Hedge: the eval harness is valuable even if H1 fails

If trace calibration turns out to be noise, the rig built in phases 1–2 still answers a
question nobody currently answers for users: *which off-the-shelf quant should I run, given
my actual work?* Point `evaluate.py` at bartowski/unsloth GGUFs of the same model and rank
them by KLD on your held-out traces. That is a genuinely useful tool, it reuses everything,
and it's the natural fallback product. Worth knowing before you start that the downside
case still ships something.

## Fuller proposal (v1) — only if H1 and H2 pass

What the README becomes, with the earlier critiques addressed:

1. **Lead with measurements**, not intuition. Base model, RAM before/after, tok/s, and KLD
   vs F16 on held-out traces. Include the comparison to "just download the smaller general
   quant" above the fold, because that's the reader's first objection.
2. **Drop the capability-removal framing.** Replace "the model knows Haskell and Arabic you
   don't need" with the defensible version: better bit allocation for your distribution.
   Quantization does not remove separable capabilities, and claiming it does will lose
   exactly the readers who could evaluate the work.
3. **Eval gate as the headline feature.** Never replace a model without replaying held-out
   traces and reporting divergence; refuse to install a regression. This is the answer to
   "how do I know it didn't get worse," it's what makes the tool trustworthy, and it's
   more defensible than the quantization itself.
4. **General-calibration floor.** Ship variant D (hybrid), not pure trace calibration, so
   the model retains outlier-channel coverage and can't narrow catastrophically —
   contingent on what H3 shows.
5. **Bound the drift.** Pin versions, keep the previous quant on disk, `smollest rollback`,
   and re-run the eval gate on every requantization. Report drift over time instead of
   silently updating weekly.
6. **OOD detection, concretely.** Track rolling KLD/perplexity of live traffic against the
   calibration baseline; warn past a threshold; be explicit it's a heuristic. Remediation
   must be real — fall back to the retained general quant, not just print a warning.
7. **Local by default.** HF publishing opt-in with explicit consent, since a model
   calibrated on proprietary traces is derived from them. Note base-model license
   constraints on redistributing derivatives.
8. **Cold start + adapters.** Ship general calibration; specialize once N tokens exist.
   Treat Claude Code / Codex transcript formats as unstable adapters with fixtures, because
   they are undocumented and will churn.
