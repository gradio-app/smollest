"""Measure how far a quantized model drifts from its unquantized source.

The primary metric is KL divergence against the F16 model's own logits on a
held-out corpus, not perplexity. Perplexity says how surprised a model is;
KL divergence says how differently it behaves from the model you actually
wanted, which is the question a quantization choice raises.

The workflow is two-pass, matching llama.cpp's:

1. :func:`save_base_logits` runs the F16 model and writes its log-probabilities.
2. :func:`kld` runs each quantized model against that file.

Both passes must use the same context size, or llama.cpp rejects the logits
file. The context is recorded in a sidecar next to it and checked, because the
native error surfaces late and reads as a corrupt-file problem.

Note what this metric does *not* cover: fidelity to the F16 model on your
distribution is necessary but not sufficient. It says nothing about task
success in an agent loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from smollest.quantize import ToolchainError, run

PERPLEXITY_BIN = "llama-perplexity"
BENCH_BIN = "llama-bench"

DEFAULT_CTX = 512
LOGITS_DTYPE_BYTES = 2

_NUM = r"(-?\d+(?:\.\d+)?)"
_DELTA_P = "\u0394p"

_PATTERNS: dict[str, str] = {
    "mean_kld": rf"Mean\s+KLD:\s*{_NUM}",
    "mean_kld_unc": rf"Mean\s+KLD:\s*{_NUM}\s*±\s*{_NUM}",
    "median_kld": rf"Median\s+KLD:\s*{_NUM}",
    "max_kld": rf"Maximum\s+KLD:\s*{_NUM}",
    "kld_99": rf"99\.9%\s+KLD:\s*{_NUM}",
    "same_top_pct": rf"Same top p:\s*{_NUM}",
    "rms_delta_p": rf"RMS\s+{_DELTA_P}\s*:\s*{_NUM}",
    "mean_delta_p": rf"Mean\s+{_DELTA_P}:\s*{_NUM}",
    "ppl_quant": rf"Mean PPL\(Q\)\s*:\s*{_NUM}",
    "ppl_base": rf"Mean PPL\(base\)\s*:\s*{_NUM}",
    "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*{_NUM}",
}


@dataclass
class EvalResult:
    """Parsed divergence metrics plus the raw output they came from."""

    metrics: dict[str, float]
    stdout: str = field(repr=False, default="")

    @property
    def mean_kld(self) -> float | None:
        return self.metrics.get("mean_kld")

    @property
    def same_top_pct(self) -> float | None:
        return self.metrics.get("same_top_pct")


def logits_bytes(n_vocab: int, chunks: int, ctx: int = DEFAULT_CTX) -> int:
    """Predict the size of a saved logits file.

    llama.cpp scores only the second half of each context window and stores
    log-probabilities as uint16, so the file is roughly a quarter of the naive
    ``tokens x vocab x float32`` estimate.
    """
    scored_per_chunk = max(0, ctx - 1 - ctx // 2)
    return scored_per_chunk * chunks * n_vocab * LOGITS_DTYPE_BYTES


def _meta_path(logits: Path) -> Path:
    return logits.with_suffix(logits.suffix + ".meta.json")


def save_base_logits(
    model: Path,
    corpus: Path,
    out: Path,
    chunks: int,
    ctx: int = DEFAULT_CTX,
    timeout: float | None = None,
) -> Path:
    """Run the reference model and save its log-probabilities."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            PERPLEXITY_BIN,
            "-m",
            str(model),
            "-f",
            str(corpus),
            "--kl-divergence-base",
            str(out),
            "-c",
            str(ctx),
            "--chunks",
            str(chunks),
        ],
        timeout=timeout,
    )
    if not out.exists():
        raise ToolchainError(f"{PERPLEXITY_BIN} reported success but {out} is missing")
    _meta_path(out).write_text(
        json.dumps(
            {"model": str(model), "corpus": str(corpus), "ctx": ctx, "chunks": chunks}
        ),
        encoding="utf-8",
    )
    return out


def parse_kld(stdout: str) -> dict[str, float]:
    """Extract divergence metrics from llama-perplexity output."""
    metrics: dict[str, float] = {}
    for key, pattern in _PATTERNS.items():
        if key.endswith("_unc"):
            continue
        match = re.search(pattern, stdout)
        if match:
            metrics[key] = float(match.group(1))
    uncertainty = re.search(_PATTERNS["mean_kld_unc"], stdout)
    if uncertainty:
        metrics["mean_kld_unc"] = float(uncertainty.group(2))
    return metrics


def kld(
    model: Path,
    corpus: Path,
    base_logits: Path,
    chunks: int,
    ctx: int = DEFAULT_CTX,
    timeout: float | None = None,
) -> EvalResult:
    """Measure a quantized model's divergence from the saved reference logits."""
    if not base_logits.exists():
        raise ToolchainError(f"{base_logits} not found; run save_base_logits first")
    meta = _meta_path(base_logits)
    if meta.exists():
        recorded = json.loads(meta.read_text(encoding="utf-8")).get("ctx")
        if recorded is not None and int(recorded) != ctx:
            raise ToolchainError(
                f"context mismatch: {base_logits} was saved with -c {recorded}, "
                f"but this run uses -c {ctx}; llama.cpp requires them to match"
            )

    result = run(
        [
            PERPLEXITY_BIN,
            "-m",
            str(model),
            "-f",
            str(corpus),
            "--kl-divergence",
            "--kl-divergence-base",
            str(base_logits),
            "-c",
            str(ctx),
            "--chunks",
            str(chunks),
        ],
        timeout=timeout,
    )
    metrics = parse_kld(result.stdout)
    if "mean_kld" not in metrics:
        raise ToolchainError(
            f"could not parse mean KLD from {PERPLEXITY_BIN} output; "
            f"last lines:\n{chr(10).join(result.stdout.splitlines()[-15:])}"
        )
    return EvalResult(metrics=metrics, stdout=result.stdout)


def parse_bench(stdout: str) -> dict[str, float]:
    """Extract prompt and generation throughput from llama-bench JSON."""
    rows = json.loads(stdout)
    speeds: dict[str, float] = {}
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        tps = row.get("avg_ts")
        if tps is None:
            continue
        if n_prompt and not n_gen:
            speeds["prompt_tps"] = float(tps)
        elif n_gen:
            speeds["gen_tps"] = float(tps)
    return speeds


def bench(
    model: Path,
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 3,
    timeout: float | None = None,
) -> dict[str, float]:
    """Measure throughput. Only comparable within one machine and one build."""
    result = run(
        [
            BENCH_BIN,
            "-m",
            str(model),
            "-p",
            str(n_prompt),
            "-n",
            str(n_gen),
            "-r",
            str(repetitions),
            "-o",
            "json",
        ],
        timeout=timeout,
        merge_stderr=False,
    )
    try:
        return parse_bench(result.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ToolchainError(f"could not parse {BENCH_BIN} JSON output: {exc}") from exc
