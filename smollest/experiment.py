"""Run the calibration experiment and record its results.

The question is whether calibrating quantization on your own agent traces beats
calibrating on a general corpus at the same model size -- and if so, whether it
beats it by enough to be worth a whole quantization level, which is the only
version of the claim that would justify the machinery over simply downloading a
smaller general quant.

Stage A compares corpus recipes cheaply at one bit width. Stage B takes the
winner and runs the real comparison:

===========  =========================================================
variant      importance matrix
===========  =========================================================
``none``     no imatrix at all
``general``  a public general-purpose calibration corpus
``traces``   your traces, winning recipe from stage A
``hybrid``   general merged with traces, keeping a general floor
===========  =========================================================

Three controls decide whether the output means anything:

*Budget parity.* Every variant consumes the same number of chunks, capped by
what the smallest corpus can supply, or the comparison measures corpus size
rather than corpus content.

*A noise floor.* The ``traces`` variant is repeated over disjoint subsamples so
that a difference can be tested against run-to-run spread instead of being
eyeballed.

*Holdout discipline.* Evaluation corpora come from projects and time ranges
excluded from calibration.

Results append to a JSON file keyed by a hash of each cell's inputs, so an
interrupted run resumes instead of recomputing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from smollest import evaluate, quantize, traces

DEFAULT_ROOT = Path.home() / ".cache" / "smollest" / "experiment"

VARIANT_NONE = "none"
VARIANT_GENERAL = "general"
VARIANT_TRACES = "traces"
VARIANT_HYBRID = "hybrid"


@dataclass
class Config:
    """Everything that defines a run, and therefore its cache keys."""

    base_model: Path
    general_corpus: Path
    general_eval: Path | None = None
    root: Path = DEFAULT_ROOT
    qtypes: list[str] = field(default_factory=lambda: ["Q3_K_M", "Q4_K_M"])
    stage_a_qtype: str = "Q3_K_M"
    recipes: list[str] = field(default_factory=lambda: ["prose", "authored", "full"])
    chunks: int = 128
    ctx: int = evaluate.DEFAULT_CTX
    eval_chunks: int = 20
    calib_budget_tokens: int = 400_000
    eval_budget_tokens: int = 20_000
    replicates: int = 3
    holdout_frac: float = 0.25
    time_frac: float = 0.15
    seed: int = 0

    def dirs(self) -> dict[str, Path]:
        return {
            name: self.root / name
            for name in ("corpora", "imatrices", "models", "logits")
        }

    @property
    def results_path(self) -> Path:
        return self.root / "results.json"


def cell_key(**parts: object) -> str:
    """Stable identifier for one measured cell, used to resume runs."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def append_result(path: Path, row: dict) -> None:
    rows = load_results(path)
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def prepare_corpora(config: Config) -> dict[str, Path]:
    """Build calibration and evaluation corpora, honouring the holdout splits."""
    corpora_dir = config.dirs()["corpora"]
    corpora_dir.mkdir(parents=True, exist_ok=True)

    scanned = traces.scan(traces.find_transcripts())
    if not scanned:
        raise RuntimeError("no transcripts found; nothing to calibrate on")

    holdout_projects = traces.pick_holdout_projects(
        scanned, frac=config.holdout_frac, seed=config.seed
    )
    calib, held_projects = traces.split_by_project(scanned, holdout_projects)
    cutoff = traces.pick_time_cutoff(calib, frac=config.time_frac)
    calib_early, held_time = (
        traces.split_by_time(calib, cutoff) if cutoff else (calib, [])
    )

    built: dict[str, Path] = {}

    for recipe in config.recipes:
        loaded = traces.load_blocks([t.path for t in calib_early], recipe)
        for replicate in range(config.replicates):
            blocks = traces.window(
                loaded,
                budget_tokens=config.calib_budget_tokens,
                seed=config.seed,
                replicate=replicate,
            )
            if not blocks:
                continue
            path = corpora_dir / f"calib-{recipe}-r{replicate}.txt"
            stats = traces.write_corpus(path, blocks)
            built[f"calib:{recipe}:{replicate}"] = path
            print(
                f"  calib {recipe} r{replicate}: {stats['est_tokens']:,} est tokens, "
                f"{stats['blocks']} blocks"
            )

    for label, pool in (("E1", held_projects), ("E2", held_time)):
        if not pool:
            print(f"  {label}: empty, skipping")
            continue
        blocks = traces.select_blocks(
            [t.path for t in pool],
            "full",
            budget_tokens=config.eval_budget_tokens,
            seed=config.seed + 1,
        )
        path = corpora_dir / f"eval-{label}.txt"
        stats = traces.write_corpus(path, blocks)
        built[f"eval:{label}"] = path
        print(
            f"  {label}: {stats['est_tokens']:,} est tokens, {stats['blocks']} blocks"
        )

    if config.general_eval is not None:
        if config.general_eval.resolve() == config.general_corpus.resolve():
            raise ValueError(
                "general_eval must differ from general_corpus, otherwise the "
                "general variant is evaluated on its own calibration data"
            )
        built["eval:E3"] = config.general_eval

    return built


def ensure_imatrix(config: Config, name: str, corpus: Path) -> Path:
    """Compute an importance matrix once and reuse it thereafter."""
    out = config.dirs()["imatrices"] / f"{name}.gguf"
    if out.exists():
        return out
    print(f"  imatrix {name} (chunks={config.chunks}, ctx={config.ctx})")
    return quantize.compute_imatrix(
        config.base_model, corpus, out, chunks=config.chunks, ctx=config.ctx
    )


def ensure_quant(config: Config, name: str, qtype: str, imatrix: Path | None) -> Path:
    """Quantize once and reuse thereafter."""
    out = config.dirs()["models"] / f"{name}-{qtype}.gguf"
    if out.exists():
        return out
    print(f"  quantize {name} {qtype}")
    return quantize.quantize(config.base_model, qtype, out, imatrix)


def ensure_base_logits(config: Config, label: str, corpus: Path) -> Path:
    """Save reference logits for one evaluation corpus, once."""
    out = config.dirs()["logits"] / f"{label}.dat"
    if out.exists():
        return out
    print(f"  base logits {label} (chunks={config.eval_chunks})")
    return evaluate.save_base_logits(
        config.base_model, corpus, out, chunks=config.eval_chunks, ctx=config.ctx
    )


def measure(
    config: Config,
    variant: str,
    qtype: str,
    model: Path,
    eval_label: str,
    eval_corpus: Path,
    base_logits: Path,
    replicate: int = 0,
    toolchain: str = "",
) -> dict:
    """Evaluate one cell, or return the recorded row if it already exists."""
    key = cell_key(
        variant=variant,
        qtype=qtype,
        eval_label=eval_label,
        replicate=replicate,
        chunks=config.chunks,
        eval_chunks=config.eval_chunks,
        ctx=config.ctx,
        model=config.base_model.name,
        toolchain=toolchain,
    )
    for row in load_results(config.results_path):
        if row.get("cell") == key:
            return row

    result = evaluate.kld(
        model, eval_corpus, base_logits, chunks=config.eval_chunks, ctx=config.ctx
    )
    row = {
        "cell": key,
        "variant": variant,
        "qtype": qtype,
        "eval": eval_label,
        "replicate": replicate,
        "size_bytes": quantize.model_size(model),
        "toolchain": toolchain,
        **result.metrics,
    }
    append_result(config.results_path, row)
    print(
        f"    {variant}/{qtype}/{eval_label} r{replicate}: "
        f"KLD={row.get('mean_kld')} same_top={row.get('same_top_pct')}%"
    )
    return row


def stage_a(config: Config) -> str:
    """Compare corpus recipes at one bit width and return the best."""
    toolchain = quantize.toolchain_version()
    print(f"stage A -- recipe selection (llama.cpp {toolchain})")
    corpora = prepare_corpora(config)

    eval_label = "E1" if "eval:E1" in corpora else "E2"
    eval_corpus = corpora.get(f"eval:{eval_label}")
    if eval_corpus is None:
        raise RuntimeError("no holdout evaluation corpus available")
    base_logits = ensure_base_logits(config, eval_label, eval_corpus)

    scores: dict[str, float] = {}
    for recipe in config.recipes:
        corpus = corpora.get(f"calib:{recipe}:0")
        if corpus is None:
            continue
        imatrix = ensure_imatrix(config, f"traces-{recipe}", corpus)
        model = ensure_quant(config, f"traces-{recipe}", config.stage_a_qtype, imatrix)
        row = measure(
            config,
            variant=f"traces-{recipe}",
            qtype=config.stage_a_qtype,
            model=model,
            eval_label=eval_label,
            eval_corpus=eval_corpus,
            base_logits=base_logits,
            toolchain=toolchain,
        )
        if row.get("mean_kld") is not None:
            scores[recipe] = row["mean_kld"]

    if not scores:
        raise RuntimeError("stage A produced no measurements")
    best = min(scores, key=scores.get)
    print(f"\nstage A scores (lower is better): {scores}")
    print(f"winning recipe: {best}")
    return best


def stage_b(config: Config, recipe: str) -> list[dict]:
    """Run the full variant comparison with the chosen recipe."""
    toolchain = quantize.toolchain_version()
    print(f"stage B -- variant comparison on recipe {recipe} (llama.cpp {toolchain})")
    corpora = prepare_corpora(config)

    general_imatrix = ensure_imatrix(config, "general", config.general_corpus)
    trace_imatrices = {}
    for replicate in range(config.replicates):
        corpus = corpora.get(f"calib:{recipe}:{replicate}")
        if corpus is None:
            continue
        trace_imatrices[replicate] = ensure_imatrix(
            config, f"traces-{recipe}-r{replicate}", corpus
        )
    if not trace_imatrices:
        raise RuntimeError("no trace calibration corpora were built")

    hybrid_imatrix = config.dirs()["imatrices"] / "hybrid.gguf"
    if not hybrid_imatrix.exists():
        print("  imatrix hybrid (merge of general + traces)")
        quantize.merge_imatrix([general_imatrix, trace_imatrices[0]], hybrid_imatrix)

    eval_sets = {
        label.split(":", 1)[1]: path
        for label, path in corpora.items()
        if label.startswith("eval:")
    }
    base_logits = {
        label: ensure_base_logits(config, label, path)
        for label, path in eval_sets.items()
    }

    rows: list[dict] = []
    for qtype in config.qtypes:
        plans: list[tuple[str, Path | None, int]] = [
            (VARIANT_NONE, None, 0),
            (VARIANT_GENERAL, general_imatrix, 0),
            (VARIANT_HYBRID, hybrid_imatrix, 0),
        ]
        plans += [
            (VARIANT_TRACES, imatrix, replicate)
            for replicate, imatrix in trace_imatrices.items()
        ]
        for variant, imatrix, replicate in plans:
            suffix = f"-r{replicate}" if variant == VARIANT_TRACES else ""
            model = ensure_quant(config, f"{variant}{suffix}", qtype, imatrix)
            for label, path in eval_sets.items():
                rows.append(
                    measure(
                        config,
                        variant=variant,
                        qtype=qtype,
                        model=model,
                        eval_label=label,
                        eval_corpus=path,
                        base_logits=base_logits[label],
                        replicate=replicate,
                        toolchain=toolchain,
                    )
                )
    return rows


def noise_floor(rows: list[dict], qtype: str, eval_label: str) -> float | None:
    """Standard deviation of the trace variant across replicates."""
    values = [
        row["mean_kld"]
        for row in rows
        if row.get("variant") == VARIANT_TRACES
        and row.get("qtype") == qtype
        and row.get("eval") == eval_label
        and row.get("mean_kld") is not None
    ]
    return statistics.stdev(values) if len(values) > 1 else None


def markdown_table(rows: list[dict]) -> str:
    """Render results as a table, collapsing replicates to mean +- spread."""
    header = (
        "| variant | qtype | eval | mean KLD | sigma | same top % | size (GB) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("variant")), str(row.get("qtype")), str(row.get("eval"))), []
        ).append(row)

    lines = []
    for (variant, qtype, eval_label), group in sorted(grouped.items()):
        klds = [r["mean_kld"] for r in group if r.get("mean_kld") is not None]
        if not klds:
            continue
        tops = [r["same_top_pct"] for r in group if r.get("same_top_pct") is not None]
        sizes = [r["size_bytes"] for r in group if r.get("size_bytes")]
        sigma = f"{statistics.stdev(klds):.6f}" if len(klds) > 1 else "-"
        lines.append(
            f"| {variant} | {qtype} | {eval_label} | {statistics.fmean(klds):.6f} | "
            f"{sigma} | {statistics.fmean(tops):.3f} | "
            f"{(statistics.fmean(sizes) / 1e9):.2f} |"
        )
    return header + "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smollest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("stage-a", "stage-b", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name == "report":
            continue
        p.add_argument("--base-model", type=Path, required=True)
        p.add_argument("--general-corpus", type=Path, required=True)
        p.add_argument("--general-eval", type=Path, default=None)
        p.add_argument("--chunks", type=int, default=128)
        p.add_argument("--eval-chunks", type=int, default=20)
        p.add_argument("--ctx", type=int, default=evaluate.DEFAULT_CTX)
        p.add_argument("--replicates", type=int, default=3)
        if name == "stage-b":
            p.add_argument("--recipe", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "report":
        rows = load_results(args.root / "results.json")
        if not rows:
            print("no results yet")
            return 1
        print(markdown_table(rows))
        return 0

    config = Config(
        base_model=args.base_model,
        general_corpus=args.general_corpus,
        general_eval=args.general_eval,
        root=args.root,
        chunks=args.chunks,
        eval_chunks=args.eval_chunks,
        ctx=args.ctx,
        replicates=args.replicates,
    )
    for path in config.dirs().values():
        path.mkdir(parents=True, exist_ok=True)
    (config.root / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str), encoding="utf-8"
    )

    if args.command == "stage-a":
        stage_a(config)
        return 0

    rows = stage_b(config, args.recipe)
    print()
    print(markdown_table(rows))
    for qtype in config.qtypes:
        for label in sorted({str(r.get("eval")) for r in rows}):
            sigma = noise_floor(rows, qtype, label)
            if sigma is not None:
                print(f"noise floor sigma ({qtype}, {label}): {sigma:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
