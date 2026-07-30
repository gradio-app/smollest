from __future__ import annotations

import json

import pytest

from smollest import experiment


def make_config(tmp_path, **overrides):
    defaults = dict(
        base_model=tmp_path / "base.gguf",
        general_corpus=tmp_path / "general.txt",
        root=tmp_path / "run",
    )
    defaults.update(overrides)
    return experiment.Config(**defaults)


def row(variant, qtype, eval_label, kld, replicate=0, top=99.0, size=2_000_000_000):
    return {
        "cell": f"{variant}-{qtype}-{eval_label}-{replicate}",
        "variant": variant,
        "qtype": qtype,
        "eval": eval_label,
        "replicate": replicate,
        "mean_kld": kld,
        "same_top_pct": top,
        "size_bytes": size,
    }


def test_cell_key_is_stable_and_order_independent():
    a = experiment.cell_key(variant="traces", qtype="Q3_K_M", eval_label="E1")
    b = experiment.cell_key(eval_label="E1", qtype="Q3_K_M", variant="traces")
    assert a == b


def test_cell_key_changes_with_inputs():
    a = experiment.cell_key(variant="traces", chunks=128)
    b = experiment.cell_key(variant="traces", chunks=200)
    assert a != b


def test_results_roundtrip(tmp_path):
    path = tmp_path / "results.json"
    assert experiment.load_results(path) == []
    experiment.append_result(path, {"cell": "x", "mean_kld": 0.1})
    experiment.append_result(path, {"cell": "y", "mean_kld": 0.2})
    rows = experiment.load_results(path)
    assert [r["cell"] for r in rows] == ["x", "y"]


def test_measure_resumes_from_recorded_cell(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.results_path.parent.mkdir(parents=True, exist_ok=True)

    def explode(*args, **kwargs):
        raise AssertionError("kld should not run for an already-recorded cell")

    monkeypatch.setattr(experiment.evaluate, "kld", explode)

    key = experiment.cell_key(
        variant="traces",
        qtype="Q3_K_M",
        eval_label="E1",
        replicate=0,
        chunks=config.chunks,
        eval_chunks=config.eval_chunks,
        ctx=config.ctx,
        model=config.base_model.name,
        toolchain="b1",
    )
    experiment.append_result(config.results_path, {"cell": key, "mean_kld": 0.5})

    result = experiment.measure(
        config,
        variant="traces",
        qtype="Q3_K_M",
        model=tmp_path / "m.gguf",
        eval_label="E1",
        eval_corpus=tmp_path / "e.txt",
        base_logits=tmp_path / "b.dat",
        toolchain="b1",
    )
    assert result["mean_kld"] == 0.5


def test_measure_records_new_cell(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"0" * 1024)

    monkeypatch.setattr(
        experiment.evaluate,
        "kld",
        lambda *a, **k: experiment.evaluate.EvalResult(
            metrics={"mean_kld": 0.0123, "same_top_pct": 98.7}
        ),
    )
    monkeypatch.setattr(experiment.quantize, "toolchain_version", lambda: "b1")

    result = experiment.measure(
        config,
        variant="general",
        qtype="Q4_K_M",
        model=model,
        eval_label="E1",
        eval_corpus=tmp_path / "e.txt",
        base_logits=tmp_path / "b.dat",
        toolchain="b1",
    )
    assert result["mean_kld"] == 0.0123
    assert result["size_bytes"] == 1024
    assert len(experiment.load_results(config.results_path)) == 1


def test_noise_floor_uses_trace_replicates_only():
    rows = [
        row("traces", "Q3_K_M", "E1", 0.010, replicate=0),
        row("traces", "Q3_K_M", "E1", 0.012, replicate=1),
        row("traces", "Q3_K_M", "E1", 0.014, replicate=2),
        row("general", "Q3_K_M", "E1", 0.500),
    ]
    sigma = experiment.noise_floor(rows, "Q3_K_M", "E1")
    assert sigma == pytest.approx(0.002, abs=1e-9)


def test_noise_floor_none_with_single_replicate():
    rows = [row("traces", "Q3_K_M", "E1", 0.010)]
    assert experiment.noise_floor(rows, "Q3_K_M", "E1") is None


def test_markdown_table_collapses_replicates():
    rows = [
        row("traces", "Q3_K_M", "E1", 0.010, replicate=0),
        row("traces", "Q3_K_M", "E1", 0.020, replicate=1),
        row("general", "Q3_K_M", "E1", 0.030),
    ]
    table = experiment.markdown_table(rows)
    assert "| traces | Q3_K_M | E1 | 0.015000 |" in table
    assert "| general | Q3_K_M | E1 | 0.030000 | - |" in table


def test_markdown_table_skips_rows_without_kld():
    assert "nan" not in experiment.markdown_table([{"variant": "x", "qtype": "y"}])


def test_general_eval_must_differ_from_calibration(tmp_path, monkeypatch):
    shared = tmp_path / "cal.txt"
    shared.write_text("data", encoding="utf-8")
    config = make_config(tmp_path, general_corpus=shared, general_eval=shared)
    monkeypatch.setattr(
        experiment.traces, "find_transcripts", lambda root=None: [tmp_path / "t.jsonl"]
    )
    monkeypatch.setattr(
        experiment.traces,
        "scan",
        lambda paths: [
            experiment.traces.Transcript(
                path=tmp_path / "t.jsonl", project="p", start=None, end=None
            )
        ],
    )
    monkeypatch.setattr(experiment.traces, "load_blocks", lambda *a, **k: [])
    monkeypatch.setattr(experiment.traces, "select_blocks", lambda *a, **k: ["block"])
    with pytest.raises(ValueError, match="must differ"):
        experiment.prepare_corpora(config)


def test_prepare_corpora_errors_without_transcripts(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(experiment.traces, "find_transcripts", lambda root=None: [])
    monkeypatch.setattr(experiment.traces, "scan", lambda paths: [])
    with pytest.raises(RuntimeError, match="no transcripts found"):
        experiment.prepare_corpora(config)


def test_ensure_imatrix_reuses_existing(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    out = config.dirs()["imatrices"] / "general.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"cached")

    def explode(*args, **kwargs):
        raise AssertionError("should not recompute an existing imatrix")

    monkeypatch.setattr(experiment.quantize, "compute_imatrix", explode)
    assert experiment.ensure_imatrix(config, "general", tmp_path / "c.txt") == out


def test_ensure_quant_reuses_existing(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    out = config.dirs()["models"] / "general-Q3_K_M.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"cached")

    def explode(*args, **kwargs):
        raise AssertionError("should not requantize an existing model")

    monkeypatch.setattr(experiment.quantize, "quantize", explode)
    assert experiment.ensure_quant(config, "general", "Q3_K_M", None) == out


def test_chunks_default_fits_general_corpus():
    config = experiment.Config(base_model=None, general_corpus=None)
    assert config.chunks <= 134


def test_cli_report_without_results(tmp_path, capsys):
    code = experiment.main(["report", "--root", str(tmp_path)])
    assert code == 1
    assert "no results yet" in capsys.readouterr().out


def test_cli_report_renders_table(tmp_path, capsys):
    (tmp_path / "results.json").write_text(
        json.dumps([row("traces", "Q3_K_M", "E1", 0.01)]), encoding="utf-8"
    )
    code = experiment.main(["report", "--root", str(tmp_path)])
    assert code == 0
    assert "mean KLD" in capsys.readouterr().out


def test_cli_stage_b_requires_recipe(tmp_path):
    with pytest.raises(SystemExit):
        experiment.main(
            [
                "stage-b",
                "--base-model",
                str(tmp_path / "m.gguf"),
                "--general-corpus",
                str(tmp_path / "c.txt"),
            ]
        )
