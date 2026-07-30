from __future__ import annotations

import json
import subprocess

import pytest

from smollest import evaluate, quantize

KLD_OUTPUT = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   7.123456 ±   0.045678
Mean PPL(base)                :   7.001234 ±   0.044444
Cor(ln(PPL(Q)), ln(PPL(base))):  99.12%
Mean ln(PPL(Q)/PPL(base))     :   0.017321 ±   0.001234
Mean PPL(Q)/PPL(base)         :   1.017472 ±   0.001255
Mean PPL(Q)-PPL(base)         :   0.122222 ±   0.008888

====== KL divergence statistics ======
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   1.234567
99.9%   KLD:   0.456789
99.0%   KLD:   0.234567
Median  KLD:   0.001234
Minimum KLD:   0.000001

====== Token probability statistics ======
Mean    Δp:  -0.123 ±  0.045 %
Maximum Δp: 12.345%
Median  Δp: -0.001%
RMS Δp    :   1.234 ± 0.056 %
Same top p:  98.765 ± 0.123 %
"""

BENCH_JSON = json.dumps(
    [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 1234.56},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 45.67},
    ]
)


@pytest.fixture
def fake_binaries(monkeypatch):
    monkeypatch.setattr(quantize.shutil, "which", lambda name: f"/fake/bin/{name}")


class Recorder:
    def __init__(self, stdout="", stderr="", returncode=0, create_output=True):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.create_output = create_output
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if self.create_output and "--kl-divergence-base" in args:
            if "--kl-divergence" not in args:
                open(args[args.index("--kl-divergence-base") + 1], "w").close()
        return subprocess.CompletedProcess(
            args, self.returncode, stdout=self.stdout, stderr=self.stderr
        )

    @property
    def last(self):
        return self.calls[-1]


def test_parse_kld_extracts_every_metric():
    metrics = evaluate.parse_kld(KLD_OUTPUT)
    assert metrics["mean_kld"] == 0.012345
    assert metrics["mean_kld_unc"] == 0.000123
    assert metrics["median_kld"] == 0.001234
    assert metrics["max_kld"] == 1.234567
    assert metrics["kld_99"] == 0.456789
    assert metrics["same_top_pct"] == 98.765
    assert metrics["rms_delta_p"] == 1.234
    assert metrics["mean_delta_p"] == -0.123
    assert metrics["ppl_quant"] == 7.123456
    assert metrics["ppl_base"] == 7.001234
    assert metrics["ppl_ratio"] == 1.017472


def test_parse_kld_on_unrelated_output_is_empty():
    assert evaluate.parse_kld("loading model...\ndone\n") == {}


def test_logits_bytes_accounts_for_half_context_and_uint16():
    assert evaluate.logits_bytes(n_vocab=100, chunks=10, ctx=512) == 255 * 10 * 100 * 2
    assert evaluate.logits_bytes(n_vocab=100, chunks=1, ctx=2) == 0


def test_save_base_logits_writes_sidecar(fake_binaries, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", Recorder())
    out = tmp_path / "base.dat"
    evaluate.save_base_logits(
        tmp_path / "m.gguf", tmp_path / "e.txt", out, chunks=4, ctx=256
    )
    meta = json.loads((tmp_path / "base.dat.meta.json").read_text())
    assert meta["ctx"] == 256 and meta["chunks"] == 4


def test_save_base_logits_omits_kl_divergence_flag(
    fake_binaries, monkeypatch, tmp_path
):
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    evaluate.save_base_logits(
        tmp_path / "m.gguf", tmp_path / "e.txt", tmp_path / "base.dat", chunks=4
    )
    assert "--kl-divergence" not in recorder.last
    assert "--kl-divergence-base" in recorder.last


def test_kld_requires_existing_base_logits(fake_binaries, tmp_path):
    with pytest.raises(quantize.ToolchainError, match="run save_base_logits first"):
        evaluate.kld(
            tmp_path / "q.gguf", tmp_path / "e.txt", tmp_path / "missing.dat", chunks=4
        )


def test_kld_rejects_context_mismatch(fake_binaries, tmp_path):
    base = tmp_path / "base.dat"
    base.write_bytes(b"x")
    (tmp_path / "base.dat.meta.json").write_text(json.dumps({"ctx": 512}))
    with pytest.raises(quantize.ToolchainError, match="context mismatch"):
        evaluate.kld(tmp_path / "q.gguf", tmp_path / "e.txt", base, chunks=4, ctx=256)


def test_kld_parses_metrics(fake_binaries, monkeypatch, tmp_path):
    base = tmp_path / "base.dat"
    base.write_bytes(b"x")
    (tmp_path / "base.dat.meta.json").write_text(json.dumps({"ctx": 512}))
    monkeypatch.setattr(subprocess, "run", Recorder(stdout=KLD_OUTPUT))
    result = evaluate.kld(
        tmp_path / "q.gguf", tmp_path / "e.txt", base, chunks=4, ctx=512
    )
    assert result.mean_kld == 0.012345
    assert result.same_top_pct == 98.765
    assert result.stdout == KLD_OUTPUT


def test_kld_raises_when_output_unparseable(fake_binaries, monkeypatch, tmp_path):
    base = tmp_path / "base.dat"
    base.write_bytes(b"x")
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="no stats here"))
    with pytest.raises(quantize.ToolchainError, match="could not parse mean KLD"):
        evaluate.kld(tmp_path / "q.gguf", tmp_path / "e.txt", base, chunks=4)


def test_parse_bench_separates_prompt_and_generation():
    speeds = evaluate.parse_bench(BENCH_JSON)
    assert speeds == {"prompt_tps": 1234.56, "gen_tps": 45.67}


def test_bench_keeps_stderr_unmerged(fake_binaries, monkeypatch, tmp_path):
    recorder = Recorder(stdout=BENCH_JSON, stderr="ggml_metal_device_init: noise")
    monkeypatch.setattr(subprocess, "run", recorder)
    speeds = evaluate.bench(tmp_path / "q.gguf")
    assert speeds["gen_tps"] == 45.67
    assert "-o" in recorder.last and "json" in recorder.last


def test_bench_reports_unparseable_json(fake_binaries, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="not json at all"))
    with pytest.raises(quantize.ToolchainError, match="could not parse"):
        evaluate.bench(tmp_path / "q.gguf")


def test_run_can_merge_or_split_stderr(fake_binaries, monkeypatch):
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="out", stderr="err"))
    merged = quantize.run(["llama-bench", "-m", "x"])
    split = quantize.run(["llama-bench", "-m", "x"], merge_stderr=False)
    assert merged.stdout == "out"
    assert split.stderr == "err"
