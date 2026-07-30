from __future__ import annotations

import subprocess

import pytest

from smollest import quantize


@pytest.fixture
def fake_binaries(monkeypatch):
    monkeypatch.setattr(quantize.shutil, "which", lambda name: f"/fake/bin/{name}")


class Recorder:
    """Stand-in for subprocess.run that records calls and touches outputs."""

    def __init__(self, returncode=0, stdout="ok", create_output=True):
        self.returncode = returncode
        self.stdout = stdout
        self.create_output = create_output
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if self.create_output and "-o" in args:
            target = args[args.index("-o") + 1]
            open(target, "w").close()
        if self.create_output and args[0].endswith("llama-quantize"):
            open(args[-2], "w").close()
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout)

    @property
    def last(self):
        return self.calls[-1]


def test_require_binary_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(quantize.shutil, "which", lambda name: None)
    with pytest.raises(quantize.ToolchainError, match="brew install llama.cpp"):
        quantize.require_binary("llama-imatrix")


def test_run_raises_on_nonzero_exit(fake_binaries, monkeypatch):
    recorder = Recorder(returncode=1, stdout="boom\nfailed to load model")
    monkeypatch.setattr(subprocess, "run", recorder)
    with pytest.raises(quantize.ToolchainError, match="failed to load model"):
        quantize.run(["llama-imatrix", "-m", "x"])


def test_run_returns_full_stdout(fake_binaries, monkeypatch):
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="line1\nline2"))
    result = quantize.run(["llama-imatrix", "-m", "x"])
    assert result.stdout == "line1\nline2"
    assert result.returncode == 0


def test_compute_imatrix_passes_chunks_and_ctx(fake_binaries, monkeypatch, tmp_path):
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    out = tmp_path / "nested" / "imatrix.gguf"
    result = quantize.compute_imatrix(
        tmp_path / "m.gguf", tmp_path / "c.txt", out, chunks=42, ctx=256
    )
    assert result == out and out.exists()
    assert "--chunks" in recorder.last
    assert recorder.last[recorder.last.index("--chunks") + 1] == "42"
    assert recorder.last[recorder.last.index("-c") + 1] == "256"


def test_compute_imatrix_detects_missing_output(fake_binaries, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", Recorder(create_output=False))
    with pytest.raises(quantize.ToolchainError, match="is missing"):
        quantize.compute_imatrix(
            tmp_path / "m.gguf", tmp_path / "c.txt", tmp_path / "i.gguf"
        )


def test_merge_imatrix_passes_each_input(fake_binaries, monkeypatch, tmp_path):
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    quantize.merge_imatrix([a, b], tmp_path / "merged.gguf")
    assert recorder.last.count("--in-file") == 2
    assert str(a) in recorder.last and str(b) in recorder.last


def test_merge_imatrix_requires_two_inputs(fake_binaries, tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        quantize.merge_imatrix([tmp_path / "a.gguf"], tmp_path / "out.gguf")


def test_quantize_without_imatrix_omits_flag(fake_binaries, monkeypatch, tmp_path):
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    out = tmp_path / "q.gguf"
    quantize.quantize(tmp_path / "m.gguf", "Q3_K_M", out)
    assert "--imatrix" not in recorder.last
    assert recorder.last[-1] == "Q3_K_M"


def test_quantize_with_imatrix_includes_flag(fake_binaries, monkeypatch, tmp_path):
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    imatrix = tmp_path / "i.gguf"
    quantize.quantize(tmp_path / "m.gguf", "Q4_K_M", tmp_path / "q.gguf", imatrix)
    assert recorder.last[recorder.last.index("--imatrix") + 1] == str(imatrix)


def test_quantize_never_allows_requantize(fake_binaries, monkeypatch, tmp_path):
    recorder = Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    quantize.quantize(tmp_path / "m.gguf", "Q3_K_M", tmp_path / "q.gguf")
    assert "--allow-requantize" not in recorder.last


def test_imatrix_applied_detects_load_line():
    applied = quantize.CommandResult(
        args=[],
        stdout="load_imatrix: loaded 219 importance matrix entries",
        returncode=0,
    )
    ignored = quantize.CommandResult(args=[], stdout="quantizing tensors", returncode=0)
    assert quantize.imatrix_applied(applied)
    assert not quantize.imatrix_applied(ignored)


def test_toolchain_version_parses_build(fake_binaries, monkeypatch):
    stdout = (
        "ggml_metal_device_init: ok\nversion: 8140 (39fb81f87)\nbuilt with AppleClang"
    )
    monkeypatch.setattr(subprocess, "run", Recorder(stdout=stdout))
    assert quantize.toolchain_version() == "8140 (39fb81f87)"


def test_toolchain_version_unknown_when_absent(fake_binaries, monkeypatch):
    monkeypatch.setattr(subprocess, "run", Recorder(stdout="no version here"))
    assert quantize.toolchain_version() == "unknown"


def test_model_size(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"abcd")
    assert quantize.model_size(path) == 4
