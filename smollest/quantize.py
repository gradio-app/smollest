"""Wrappers around the llama.cpp quantization binaries.

Everything here shells out to the ``llama-*`` executables rather than binding
libllama, so the toolchain is whatever is on ``PATH``. The build identity is
recorded alongside results because importance-matrix and quantization
behaviour changes between llama.cpp releases, and cross-build comparisons are
not meaningful.

The source model must be F16 or BF16. Requantizing an already-quantized GGUF
compounds error and would invalidate any comparison between variants, so
:func:`quantize` refuses to pass ``--allow-requantize``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

IMATRIX_BIN = "llama-imatrix"
QUANTIZE_BIN = "llama-quantize"
CLI_BIN = "llama-cli"

DEFAULT_CTX = 512
DEFAULT_CHUNKS = 200


class ToolchainError(RuntimeError):
    """A llama.cpp binary is missing or failed."""


@dataclass(frozen=True)
class CommandResult:
    """A completed subprocess, kept whole so parse failures never lose a run."""

    args: list[str]
    stdout: str
    returncode: int
    stderr: str = ""


def require_binary(name: str) -> str:
    """Return the absolute path to a llama.cpp binary, or raise."""
    path = shutil.which(name)
    if path is None:
        raise ToolchainError(
            f"{name} not found on PATH; install llama.cpp (e.g. brew install llama.cpp)"
        )
    return path


def run(
    args: list[str], timeout: float | None = None, merge_stderr: bool = True
) -> CommandResult:
    """Run a llama.cpp command and capture its output.

    Diagnostics go to stderr and results to stdout, so tools whose output is
    parsed as text can merge the two, while tools emitting structured output
    (``llama-bench -o json``) must keep them apart or the JSON is corrupted by
    backend init logging.
    """
    binary = require_binary(args[0])
    completed = subprocess.run(
        [binary, *args[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    result = CommandResult(
        args=list(args),
        stdout=completed.stdout,
        returncode=completed.returncode,
        stderr=completed.stderr or "",
    )
    if completed.returncode != 0:
        raise ToolchainError(
            f"{args[0]} exited {completed.returncode}\n"
            f"{_tail(completed.stdout + (completed.stderr or ''))}"
        )
    return result


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def toolchain_version() -> str:
    """Return the llama.cpp build identity, for recording with every result."""
    result = subprocess.run(
        [require_binary(CLI_BIN), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    match = re.search(r"^version:\s*(.+)$", result.stdout, re.MULTILINE)
    return match.group(1).strip() if match else "unknown"


def compute_imatrix(
    model: Path,
    corpus: Path,
    out: Path,
    chunks: int = DEFAULT_CHUNKS,
    ctx: int = DEFAULT_CTX,
    timeout: float | None = None,
) -> Path:
    """Compute an importance matrix over ``corpus``.

    ``chunks`` caps how much of the corpus is consumed, so holding it constant
    is what keeps the comparison about corpus *content* rather than size.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            IMATRIX_BIN,
            "-m",
            str(model),
            "-f",
            str(corpus),
            "-o",
            str(out),
            "--chunks",
            str(chunks),
            "-c",
            str(ctx),
        ],
        timeout=timeout,
    )
    if not out.exists():
        raise ToolchainError(f"{IMATRIX_BIN} reported success but {out} is missing")
    return out


def merge_imatrix(
    inputs: list[Path], out: Path, model: Path, timeout: float | None = None
) -> Path:
    """Combine importance matrices into one.

    Used for the hybrid variant, which keeps a general-calibration floor under
    the trace-derived matrix so that channels the traces never exercise are
    still covered.

    Inputs are passed as a single comma-separated ``--in-file``: repeating the
    flag is deprecated and silently keeps only the last value, which would
    yield a "merged" matrix identical to one input. ``-m`` is required even
    though no prompt is processed.
    """
    if len(inputs) < 2:
        raise ValueError("merging requires at least two importance matrices")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            IMATRIX_BIN,
            "-m",
            str(model),
            "--in-file",
            ",".join(str(path) for path in inputs),
            "-o",
            str(out),
        ],
        timeout=timeout,
    )
    if not out.exists():
        raise ToolchainError(f"{IMATRIX_BIN} reported success but {out} is missing")
    return out


def quantize(
    model: Path,
    qtype: str,
    out: Path,
    imatrix: Path | None = None,
    timeout: float | None = None,
) -> Path:
    """Quantize an F16/BF16 model to ``qtype``, optionally guided by an imatrix."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [QUANTIZE_BIN]
    if imatrix is not None:
        args += ["--imatrix", str(imatrix)]
    args += [str(model), str(out), qtype]
    run(args, timeout=timeout)
    if not out.exists():
        raise ToolchainError(f"{QUANTIZE_BIN} reported success but {out} is missing")
    return out


def imatrix_applied(result: CommandResult) -> bool:
    """Whether llama-quantize reported loading an importance matrix.

    Guards the failure mode where ``--imatrix`` is accepted but silently
    ignored, which would make every variant identical while still producing a
    plausible results table.
    """
    return "load_imatrix" in result.stdout or "imatrix" in result.stdout.lower()


def model_size(path: Path) -> int:
    """Size of a model file in bytes."""
    return path.stat().st_size
