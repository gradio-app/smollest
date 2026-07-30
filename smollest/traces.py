"""Build calibration and evaluation corpora from local coding-agent transcripts.

Claude Code writes one JSONL transcript per session under ``~/.claude/projects``,
in a directory whose name is derived from the session's working directory. Only
``user`` and ``assistant`` records carry model traffic; the remaining record
types (``queue-operation``, ``system``, ``attachment``, ...) are UI and
bookkeeping noise.

Three recipes select progressively more of that traffic:

``prose``
    User prompts and assistant text only.
``authored``
    Adds tool-call arguments, so everything the model itself produced.
``full``
    Adds tool results, approximating the whole input distribution the model
    sees inside an agent loop.

``thinking`` blocks are persisted without their text, so they contribute
nothing to any recipe regardless of whether a recipe selects them.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TRACE_ROOT = Path.home() / ".claude" / "projects"

RECIPES: dict[str, frozenset[str]] = {
    "prose": frozenset({"text", "thinking"}),
    "authored": frozenset({"text", "thinking", "tool_use"}),
    "full": frozenset({"text", "thinking", "tool_use", "tool_result"}),
}

MIN_BLOCK_CHARS = 40
CHARS_PER_TOKEN = 4
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Transcript:
    """A single session transcript with the metadata both splits need."""

    path: Path
    project: str
    start: str | None
    end: str | None


def estimate_tokens(text: str) -> int:
    """Approximate token count.

    A crude chars-per-token heuristic. llama.cpp does the real tokenization at
    imatrix time, so this only needs to be good enough to hold corpus budgets
    roughly equal across recipes.
    """
    return len(text) // CHARS_PER_TOKEN


def find_transcripts(root: Path | None = None) -> list[Path]:
    """Return every transcript path under ``root``, sorted for determinism."""
    root = root or DEFAULT_TRACE_ROOT
    return sorted(root.glob("**/*.jsonl"))


def _iter_json_lines(path: Path) -> Iterator[dict]:
    with path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def iter_records(paths: Iterable[Path]) -> Iterator[dict]:
    """Yield model-traffic records, skipping noise record types and metadata."""
    for path in paths:
        for record in _iter_json_lines(path):
            if record.get("type") not in ("user", "assistant"):
                continue
            if record.get("isMeta"):
                continue
            yield record


def scan(paths: Iterable[Path]) -> list[Transcript]:
    """Collect per-transcript project and timestamp range in a single pass."""
    transcripts = []
    for path in paths:
        project = path.parent.name
        stamps = []
        for record in _iter_json_lines(path):
            stamp = record.get("timestamp")
            if stamp:
                stamps.append(stamp)
        transcripts.append(
            Transcript(
                path=path,
                project=project,
                start=min(stamps) if stamps else None,
                end=max(stamps) if stamps else None,
            )
        )
    return transcripts


def projects(transcripts: Iterable[Transcript]) -> list[str]:
    """Return distinct project names, most transcripts first."""
    counts: dict[str, int] = {}
    for transcript in transcripts:
        counts[transcript.project] = counts.get(transcript.project, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def pick_holdout_projects(
    transcripts: Iterable[Transcript], frac: float = 0.25, seed: int = 0
) -> list[str]:
    """Choose a deterministic subset of projects to hold out of calibration."""
    names = sorted(projects(transcripts))
    if not names:
        return []
    count = max(1, round(len(names) * frac))
    return sorted(random.Random(seed).sample(names, count))


def split_by_project(
    transcripts: Iterable[Transcript], holdout_projects: Iterable[str]
) -> tuple[list[Transcript], list[Transcript]]:
    """Partition transcripts into (calibration, holdout) by project name.

    The holdout side tests generalization to work the calibration corpus never
    saw. Evaluating on a project that appears in calibration inflates every
    downstream number, so this split is the load-bearing control.
    """
    holdout = set(holdout_projects)
    calib, held = [], []
    for transcript in transcripts:
        (held if transcript.project in holdout else calib).append(transcript)
    return calib, held


def split_by_time(
    transcripts: Iterable[Transcript], cutoff: str
) -> tuple[list[Transcript], list[Transcript]]:
    """Partition transcripts into (before, after) an ISO-8601 cutoff.

    A transcript is "after" when it ends at or past the cutoff, so sessions
    straddling the boundary are treated as recent.
    """
    before, after = [], []
    for transcript in transcripts:
        end = transcript.end
        (after if end is not None and end >= cutoff else before).append(transcript)
    return before, after


def pick_time_cutoff(
    transcripts: Iterable[Transcript], frac: float = 0.15
) -> str | None:
    """Return the cutoff placing roughly ``frac`` of transcripts in the future.

    Calendar cutoffs are useless here because session activity is extremely
    bursty -- picking a round date can leave a handful of transcripts on the
    recent side. Choosing by quantile keeps the temporal holdout usable.
    """
    ends = sorted(t.end for t in transcripts if t.end is not None)
    if not ends:
        return None
    index = min(len(ends) - 1, max(0, round(len(ends) * (1.0 - frac))))
    return ends[index]


def _render_tool_result(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def render_block(block: object, kinds: frozenset[str]) -> str:
    """Render one content block to text, or "" when it is not selected."""
    if isinstance(block, str):
        return block if "text" in kinds else ""
    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind not in kinds:
        return ""
    if kind == "text":
        return block.get("text") or ""
    if kind == "thinking":
        return block.get("thinking") or ""
    if kind == "tool_use":
        name = block.get("name") or ""
        args = json.dumps(block.get("input", {}), ensure_ascii=False, sort_keys=True)
        return f"{name}\n{args}"
    if kind == "tool_result":
        return _render_tool_result(block.get("content"))
    return ""


def iter_blocks(
    paths: Iterable[Path],
    recipe: str = "full",
    include_sidechains: bool = True,
    min_chars: int = MIN_BLOCK_CHARS,
) -> Iterator[str]:
    """Yield rendered text blocks for a recipe, in transcript order."""
    try:
        kinds = RECIPES[recipe]
    except KeyError:
        raise ValueError(
            f"unknown recipe {recipe!r}; expected one of {sorted(RECIPES)}"
        ) from None
    for record in iter_records(paths):
        if record.get("isSidechain") and not include_sidechains:
            continue
        content = (record.get("message") or {}).get("content")
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            text = render_block(block, kinds)
            if len(text) >= min_chars:
                yield text


def dedup(blocks: Iterable[str]) -> Iterator[str]:
    """Drop exact-duplicate blocks, preserving first-seen order."""
    seen: set[str] = set()
    for block in blocks:
        digest = hashlib.sha1(block.encode("utf-8", "ignore")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        yield block


def load_blocks(
    paths: Iterable[Path], recipe: str = "full", include_sidechains: bool = True
) -> list[str]:
    """Read and deduplicate every block once.

    Transcripts are live: an agent session appends to its own transcript while
    it runs, so re-reading between replicates yields a slightly different
    corpus and breaks the disjointness that makes replicates a valid noise
    floor. Read once with this, then take windows with :func:`window`.
    """
    return list(dedup(iter_blocks(paths, recipe, include_sidechains)))


def window(
    blocks: Sequence[str],
    budget_tokens: int | None = None,
    seed: int = 0,
    replicate: int = 0,
) -> list[str]:
    """Take one budget-sized window from an already-loaded block list.

    Blocks are shuffled under ``seed`` and sliced into consecutive windows;
    ``replicate`` selects one. Windows over a single loaded list are disjoint,
    which is what makes them usable as a noise floor rather than as correlated
    resamples.
    """
    if budget_tokens is None:
        return list(blocks)

    shuffled = list(blocks)
    random.Random(seed).shuffle(shuffled)
    budget_chars = budget_tokens * CHARS_PER_TOKEN
    lo = replicate * budget_chars
    hi = lo + budget_chars

    selected: list[str] = []
    used = 0
    for block in shuffled:
        if lo <= used < hi:
            selected.append(block)
        used += len(block) + len(SEPARATOR)
        if used >= hi:
            break
    return selected


def select_blocks(
    paths: Iterable[Path],
    recipe: str = "full",
    budget_tokens: int | None = None,
    seed: int = 0,
    replicate: int = 0,
    include_sidechains: bool = True,
) -> list[str]:
    """Load and window in one step, for single-corpus callers.

    Do not call this once per replicate -- see :func:`load_blocks`.

    Returns blocks rather than joined text so that callers can reason about real
    block boundaries. Blocks may themselves contain the separator, so splitting
    the joined corpus back apart does not recover them.
    """
    blocks = load_blocks(paths, recipe, include_sidechains)
    return window(blocks, budget_tokens, seed, replicate)


def build_corpus(
    paths: Iterable[Path],
    recipe: str = "full",
    budget_tokens: int | None = None,
    seed: int = 0,
    replicate: int = 0,
    include_sidechains: bool = True,
) -> str:
    """Assemble a corpus as the text llama.cpp will read."""
    return SEPARATOR.join(
        select_blocks(paths, recipe, budget_tokens, seed, replicate, include_sidechains)
    )


def corpus_stats(blocks: Sequence[str]) -> dict[str, int]:
    """Summarize selected blocks for logging, so budget parity stays visible."""
    chars = sum(len(b) for b in blocks) + len(SEPARATOR) * max(0, len(blocks) - 1)
    return {
        "chars": chars,
        "est_tokens": chars // CHARS_PER_TOKEN,
        "blocks": len(blocks),
    }


def write_corpus(path: Path, blocks: Sequence[str]) -> dict[str, int]:
    """Write selected blocks to disk for llama.cpp and return their stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SEPARATOR.join(blocks), encoding="utf-8")
    return corpus_stats(blocks)
