from __future__ import annotations

import json

import pytest

from smollest import traces

LONG = "x" * 60
OTHER = "y" * 60


def write_transcript(directory, name, records):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def record(role, blocks, **extra):
    return {"type": role, "message": {"content": blocks}, **extra}


def test_estimate_tokens():
    assert traces.estimate_tokens("a" * 400) == 100
    assert traces.estimate_tokens("") == 0


def test_iter_records_skips_noise_types_and_meta(tmp_path):
    path = write_transcript(
        tmp_path / "proj",
        "s.jsonl",
        [
            record("user", [{"type": "text", "text": LONG}]),
            record("assistant", [{"type": "text", "text": LONG}]),
            {"type": "queue-operation", "content": LONG},
            {"type": "system", "subtype": "hook"},
            record("user", [{"type": "text", "text": LONG}], isMeta=True),
        ],
    )
    assert [r["type"] for r in traces.iter_records([path])] == ["user", "assistant"]


def test_recipes_select_progressively_more(tmp_path):
    blocks = [
        {"type": "text", "text": LONG},
        {"type": "tool_use", "name": "Read", "input": {"file_path": OTHER}},
        {"type": "tool_result", "content": "z" * 60},
    ]
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("assistant", blocks)])

    assert len(list(traces.iter_blocks([path], "prose"))) == 1
    assert len(list(traces.iter_blocks([path], "authored"))) == 2
    assert len(list(traces.iter_blocks([path], "full"))) == 3


def test_unknown_recipe_rejected(tmp_path):
    path = write_transcript(tmp_path / "proj", "s.jsonl", [])
    with pytest.raises(ValueError, match="unknown recipe"):
        list(traces.iter_blocks([path], "nope"))


def test_bare_string_content_treated_as_text(tmp_path):
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("user", LONG)])
    assert list(traces.iter_blocks([path], "prose")) == [LONG]


def test_tool_result_list_form_extracts_text(tmp_path):
    blocks = [
        {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": LONG},
                {"type": "tool_reference", "id": "abc"},
                {"type": "text", "text": OTHER},
            ],
        }
    ]
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("user", blocks)])
    assert list(traces.iter_blocks([path], "full")) == [f"{LONG}\n{OTHER}"]


def test_thinking_blocks_are_empty_in_practice(tmp_path):
    blocks = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("assistant", blocks)])
    assert list(traces.iter_blocks([path], "prose")) == []


def test_short_blocks_filtered(tmp_path):
    blocks = [{"type": "text", "text": "tiny"}, {"type": "text", "text": LONG}]
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("assistant", blocks)])
    assert list(traces.iter_blocks([path], "prose")) == [LONG]


def test_image_blocks_ignored(tmp_path):
    blocks = [{"type": "image", "source": {"data": "b64" * 100}}]
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("user", blocks)])
    assert list(traces.iter_blocks([path], "full")) == []


def test_sidechains_can_be_excluded(tmp_path):
    path = write_transcript(
        tmp_path / "proj",
        "s.jsonl",
        [
            record("assistant", [{"type": "text", "text": LONG}], isSidechain=True),
            record("assistant", [{"type": "text", "text": OTHER}]),
        ],
    )
    assert len(list(traces.iter_blocks([path], "prose"))) == 2
    assert list(traces.iter_blocks([path], "prose", include_sidechains=False)) == [
        OTHER
    ]


def test_tool_use_renders_name_and_sorted_args(tmp_path):
    blocks = [{"type": "tool_use", "name": "Bash", "input": {"b": 2, "a": LONG}}]
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("assistant", blocks)])
    (rendered,) = list(traces.iter_blocks([path], "authored"))
    assert rendered.startswith("Bash\n")
    assert rendered.index('"a"') < rendered.index('"b"')


def test_dedup_preserves_first_seen_order():
    assert list(traces.dedup([LONG, OTHER, LONG])) == [LONG, OTHER]


def test_scan_collects_project_and_time_range(tmp_path):
    path = write_transcript(
        tmp_path / "-Users-me-dev-thing",
        "s.jsonl",
        [
            record("user", LONG, timestamp="2026-07-02T10:00:00Z"),
            record("assistant", LONG, timestamp="2026-07-05T10:00:00Z"),
        ],
    )
    (meta,) = traces.scan([path])
    assert meta.project == "-Users-me-dev-thing"
    assert meta.start == "2026-07-02T10:00:00Z"
    assert meta.end == "2026-07-05T10:00:00Z"


def test_scan_handles_missing_timestamps(tmp_path):
    path = write_transcript(tmp_path / "proj", "s.jsonl", [record("user", LONG)])
    (meta,) = traces.scan([path])
    assert meta.start is None and meta.end is None


def test_split_by_project_partitions_on_holdout(tmp_path):
    a = write_transcript(tmp_path / "alpha", "s.jsonl", [record("user", LONG)])
    b = write_transcript(tmp_path / "beta", "s.jsonl", [record("user", LONG)])
    calib, held = traces.split_by_project(traces.scan([a, b]), ["beta"])
    assert [t.project for t in calib] == ["alpha"]
    assert [t.project for t in held] == ["beta"]


def test_pick_holdout_projects_is_deterministic(tmp_path):
    paths = [
        write_transcript(tmp_path / name, "s.jsonl", [record("user", LONG)])
        for name in ("a", "b", "c", "d")
    ]
    scanned = traces.scan(paths)
    first = traces.pick_holdout_projects(scanned, frac=0.5, seed=7)
    assert first == traces.pick_holdout_projects(scanned, frac=0.5, seed=7)
    assert len(first) == 2


def test_split_by_time_treats_straddling_sessions_as_recent(tmp_path):
    old = write_transcript(
        tmp_path / "p",
        "old.jsonl",
        [record("user", LONG, timestamp="2026-07-01T00:00:00Z")],
    )
    new = write_transcript(
        tmp_path / "p",
        "new.jsonl",
        [
            record("user", LONG, timestamp="2026-07-01T00:00:00Z"),
            record("user", OTHER, timestamp="2026-07-20T00:00:00Z"),
        ],
    )
    before, after = traces.split_by_time(traces.scan([old, new]), "2026-07-10")
    assert [t.path.name for t in before] == ["old.jsonl"]
    assert [t.path.name for t in after] == ["new.jsonl"]


def test_build_corpus_without_budget_includes_everything(tmp_path):
    path = write_transcript(
        tmp_path / "p",
        "s.jsonl",
        [
            record(
                "user",
                [{"type": "text", "text": LONG}, {"type": "text", "text": OTHER}],
            )
        ],
    )
    corpus = traces.build_corpus([path], "prose")
    assert corpus == f"{LONG}{traces.SEPARATOR}{OTHER}"


def test_build_corpus_respects_budget(tmp_path):
    blocks = [{"type": "text", "text": str(i) * 100} for i in range(10)]
    path = write_transcript(tmp_path / "p", "s.jsonl", [record("user", blocks)])
    corpus = traces.build_corpus([path], "prose", budget_tokens=50)
    assert traces.estimate_tokens(corpus) <= 60


def test_replicates_are_disjoint(tmp_path):
    blocks = [{"type": "text", "text": str(i) * 100} for i in range(30)]
    path = write_transcript(tmp_path / "p", "s.jsonl", [record("user", blocks)])
    loaded = traces.load_blocks([path], "prose")
    windows = [traces.window(loaded, budget_tokens=50, replicate=r) for r in range(3)]
    assert all(windows)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not set(windows[i]) & set(windows[j])


def test_replicates_stay_disjoint_when_blocks_contain_separator(tmp_path):
    blocks = [
        {"type": "text", "text": f"{str(i) * 60}{traces.SEPARATOR}{str(i) * 60}"}
        for i in range(30)
    ]
    path = write_transcript(tmp_path / "p", "s.jsonl", [record("user", blocks)])
    loaded = traces.load_blocks([path], "prose")
    first = traces.window(loaded, budget_tokens=60, replicate=0)
    second = traces.window(loaded, budget_tokens=60, replicate=1)
    assert first and second
    assert not set(first) & set(second)


def test_window_covers_every_block_across_replicates(tmp_path):
    blocks = [{"type": "text", "text": f"{i:02d}" * 50} for i in range(12)]
    path = write_transcript(tmp_path / "p", "s.jsonl", [record("user", blocks)])
    loaded = traces.load_blocks([path], "prose")
    assert len(loaded) == 12
    collected = []
    for r in range(len(loaded) + 2):
        collected.extend(traces.window(loaded, budget_tokens=25, replicate=r))
    assert set(collected) == set(loaded)
    assert len(collected) == len(loaded)


def test_window_without_budget_returns_a_copy():
    loaded = [LONG, OTHER]
    result = traces.window(loaded)
    assert result == loaded and result is not loaded


def test_corpus_stats_and_write(tmp_path):
    out = tmp_path / "nested" / "corpus.txt"
    stats = traces.write_corpus(out, [LONG, OTHER])
    assert out.read_text(encoding="utf-8") == f"{LONG}{traces.SEPARATOR}{OTHER}"
    assert stats["blocks"] == 2
    assert stats["chars"] == len(LONG) + len(traces.SEPARATOR) + len(OTHER)


def test_corpus_stats_counts_blocks_not_separator_fragments():
    stats = traces.corpus_stats([f"a{traces.SEPARATOR}b", "c"])
    assert stats["blocks"] == 2


def test_pick_time_cutoff_targets_a_usable_fraction(tmp_path):
    paths = [
        write_transcript(
            tmp_path / "p",
            f"s{i}.jsonl",
            [record("user", LONG, timestamp=f"2026-07-{i + 1:02d}T00:00:00Z")],
        )
        for i in range(20)
    ]
    scanned = traces.scan(paths)
    cutoff = traces.pick_time_cutoff(scanned, frac=0.25)
    before, after = traces.split_by_time(scanned, cutoff)
    assert len(after) == 5
    assert len(before) == 15


def test_pick_time_cutoff_without_timestamps(tmp_path):
    path = write_transcript(tmp_path / "p", "s.jsonl", [record("user", LONG)])
    assert traces.pick_time_cutoff(traces.scan([path])) is None


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "p" / "s.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(["not json", "", json.dumps(record("user", LONG)), "[1,2,3]"]),
        encoding="utf-8",
    )
    assert list(traces.iter_blocks([path], "prose")) == [LONG]
