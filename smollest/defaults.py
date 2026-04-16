from __future__ import annotations

from datetime import datetime, timezone

DEFAULT_CANDIDATES = [
    "Qwen/Qwen3.5-3B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
]

MODEL_PRESETS_BY_MONTH: dict[str, list[str]] = {
    "2026-04": [
        "Qwen/Qwen3.5-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ],
    "2026-03": [
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501",
    ],
}

COST_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-20250514": {"input": 0.80, "output": 4.00},
}


def _latest_preset_month() -> str:
    now = datetime.now(timezone.utc)
    key = f"{now.year:04d}-{now.month:02d}"
    if key in MODEL_PRESETS_BY_MONTH:
        return key
    return sorted(MODEL_PRESETS_BY_MONTH.keys())[-1]


def get_default_candidates(month: str | None = None) -> list[str]:
    selected_month = month or _latest_preset_month()
    if selected_month in MODEL_PRESETS_BY_MONTH:
        return list(MODEL_PRESETS_BY_MONTH[selected_month])
    return list(DEFAULT_CANDIDATES)


def infer_model_size_bucket(model: str) -> str:
    normalized = model.lower()
    if "nano" in normalized:
        return "nano"
    if "mini" in normalized or "small" in normalized or "-3b" in normalized:
        return "small"
    if "-7b" in normalized or "-8b" in normalized or "-9b" in normalized:
        return "medium"
    if "-13b" in normalized or "-14b" in normalized or "-17b" in normalized:
        return "large"
    if "-24b" in normalized or "-32b" in normalized or "-70b" in normalized:
        return "xlarge"
    if "haiku" in normalized:
        return "small"
    if "sonnet" in normalized:
        return "large"
    if "opus" in normalized:
        return "xlarge"
    return "unknown"


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = COST_PER_1M_TOKENS.get(model)
    if pricing is None:
        for key, val in COST_PER_1M_TOKENS.items():
            if key in model or model in key:
                pricing = val
                break
    if pricing is None:
        return None
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000
