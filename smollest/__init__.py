import json
from pathlib import Path


def _read_version() -> str:
    package_json = Path(__file__).with_name("package.json")
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
        return data["version"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "0.0.0"


__version__ = _read_version()

from smollest import anthropic, openai  # noqa: E402
from smollest.defaults import get_default_candidates  # noqa: E402
from smollest.metrics import (  # noqa: E402
    clear_secondary_metrics,
    register_secondary_metric,
)
from smollest.results import report  # noqa: E402
from smollest.web import show  # noqa: E402

__all__ = [
    "openai",
    "anthropic",
    "report",
    "show",
    "register_secondary_metric",
    "clear_secondary_metrics",
    "get_default_candidates",
]
