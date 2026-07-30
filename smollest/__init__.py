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

__all__ = ["__version__"]
