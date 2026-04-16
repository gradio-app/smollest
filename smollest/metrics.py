from __future__ import annotations

from typing import Callable

SecondaryMetricCallback = Callable[[dict], dict[str, float | int | str | bool | None]]

_callbacks: list[SecondaryMetricCallback] = []


def register_secondary_metric(callback: SecondaryMetricCallback) -> None:
    _callbacks.append(callback)


def clear_secondary_metrics() -> None:
    _callbacks.clear()


def compute_secondary_metrics(
    payload: dict,
) -> dict[str, float | int | str | bool | None]:
    metrics: dict[str, float | int | str | bool | None] = {}
    for callback in _callbacks:
        result = callback(payload) or {}
        for key, value in result.items():
            metrics[key] = value
    return metrics
