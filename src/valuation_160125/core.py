from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class CrossCheck:
    status: str
    left: float | None
    right: float | None
    delta: float | None


def benchmark_return(
    previous_hsi: float,
    current_hsi: float,
    previous_hkd_cny: float = 1.0,
    current_hkd_cny: float = 1.0,
    equity_weight: float = 0.95,
    cash_return: float = 0.0,
) -> float:
    """Return the RMB-marked benchmark return for one trading day."""
    if not all(isfinite(value) for value in (previous_hsi, current_hsi)):
        raise ValueError("HSI levels must be finite")
    if not all(isfinite(value) for value in (previous_hkd_cny, current_hkd_cny)):
        raise ValueError("HKD/CNY levels must be finite")
    if not isfinite(cash_return):
        raise ValueError("cash_return must be finite")
    if previous_hsi <= 0 or current_hsi <= 0:
        raise ValueError("HSI levels must be positive")
    if previous_hkd_cny <= 0 or current_hkd_cny <= 0:
        raise ValueError("HKD/CNY levels must be positive")
    if not 0 <= equity_weight <= 1:
        raise ValueError("equity_weight must be between 0 and 1")

    equity_return = (current_hsi / previous_hsi) * (
        current_hkd_cny / previous_hkd_cny
    ) - 1.0
    return equity_weight * equity_return + (1.0 - equity_weight) * cash_return


def estimate_nav(previous_nav: float, daily_return: float) -> float:
    if not isfinite(previous_nav) or not isfinite(daily_return):
        raise ValueError("NAV and return must be finite")
    if previous_nav <= 0:
        raise ValueError("previous_nav must be positive")
    return previous_nav * (1.0 + daily_return)


def numeric_cross_check(
    left: float | None,
    right: float | None,
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-4,
) -> CrossCheck:
    if left is None or right is None:
        return CrossCheck("missing", left, right, None)
    if not isfinite(float(left)) or not isfinite(float(right)):
        return CrossCheck("invalid", float(left), float(right), None)
    delta = float(right) - float(left)
    tolerance = max(absolute_tolerance, abs(float(left)) * relative_tolerance)
    status = "match" if abs(delta) <= tolerance else "mismatch"
    return CrossCheck(status, float(left), float(right), delta)


def summarize_backtest_errors(errors: Iterable[float]) -> dict[str, float | int | None]:
    values = [abs(float(error)) for error in errors if isfinite(float(error))]
    if not values:
        return {
            "observations": 0,
            "mae": None,
            "max_abs_error": None,
            "p90_abs_error": None,
        }
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, max(0, int(0.9 * len(ordered)) - 1))
    return {
        "observations": len(values),
        "mae": mean(values),
        "max_abs_error": max(values),
        "p90_abs_error": ordered[p90_index],
    }
