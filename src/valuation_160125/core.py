from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from collections.abc import Iterable, Mapping, Sequence


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


def disclosed_holdings_return(
    holdings: Iterable[Mapping[str, float]],
    residual_return: float,
) -> tuple[float, float]:
    """Mark disclosed holdings and use a proxy for the undisclosed residual.

    Public-fund reports normally disclose the top holdings, not the complete
    intraday portfolio.  ``weight`` is expressed as a fraction of NAV and
    ``return`` is the local-currency-to-RMB total return of that security.
    The undisclosed weight is explicitly assigned to ``residual_return`` so
    the estimate remains fully invested instead of silently renormalising the
    disclosed top holdings to 100%.
    """
    if not isfinite(residual_return):
        raise ValueError("residual_return must be finite")
    covered_weight = 0.0
    marked_return = 0.0
    for holding in holdings:
        weight = float(holding["weight"])
        security_return = float(holding["return"])
        if not isfinite(weight) or not isfinite(security_return):
            raise ValueError("holding weights and returns must be finite")
        if weight < 0 or weight > 1:
            raise ValueError("holding weight must be between 0 and 1")
        covered_weight += weight
        marked_return += weight * security_return
    if covered_weight > 1.0 + 1e-9:
        raise ValueError("disclosed holding weights cannot exceed 100%")
    covered_weight = min(1.0, covered_weight)
    return marked_return + (1.0 - covered_weight) * residual_return, covered_weight


def calibrated_return(
    current_proxy_return: float,
    historical_fund_returns: Sequence[float],
    historical_proxy_returns: Sequence[float],
    *,
    min_observations: int = 20,
) -> tuple[float, dict[str, float | int]] | None:
    """Forecast today's fund return with a bounded rolling proxy regression."""
    if not isfinite(current_proxy_return):
        raise ValueError("current_proxy_return must be finite")
    pairs = [
        (float(fund), float(proxy))
        for fund, proxy in zip(historical_fund_returns, historical_proxy_returns)
        if isfinite(float(fund)) and isfinite(float(proxy))
    ]
    if len(pairs) < min_observations:
        return None
    fund_values = [pair[0] for pair in pairs]
    proxy_values = [pair[1] for pair in pairs]
    fund_mean = mean(fund_values)
    proxy_mean = mean(proxy_values)
    variance = sum((value - proxy_mean) ** 2 for value in proxy_values)
    if variance <= 1e-16:
        return None
    covariance = sum(
        (fund - fund_mean) * (proxy - proxy_mean)
        for fund, proxy in pairs
    )
    beta = covariance / variance
    beta = max(0.25, min(1.75, beta))
    alpha = fund_mean - beta * proxy_mean
    forecast = alpha + beta * current_proxy_return
    return forecast, {
        "observations": len(pairs),
        "alpha": alpha,
        "beta": beta,
    }


def ensemble_return(
    method_returns: Mapping[str, float],
    method_weights: Mapping[str, float],
) -> float:
    """Combine available return methods after normalising their weights."""
    selected = [
        (name, float(method_returns[name]), float(method_weights.get(name, 0.0)))
        for name in method_returns
        if name in method_weights and method_weights[name] > 0
    ]
    if not selected:
        raise ValueError("at least one weighted return method is required")
    if any(not isfinite(value) or weight < 0 for _, value, weight in selected):
        raise ValueError("method returns and weights must be finite")
    total_weight = sum(weight for _, _, weight in selected)
    if total_weight <= 0:
        raise ValueError("method weights must have positive total")
    return sum(value * weight for _, value, weight in selected) / total_weight


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
