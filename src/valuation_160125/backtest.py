from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from .cli import _normalise_holdings, _sorted_frame
from .core import (
    adaptive_method_weights,
    benchmark_return,
    calibrated_return,
    disclosed_holdings_return,
    ensemble_return,
    estimate_nav,
    summarize_backtest_errors,
)
from .providers import AKShareProvider, ProviderError, TushareProvider


def _close_series(frame: pd.DataFrame, date_column: str = "trade_date") -> dict[date, float]:
    result = _sorted_frame(frame, date_column)
    if result.empty or "close" not in result.columns:
        return {}
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["close"])
    return {
        row[date_column]: float(row["close"])
        for _, row in result.iterrows()
        if float(row["close"]) > 0
    }


def _fx_series(frame: pd.DataFrame) -> dict[date, float]:
    result = _sorted_frame(frame, "trade_date")
    if result.empty:
        return {}
    close_column = "bid_close" if "bid_close" in result.columns else "close"
    result["fx"] = pd.to_numeric(result.get(close_column), errors="coerce")
    result = result.dropna(subset=["fx"])
    return {
        row["trade_date"]: float(row["fx"]) / 7.8
        for _, row in result.iterrows()
        if float(row["fx"]) > 0
    }


def _daily_returns(series: dict[date, float]) -> dict[date, float]:
    dates = sorted(series)
    return {
        dates[index]: series[dates[index]] / series[dates[index - 1]] - 1.0
        for index in range(1, len(dates))
        if series[dates[index - 1]] > 0 and series[dates[index]] > 0
    }


def _rmb_price_returns(
    frame: pd.DataFrame,
    fx: dict[date, float],
) -> dict[date, float]:
    local_prices = _close_series(frame)
    dates = sorted(local_prices)
    result: dict[date, float] = {}
    for index in range(1, len(dates)):
        previous_date = dates[index - 1]
        current_date = dates[index]
        if current_date not in fx or previous_date not in fx:
            continue
        result[current_date] = (
            (local_prices[current_date] / local_prices[previous_date])
            * (fx[current_date] / fx[previous_date])
            - 1.0
        )
    return result


def _snapshot_reports(
    akshare: AKShareProvider,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for year in range(start_date.year - 1, end_date.year + 1):
        payload = akshare.fund_holdings(str(year))
        for report in payload.get("reports", []):
            report_date = date.fromisoformat(str(report["as_of_date"]))
            if report_date > end_date:
                continue
            # Conservative no-lookahead rule: use a quarterly snapshot only
            # 22 calendar days after its reporting date.
            reports.append(
                {
                    "report_date": report_date,
                    "available_date": report_date + timedelta(days=22),
                    "data": _normalise_holdings(report["data"]),
                }
            )
    unique: dict[date, dict[str, Any]] = {}
    for report in reports:
        unique[report["report_date"]] = report
    return sorted(unique.values(), key=lambda item: item["available_date"])


def _snapshot_for_date(
    reports: list[dict[str, Any]],
    target_date: date,
) -> dict[str, Any] | None:
    available = [report for report in reports if report["available_date"] <= target_date]
    return available[-1] if available else None


def _metric_rows(rows: list[dict[str, Any]], method: str) -> dict[str, float | int | None]:
    selected = [row for row in rows if row.get(f"{method}_nav") is not None]
    if not selected:
        return {
            "observations": 0,
            "mae_nav": None,
            "rmse_nav": None,
            "mae_return_pp": None,
            "mae_bps": None,
            "p90_bps": None,
            "bias_nav": None,
            "direction_accuracy": None,
        }
    nav_errors = [float(row[f"{method}_nav"]) - float(row["actual_nav"]) for row in selected]
    return_errors = [float(row[f"{method}_return"]) - float(row["actual_return"]) for row in selected]
    bps_errors = [abs(error / float(row["actual_nav"]) * 10_000) for error, row in zip(nav_errors, selected)]
    abs_summary = summarize_backtest_errors(nav_errors)
    direction = [
        (float(row[f"{method}_return"]) >= 0) == (float(row["actual_return"]) >= 0)
        for row in selected
        if float(row["actual_return"]) != 0
    ]
    return {
        "observations": len(selected),
        "mae_nav": abs_summary["mae"],
        "rmse_nav": sqrt(sum(error * error for error in nav_errors) / len(nav_errors)),
        "mae_return_pp": sum(abs(error) for error in return_errors) / len(return_errors) * 100,
        "mae_bps": sum(bps_errors) / len(bps_errors),
        "p90_bps": sorted(bps_errors)[min(len(bps_errors) - 1, int(0.9 * len(bps_errors)) - 1)],
        "bias_nav": sum(nav_errors) / len(nav_errors),
        "direction_accuracy": sum(direction) / len(direction) if direction else None,
    }


def run_backtest(
    start_date: date,
    end_date: date,
    *,
    token: str | None = None,
) -> dict[str, Any]:
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")
    tushare = TushareProvider(token=token)
    akshare = AKShareProvider()
    fetch_start = start_date - timedelta(days=450)
    nav_frame = tushare.fund_nav(fetch_start, end_date)
    hsi_frame = tushare.hsi(fetch_start, end_date)
    fx_frame = tushare.usdcnh(fetch_start, end_date)
    nav = _sorted_frame(nav_frame, "nav_date")
    nav["unit_nav"] = pd.to_numeric(nav["unit_nav"], errors="coerce")
    nav = nav.dropna(subset=["unit_nav"])
    hsi = _close_series(hsi_frame)
    fx = _fx_series(fx_frame)
    hsi_dates = sorted(hsi)
    proxy_returns: dict[date, float] = {}
    for index in range(1, len(hsi_dates)):
        current_date = hsi_dates[index]
        previous_date = hsi_dates[index - 1]
        current_fx = fx.get(current_date, fx.get(previous_date, 1.0))
        previous_fx = fx.get(previous_date, current_fx)
        proxy_returns[current_date] = benchmark_return(
            hsi[previous_date],
            hsi[current_date],
            previous_fx,
            current_fx,
        )
    snapshots = _snapshot_reports(akshare, start_date, end_date)
    symbols = sorted(
        {
            str(symbol).zfill(5)
            for snapshot in snapshots
            for symbol in snapshot["data"]["symbol"].tolist()
        }
    )
    price_returns: dict[str, dict[date, float]] = {}
    price_failures: list[str] = []
    def fetch_symbol(symbol: str) -> tuple[str, dict[date, float] | None]:
        try:
            frame = tushare.hk_daily(symbol, fetch_start, end_date)
            return symbol, _rmb_price_returns(frame, fx)
        except ProviderError:
            return symbol, None

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_symbol, symbol) for symbol in symbols]
        for future in as_completed(futures):
            symbol, returns = future.result()
            if returns is None:
                price_failures.append(symbol)
            else:
                price_returns[symbol] = returns

    rows: list[dict[str, Any]] = []
    nav_dates = [d for d in nav["nav_date"].tolist() if start_date < d <= end_date]
    nav_by_date = {row["nav_date"]: float(row["unit_nav"]) for _, row in nav.iterrows()}
    history_returns = [
        nav_by_date[current] / nav_by_date[previous] - 1.0
        for previous, current in zip(sorted(nav_by_date)[:-1], sorted(nav_by_date)[1:])
        if current < start_date and current in proxy_returns and nav_by_date[previous] > 0
    ]
    history_proxy_returns = [
        proxy_returns[current]
        for previous, current in zip(sorted(nav_by_date)[:-1], sorted(nav_by_date)[1:])
        if current < start_date and current in proxy_returns and nav_by_date[previous] > 0
    ]
    for target_date in nav_dates:
        prior_dates = [d for d in nav_by_date if d < target_date]
        if not prior_dates or target_date not in proxy_returns:
            continue
        previous_date = max(prior_dates)
        actual_nav = nav_by_date[target_date]
        previous_nav = nav_by_date[previous_date]
        actual_return = actual_nav / previous_nav - 1.0
        benchmark = proxy_returns[target_date]
        row: dict[str, Any] = {
            "date": target_date.isoformat(),
            "previous_nav": previous_nav,
            "actual_nav": actual_nav,
            "actual_return": actual_return,
            "benchmark_return": benchmark,
            "benchmark_nav": estimate_nav(previous_nav, benchmark),
            "carry_return": 0.0,
            "carry_nav": previous_nav,
        }
        snapshot = _snapshot_for_date(snapshots, target_date)
        if snapshot is not None:
            marked: list[dict[str, float]] = []
            for _, holding in snapshot["data"].iterrows():
                symbol = str(holding["symbol"]).zfill(5)
                if symbol in price_returns and target_date in price_returns[symbol]:
                    marked.append(
                        {
                            "weight": float(holding["weight_pct"]) / 100.0,
                            "return": price_returns[symbol][target_date],
                        }
                    )
            if marked:
                holdings_ret, covered = disclosed_holdings_return(marked, benchmark)
                row["disclosed_holdings_return"] = holdings_ret
                row["disclosed_holdings_nav"] = estimate_nav(previous_nav, holdings_ret)
                row["holdings_covered_weight"] = covered
                row["holdings_snapshot_date"] = snapshot["report_date"].isoformat()
        if "disclosed_holdings_return" in row:
            prior_fund = [
                float(previous["actual_return"])
                for previous in rows[-120:]
                if previous.get("disclosed_holdings_return") is not None
            ]
            prior_holdings = [
                float(previous["disclosed_holdings_return"])
                for previous in rows[-120:]
                if previous.get("disclosed_holdings_return") is not None
            ]
            holdings_calibration = calibrated_return(
                float(row["disclosed_holdings_return"]),
                prior_fund,
                prior_holdings,
            )
            if holdings_calibration is not None:
                row["holdings_calibration_return"] = holdings_calibration[0]
                row["holdings_calibration_nav"] = estimate_nav(
                    previous_nav, holdings_calibration[0]
                )
        historical = history_returns + [
            prior["actual_return"]
            for prior in rows[-400:]
            if prior.get("actual_return") is not None
        ]
        historical_proxy = history_proxy_returns + [
            prior["benchmark_return"]
            for prior in rows[-400:]
            if prior.get("benchmark_return") is not None
        ]
        calibration = calibrated_return(benchmark, historical, historical_proxy)
        if calibration is not None:
            row["historical_calibration_return"] = calibration[0]
            row["historical_calibration_nav"] = estimate_nav(previous_nav, calibration[0])
        methods = {"benchmark": benchmark, "carry": 0.0}
        base_weights = {"benchmark": 0.65, "carry": 0.15}
        if "disclosed_holdings_return" in row:
            methods["disclosed_holdings"] = row["disclosed_holdings_return"]
            base_weights["disclosed_holdings"] = 0.02
        if "historical_calibration_return" in row:
            methods["historical_calibration"] = row["historical_calibration_return"]
            base_weights["historical_calibration"] = 0.18
        historical_errors = {
            method: [
                float(previous.get(f"{method}_return")) - float(previous["actual_return"])
                for previous in rows[-120:]
                if previous.get(f"{method}_return") is not None
            ]
            for method in methods
        }
        weights = adaptive_method_weights(
            methods,
            historical_errors,
            base_weights,
            max_weights={
                "benchmark": 0.80,
                "carry": 0.30,
                "disclosed_holdings": 0.08,
                "historical_calibration": 0.35,
            },
        )
        final = ensemble_return(methods, weights)
        row["ensemble_return"] = final
        row["ensemble_nav"] = estimate_nav(previous_nav, final)
        row["ensemble_weights"] = weights
        rows.append(row)

    metrics = {
        method: _metric_rows(rows, method)
        for method in (
            "benchmark",
            "carry",
            "disclosed_holdings",
            "holdings_calibration",
            "historical_calibration",
            "ensemble",
        )
    }
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "observations": len(rows),
        "holding_snapshots": [
            {
                "report_date": item["report_date"].isoformat(),
                "available_date": item["available_date"].isoformat(),
                "count": len(item["data"]),
            }
            for item in snapshots
        ],
        "symbols": len(symbols),
        "price_failures": price_failures,
        "metrics": metrics,
        "rows": rows,
    }


def write_backtest(result: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{result['start_date']}_{result['end_date']}"
    json_path = target / f"{stem}.json"
    csv_path = target / f"{stem}.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(result["rows"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path
