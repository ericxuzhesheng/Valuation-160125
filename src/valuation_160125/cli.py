from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .core import CrossCheck, benchmark_return, estimate_nav, numeric_cross_check
from .emailer import send_report_email
from .providers import AKShareProvider, ProviderError, TushareProvider
from .report import render_markdown, write_report


def _date(value: str | None) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else date.today()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _check_to_dict(check: CrossCheck) -> dict[str, Any]:
    return {
        "status": check.status,
        "left": check.left,
        "right": check.right,
        "delta": check.delta,
    }


def _sorted_frame(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column]).dt.date
    return result.sort_values(date_column)


def _latest_published_nav(nav_frame: pd.DataFrame, as_of: date) -> tuple[date, float]:
    frame = _sorted_frame(nav_frame, "nav_date")
    if frame.empty:
        raise ProviderError("Tushare returned no fund NAV rows")
    frame = frame[frame["nav_date"] <= as_of]
    if frame.empty:
        raise ProviderError("no published NAV exists before the requested date")
    row = frame.iloc[-1]
    nav = _float(row.get("unit_nav"))
    if nav is None:
        raise ProviderError("published NAV is not numeric")
    return row["nav_date"], nav


def _hsi_pair(frame: pd.DataFrame, as_of: date, date_column: str) -> tuple[date, float, float, float]:
    frame = _sorted_frame(frame, date_column)
    if frame.empty:
        raise ProviderError("HSI data is empty")
    current = frame[frame[date_column] == as_of]
    previous = frame[frame[date_column] < as_of]
    if current.empty or previous.empty:
        raise ProviderError(f"HSI has no current/previous pair for {as_of}")
    current_row = current.iloc[-1]
    previous_row = previous.iloc[-1]
    current_close = _float(current_row.get("close"))
    previous_close = _float(previous_row.get("close"))
    if current_close is None or previous_close is None:
        raise ProviderError("HSI close is not numeric")
    return previous_row[date_column], previous_close, current_close, current_close / previous_close - 1


def _usdcnh_levels(frame: pd.DataFrame, as_of: date) -> tuple[float, float] | None:
    if frame.empty or "trade_date" not in frame.columns:
        return None
    frame = _sorted_frame(frame, "trade_date")
    frame = frame[frame["trade_date"] <= as_of]
    if len(frame) < 2:
        return None
    close_column = "bid_close" if "bid_close" in frame.columns else "close"
    previous = _float(frame.iloc[-2].get(close_column))
    current = _float(frame.iloc[-1].get(close_column))
    if previous is None or current is None or previous <= 0 or current <= 0:
        return None
    # HKD is effectively pegged to USD; HKD/CNY is proportional to 1/USDCNH.
    return 7.8 / previous, 7.8 / current


def build_live_report(as_of: date) -> dict[str, Any]:
    lookback = as_of - timedelta(days=14)
    token = os.environ.get("TUSHARE_TOKEN")
    tushare = TushareProvider(token=token)
    akshare = AKShareProvider()

    nav_frame = tushare.fund_nav(lookback, as_of)
    published_date, published_nav = _latest_published_nav(nav_frame, as_of - timedelta(days=1))
    tushare_hsi = tushare.hsi(lookback, as_of)
    previous_date, previous_hsi, current_hsi, _ = _hsi_pair(
        tushare_hsi, as_of, "trade_date"
    )

    notes: list[str] = []
    fx_frame = pd.DataFrame()
    fx_levels = None
    try:
        fx_frame = tushare.usdcnh(lookback, as_of)
        fx_levels = _usdcnh_levels(fx_frame, as_of)
    except ProviderError as exc:
        notes.append(str(exc))
    if fx_levels is None:
        fx_levels = (1.0, 1.0)
        notes.append("汇率数据不可用，估算暂不计入 HKD/CNY 当日变化")

    daily_return = benchmark_return(
        previous_hsi=previous_hsi,
        current_hsi=current_hsi,
        previous_hkd_cny=fx_levels[0],
        current_hkd_cny=fx_levels[1],
        equity_weight=0.95,
    )
    estimate = estimate_nav(published_nav, daily_return)

    checks: dict[str, dict[str, Any]] = {}
    try:
        ak_hsi = akshare.hsi()
        _, _, ak_current_hsi, _ = _hsi_pair(ak_hsi, as_of, "date")
        checks["hsi_close"] = _check_to_dict(
            numeric_cross_check(current_hsi, ak_current_hsi, absolute_tolerance=0.02)
        )
    except ProviderError as exc:
        checks["hsi_close"] = _check_to_dict(numeric_cross_check(current_hsi, None))
        notes.append(str(exc))

    try:
        ak_nav = akshare.fund_daily_snapshot()
        checks["fund_nav"] = _check_to_dict(
            numeric_cross_check(
                published_nav,
                _float(ak_nav.get("previous_nav")),
                absolute_tolerance=0.0001,
            )
        )
    except ProviderError as exc:
        checks["fund_nav"] = _check_to_dict(numeric_cross_check(published_nav, None))
        notes.append(str(exc))

    statuses = [check["status"] for check in checks.values()]
    confidence = "medium" if statuses and all(status == "match" for status in statuses) else "low"
    notes.append("最新持仓未纳入本版估算，当前使用恒生指数 95% + 现金 5% 基准兜底")
    return {
        "as_of_date": as_of.isoformat(),
        "fund_code": "160125",
        "published_nav_date": published_date.isoformat(),
        "published_nav": published_nav,
        "nav_estimate": estimate,
        "estimate_mode": "benchmark_fallback",
        "confidence": confidence,
        "previous_hsi": previous_hsi,
        "current_hsi": current_hsi,
        "previous_hsi_date": previous_date.isoformat(),
        "daily_return": daily_return,
        "source_checks": checks,
        "notes": notes,
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def build_demo_report() -> dict[str, Any]:
    report = {
        "as_of_date": "2026-07-31",
        "fund_code": "160125",
        "published_nav_date": "2026-07-30",
        "published_nav": 1.5568,
        "nav_estimate": estimate_nav(
            1.5568,
            benchmark_return(25_858.88086, 25_884.42969),
        ),
        "estimate_mode": "benchmark_fallback",
        "confidence": "low",
        "previous_hsi": 25_858.88086,
        "current_hsi": 25_884.42969,
        "previous_hsi_date": "2026-07-30",
        "daily_return": benchmark_return(25_858.88086, 25_884.42969),
        "source_checks": {
            "fund_nav": _check_to_dict(numeric_cross_check(1.5568, 1.5568)),
            "hsi_close": _check_to_dict(
                numeric_cross_check(25_884.43, 25_884.42969, absolute_tolerance=0.02)
            ),
        },
        "notes": ["离线演示数据，不调用外部接口", "未计入汇率与实际持仓偏离"],
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", dest="as_of", help="估值日期，YYYY-MM-DD")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--offline-demo", action="store_true")
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = build_demo_report() if args.offline_demo else build_live_report(_date(args.as_of))
    except Exception as exc:
        report = {
            "as_of_date": args.as_of or date.today().isoformat(),
            "fund_code": "160125",
            "estimate_mode": "failed",
            "confidence": "none",
            "nav_estimate": None,
            "source_checks": {},
            "notes": [f"任务失败：{type(exc).__name__}: {exc}"],
            "generated_at": datetime.now().astimezone().isoformat(),
        }
        write_report(report, args.output_dir)
        if args.send_email:
            send_report_email(
                f"160125 盘后估值失败 {report['as_of_date']}", render_markdown(report)
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    _, markdown_path = write_report(report, args.output_dir)
    if args.send_email:
        send_report_email(
            f"160125 盘后估算净值 {report['as_of_date']} = {report['nav_estimate']:.4f}",
            markdown_path.read_text(encoding="utf-8"),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

