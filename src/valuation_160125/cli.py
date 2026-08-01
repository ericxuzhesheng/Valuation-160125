from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any

import pandas as pd

from .core import (
    CrossCheck,
    adaptive_method_weights,
    benchmark_return,
    calibrated_return,
    disclosed_holdings_return,
    ensemble_return,
    estimate_nav,
    numeric_cross_check,
)
from .emailer import send_report_email
from .providers import AKShareProvider, ProviderError, TushareProvider, tencent_hk_spot
from .report import render_markdown, write_report


def _date(value: str | None) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else date.today()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and isfinite(number) else None


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
    # HKD is effectively pegged to USD; USDCNH is CNY per USD, so
    # HKD/CNY is approximately USDCNH / 7.8.
    return previous / 7.8, current / 7.8


def _price_pair(frame: pd.DataFrame, as_of: date) -> tuple[float, float] | None:
    if frame.empty:
        return None
    date_column = next(
        (column for column in ("trade_date", "日期", "date") if column in frame.columns),
        None,
    )
    close_column = next(
        (column for column in ("close", "收盘", "收盘价") if column in frame.columns),
        None,
    )
    if date_column is None or close_column is None:
        return None
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce").dt.date
    result[close_column] = pd.to_numeric(result[close_column], errors="coerce")
    result = result.dropna(subset=[date_column, close_column])
    result = result[(result[date_column] <= as_of) & (result[close_column] > 0)]
    if len(result) < 2:
        return None
    result = result.sort_values(date_column)
    return float(result.iloc[-2][close_column]), float(result.iloc[-1][close_column])


def _flexible_price_pair(frame: pd.DataFrame, as_of: date) -> tuple[float, float] | None:
    """Accept Tushare English columns and AKShare's actual Chinese columns."""
    if frame.empty:
        return None
    date_column = next(
        (column for column in ("trade_date", "日期", "date") if column in frame.columns),
        None,
    )
    close_column = next(
        (
            column
            for column in ("close", "收盘", "收盘价", "最新价", "latest")
            if column in frame.columns
        ),
        None,
    )
    if date_column is None or close_column is None:
        return None
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce").dt.date
    result[close_column] = pd.to_numeric(result[close_column], errors="coerce")
    result = result.dropna(subset=[date_column, close_column])
    result = result[(result[date_column] <= as_of) & (result[close_column] > 0)]
    if len(result) < 2:
        return None
    result = result.sort_values(date_column)
    return float(result.iloc[-2][close_column]), float(result.iloc[-1][close_column])


def _spot_price_pairs(frame: pd.DataFrame | None) -> dict[str, tuple[float, float]]:
    if frame is None or frame.empty:
        return {}
    code_column = next(
        (column for column in ("代码", "symbol", "code") if column in frame.columns),
        None,
    )
    current_column = next(
        (column for column in ("最新价", "latest", "close") if column in frame.columns),
        None,
    )
    previous_column = next(
        (column for column in ("昨收", "pre_close", "previous") if column in frame.columns),
        None,
    )
    if code_column is None or current_column is None or previous_column is None:
        return {}
    result: dict[str, tuple[float, float]] = {}
    for _, row in frame.iterrows():
        symbol = str(row[code_column]).split(".")[0].strip().zfill(5)
        previous = _float(row[previous_column])
        current = _float(row[current_column])
        if symbol and previous is not None and current is not None and previous > 0 and current > 0:
            result[symbol] = (previous, current)
    return result


def _normalise_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "weight_pct"])
    symbol_column = next(
        (column for column in ("symbol", "股票代码") if column in frame.columns), None
    )
    weight_column = next(
        (
            column
            for column in ("weight_pct", "stk_mkv_ratio", "占净值比例")
            if column in frame.columns
        ),
        None,
    )
    if symbol_column is None or weight_column is None:
        return pd.DataFrame(columns=["symbol", "weight_pct"])
    result = pd.DataFrame(
        {
            "symbol": frame[symbol_column]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .str.zfill(5),
            "weight_pct": pd.to_numeric(
                frame[weight_column].astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            ),
        }
    )
    return result.dropna(subset=["symbol", "weight_pct"])


def _historical_proxy_returns(
    nav_frame: pd.DataFrame,
    hsi_frame: pd.DataFrame,
    as_of: date,
    fx_frame: pd.DataFrame | None = None,
) -> tuple[list[float], list[float]]:
    nav = _sorted_frame(nav_frame, "nav_date")
    nav = nav[nav["nav_date"] <= as_of].copy()
    nav["nav"] = pd.to_numeric(nav.get("unit_nav"), errors="coerce")
    nav = nav.dropna(subset=["nav"])
    hsi = _sorted_frame(hsi_frame, "trade_date")
    hsi["close"] = pd.to_numeric(hsi.get("close"), errors="coerce")
    hsi = hsi.dropna(subset=["close"])
    hsi_returns: dict[date, float] = {}
    for index in range(1, len(hsi)):
        previous = float(hsi.iloc[index - 1]["close"])
        current = float(hsi.iloc[index]["close"])
        if previous > 0 and current > 0:
            hsi_returns[hsi.iloc[index]["trade_date"]] = current / previous - 1.0
    if fx_frame is not None and not fx_frame.empty:
        fx = _sorted_frame(fx_frame, "trade_date")
        close_column = "bid_close" if "bid_close" in fx.columns else "close"
        fx["fx"] = pd.to_numeric(fx.get(close_column), errors="coerce")
        fx = fx.dropna(subset=["fx"])
        fx_by_date = {
            row["trade_date"]: float(row["fx"]) / 7.8
            for _, row in fx.iterrows()
            if float(row["fx"]) > 0
        }
        for index in range(1, len(hsi)):
            current_date = hsi.iloc[index]["trade_date"]
            previous_date = hsi.iloc[index - 1]["trade_date"]
            if current_date in fx_by_date and previous_date in fx_by_date:
                hsi_returns[current_date] = (
                    (float(hsi.iloc[index]["close"]) / float(hsi.iloc[index - 1]["close"]))
                    * (fx_by_date[current_date] / fx_by_date[previous_date])
                    - 1.0
                ) * 0.95
    fund_returns: list[float] = []
    proxy_returns: list[float] = []
    for index in range(1, len(nav)):
        previous = float(nav.iloc[index - 1]["nav"])
        current = float(nav.iloc[index]["nav"])
        current_date = nav.iloc[index]["nav_date"]
        if previous > 0 and current > 0 and current_date in hsi_returns:
            fund_returns.append(current / previous - 1.0)
            proxy_returns.append(hsi_returns[current_date])
    return fund_returns, proxy_returns


def _holdings_price_returns(
    holdings: pd.DataFrame,
    tushare: TushareProvider | None,
    akshare: AKShareProvider | None,
    akshare_spot: pd.DataFrame | None,
    tencent_spot: pd.DataFrame | None,
    as_of: date,
    price_start: date,
    fx_levels: tuple[float, float],
) -> tuple[list[dict[str, float | str]], dict[str, Any]]:
    marked: list[dict[str, float | str]] = []
    stats: dict[str, Any] = {
        "total": len(holdings),
        "priced": 0,
        "tushare": 0,
        "akshare_spot": 0,
        "tencent": 0,
        "akshare": 0,
        "failures": [],
    }
    spot_pairs = _spot_price_pairs(akshare_spot)
    tencent_pairs = _spot_price_pairs(tencent_spot)
    for _, row in holdings.iterrows():
        symbol = str(row["symbol"]).zfill(5)
        price_pair = None
        price_source = None
        errors: list[str] = []
        if tushare is not None:
            try:
                price_pair = _flexible_price_pair(
                    tushare.hk_daily(symbol, price_start, as_of), as_of
                )
                if price_pair is not None:
                    price_source = "Tushare"
            except ProviderError:
                errors.append("Tushare hk_daily failed")
        if price_pair is None and symbol in spot_pairs:
            price_pair = spot_pairs[symbol]
            price_source = "AKShareSpot"
        if price_pair is None and symbol in tencent_pairs:
            price_pair = tencent_pairs[symbol]
            price_source = "Tencent"
        if price_pair is None and akshare is not None:
            try:
                price_pair = _flexible_price_pair(
                    akshare.hk_daily(symbol, price_start, as_of), as_of
                )
                if price_pair is not None:
                    price_source = "AKShare"
            except ProviderError:
                errors.append("AKShare stock_hk_hist failed")
        if price_pair is None:
            stats["failures"].append({"symbol": symbol, "errors": errors or ["no usable quote"]})
            continue
        previous_price, current_price = price_pair
        local_return = current_price / previous_price - 1.0
        rmb_return = (
            (current_price / previous_price) * (fx_levels[1] / fx_levels[0]) - 1.0
        )
        marked.append(
            {
                "symbol": symbol,
                "weight": float(row["weight_pct"]) / 100.0,
                "return": rmb_return,
                "local_return": local_return,
                "price_source": price_source or "unknown",
            }
        )
        stats["priced"] += 1
        source_key = {
            "Tushare": "tushare",
            "AKShareSpot": "akshare_spot",
            "Tencent": "tencent",
            "AKShare": "akshare",
        }.get(price_source or "")
        if source_key is not None:
            stats[source_key] += 1
    return marked, stats


def build_live_report(as_of: date) -> dict[str, Any]:
    lookback = as_of - timedelta(days=400)
    price_start = as_of - timedelta(days=10)
    notes: list[str] = []
    tushare = None
    akshare = None
    try:
        tushare = TushareProvider(token=os.environ.get("TUSHARE_TOKEN"))
    except ProviderError as exc:
        notes.append(f"Tushare 初始化失败，尝试 AKShare 备用：{exc}")
    try:
        akshare = AKShareProvider()
    except ProviderError as exc:
        notes.append(f"AKShare 初始化失败：{exc}")

    published_date: date | None = None
    published_nav: float | None = None
    tushare_nav: float | None = None
    akshare_nav: float | None = None
    nav_frame = pd.DataFrame()
    if tushare is not None:
        try:
            nav_frame = tushare.fund_nav(lookback, as_of)
            published_date, published_nav = _latest_published_nav(
                nav_frame, as_of - timedelta(days=1)
            )
            tushare_nav = published_nav
        except ProviderError as exc:
            notes.append(str(exc))
    if akshare is not None:
        try:
            ak_snapshot = akshare.fund_daily_snapshot()
            akshare_nav = _float(ak_snapshot.get("previous_nav"))
            if published_nav is None and akshare_nav is not None:
                published_nav = akshare_nav
                published_date = date.fromisoformat(str(ak_snapshot["previous_date"]))
        except ProviderError as exc:
            notes.append(str(exc))
    if published_nav is None or published_date is None:
        raise ProviderError("Tushare 与 AKShare 均未取得最近公布净值")

    tushare_pair = None
    akshare_pair = None
    hsi_frame = pd.DataFrame()
    if tushare is not None:
        try:
            hsi_frame = tushare.hsi(lookback, as_of)
            tushare_pair = _hsi_pair(hsi_frame, as_of, "trade_date")
        except ProviderError as exc:
            notes.append(str(exc))
    if akshare is not None:
        try:
            akshare_pair = _hsi_pair(akshare.hsi(), as_of, "date")
        except ProviderError as exc:
            notes.append(str(exc))
    if tushare_pair is None and akshare_pair is None:
        raise ProviderError("Tushare 与 AKShare 均未取得恒生指数当前/前一交易日数据")
    selected_pair = tushare_pair or akshare_pair
    assert selected_pair is not None
    previous_date, previous_hsi, current_hsi, _ = selected_pair

    checks: dict[str, dict[str, Any]] = {}
    if tushare_nav is not None and akshare_nav is not None:
        checks["fund_nav"] = _check_to_dict(
            numeric_cross_check(tushare_nav, akshare_nav, absolute_tolerance=0.0001)
        )
    elif tushare_nav is not None:
        checks["fund_nav"] = {"status": "tushare_only", "left": tushare_nav, "right": None}
    else:
        checks["fund_nav"] = {"status": "akshare_fallback", "left": None, "right": akshare_nav}

    if tushare_pair is not None and akshare_pair is not None:
        checks["hsi_close"] = _check_to_dict(
            numeric_cross_check(tushare_pair[2], akshare_pair[2], absolute_tolerance=0.02)
        )
    elif tushare_pair is not None:
        checks["hsi_close"] = {"status": "tushare_only", "left": current_hsi, "right": None}
    else:
        checks["hsi_close"] = {"status": "akshare_fallback", "left": None, "right": current_hsi}

    fx_levels = None
    fx_frame = pd.DataFrame()
    if tushare is not None:
        try:
            fx_frame = tushare.usdcnh(lookback, as_of)
            fx_levels = _usdcnh_levels(fx_frame, as_of)
        except ProviderError as exc:
            notes.append(str(exc))
    if fx_levels is None:
        fx_levels = (1.0, 1.0)
        checks["fx_hkd_cny"] = {"status": "missing", "left": None, "right": None}
        notes.append("汇率数据不可用，估算暂不计入 HKD/CNY 当日变化")
    else:
        checks["fx_hkd_cny"] = {
            "status": "tushare_only",
            "left": fx_levels[0],
            "right": fx_levels[1],
        }

    daily_return = benchmark_return(
        previous_hsi=previous_hsi,
        current_hsi=current_hsi,
        previous_hkd_cny=fx_levels[0],
        current_hkd_cny=fx_levels[1],
        equity_weight=0.95,
    )
    # Public holdings are quarterly and incomplete. Mark the disclosed names
    # directly, then assign the undisclosed residual to the benchmark.
    tushare_holdings = pd.DataFrame()
    akshare_holdings = pd.DataFrame()
    akshare_spot = pd.DataFrame()
    holdings_report_date: str | None = None
    if tushare is not None:
        try:
            tushare_holdings = _normalise_holdings(tushare.fund_holdings())
        except ProviderError as exc:
            notes.append(str(exc))
    if akshare is not None:
        try:
            ak_payload = akshare.fund_holdings()
            akshare_holdings = _normalise_holdings(ak_payload["data"])
            holdings_report_date = ak_payload.get("as_of_date")
        except ProviderError as exc:
            notes.append(str(exc))
        try:
            akshare_spot = akshare.hk_spot()
        except ProviderError as exc:
            notes.append(str(exc))

    checks["holdings_count"] = {
        "status": (
            "match"
            if not tushare_holdings.empty
            and not akshare_holdings.empty
            and len(tushare_holdings) == len(akshare_holdings)
            else "tushare_only"
            if not tushare_holdings.empty
            else "akshare_only"
            if not akshare_holdings.empty
            else "missing"
        ),
        "left": float(len(tushare_holdings)) if not tushare_holdings.empty else None,
        "right": float(len(akshare_holdings)) if not akshare_holdings.empty else None,
        "delta": (
            float(len(akshare_holdings) - len(tushare_holdings))
            if not tushare_holdings.empty and not akshare_holdings.empty
            else None
        ),
    }
    selected_holdings = (
        tushare_holdings if not tushare_holdings.empty else akshare_holdings
    )
    holdings_source = "Tushare" if not tushare_holdings.empty else "AKShare"
    tencent_spot = pd.DataFrame()
    if not selected_holdings.empty:
        try:
            tencent_spot = tencent_hk_spot(
                selected_holdings["symbol"].astype(str).tolist()
            )
        except ProviderError as exc:
            notes.append(str(exc))

    method_returns: dict[str, float] = {"benchmark": daily_return, "carry": 0.0}
    base_weights: dict[str, float] = {"benchmark": 0.65, "carry": 0.15}
    marked_holdings: list[dict[str, float | str]] = []
    holding_stats: dict[str, Any] = {
        "total": 0,
        "priced": 0,
        "tushare": 0,
        "akshare_spot": 0,
        "tencent": 0,
        "akshare": 0,
        "failures": [],
    }
    covered_weight = 0.0
    if not selected_holdings.empty:
        marked_holdings, holding_stats = _holdings_price_returns(
            selected_holdings,
            tushare,
            akshare,
            akshare_spot,
            tencent_spot,
            as_of,
            price_start,
            fx_levels,
        )
        if marked_holdings:
            holdings_return, covered_weight = disclosed_holdings_return(
                marked_holdings,
                residual_return=daily_return,
            )
            method_returns["disclosed_holdings"] = holdings_return
            base_weights["disclosed_holdings"] = 0.02
            notes.append(
                f"披露持仓模型使用 {holding_stats['priced']}/{holding_stats['total']} 只股票，"
                f"直接覆盖净值权重约 {covered_weight:.2%}；未披露部分使用恒生指数代理。"
            )
        else:
            notes.append("已找到披露持仓，但当前没有可用的港股价格，未纳入直接持仓模型。")

    if holding_stats.get("failures"):
        notes.append(
            "未取得行情的持仓："
            + ", ".join(item["symbol"] for item in holding_stats["failures"])
        )

    if marked_holdings:
        sources = sorted({str(item.get("price_source", "unknown")) for item in marked_holdings})
        notes.append("持仓价格来源：" + ", ".join(sources))

    calibration = None
    fund_returns: list[float] = []
    proxy_returns: list[float] = []
    if not nav_frame.empty and not hsi_frame.empty:
        fund_returns, proxy_returns = _historical_proxy_returns(
            nav_frame,
            hsi_frame,
            as_of,
            fx_frame,
        )
        calibration = calibrated_return(daily_return, fund_returns, proxy_returns)
        if calibration is not None:
            method_returns["historical_calibration"] = calibration[0]
            base_weights["historical_calibration"] = 0.18
            notes.append(
                f"历史校准使用 {calibration[1]['observations']} 个共同观测，"
                f"beta={calibration[1]['beta']:.3f}。"
            )

    historical_errors = {
        "benchmark": [fund_proxy - fund for fund, fund_proxy in zip(fund_returns, proxy_returns)],
        "carry": [-fund for fund in fund_returns],
    }
    method_weights = adaptive_method_weights(
        method_returns,
        historical_errors,
        base_weights,
        max_weights={
            "benchmark": 0.80,
            "carry": 0.30,
            "disclosed_holdings": 0.08,
            "historical_calibration": 0.35,
        },
    )
    final_return = ensemble_return(method_returns, method_weights)
    estimate = estimate_nav(published_nav, final_return)

    if marked_holdings and holding_stats["priced"] == holding_stats["total"]:
        used_fallback = any(item.get("price_source") == "Tencent" for item in marked_holdings)
        confidence = (
            "high"
            if not used_fallback and checks["fx_hkd_cny"]["status"] != "missing"
            else "medium"
        )
    elif marked_holdings:
        confidence = "medium"
    else:
        confidence = "low"
    if not marked_holdings:
        notes.append("未能使用披露持仓，估值退回恒生指数基准模型。")

    return {
        "as_of_date": as_of.isoformat(),
        "fund_code": "160125",
        "published_nav_date": published_date.isoformat(),
        "published_nav": published_nav,
        "nav_estimate": estimate,
        "estimate_mode": "disclosed_holdings_ensemble" if marked_holdings else "benchmark_fallback",
        "confidence": confidence,
        "previous_hsi": previous_hsi,
        "current_hsi": current_hsi,
        "previous_hsi_date": previous_date.isoformat(),
        "daily_return": final_return,
        "benchmark_return": daily_return,
        "method_returns": method_returns,
        "method_weights": method_weights,
        "holdings_source": holdings_source if not selected_holdings.empty else None,
        "holdings_report_date": holdings_report_date,
        "holdings_count": holding_stats["total"],
        "priced_holdings_count": holding_stats["priced"],
        "holding_price_failures": holding_stats.get("failures", []),
        "covered_weight": covered_weight,
        "holding_marks": marked_holdings,
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
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-start", default="2015-01-01")
    parser.add_argument("--backtest-end")
    args = parser.parse_args(argv)

    if args.backtest:
        from .backtest import run_backtest, write_backtest

        try:
            result = run_backtest(
                date.fromisoformat(args.backtest_start),
                date.fromisoformat(args.backtest_end) if args.backtest_end else date.today(),
                token=os.environ.get("TUSHARE_TOKEN"),
            )
            paths = write_backtest(result, args.output_dir)
            output = {
                "backtest_json": str(paths[0]),
                "backtest_csv": str(paths[1]),
                "observations": result["observations"],
                "metrics": result["metrics"],
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(
                json.dumps(
                    {"backtest": "failed", "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1

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
