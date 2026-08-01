from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd


class ProviderError(RuntimeError):
    """A data provider failed or returned unusable data."""


def _date_text(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def _yyyymmdd(value: date | str) -> str:
    return _date_text(value).replace("-", "")


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


class TushareProvider:
    def __init__(self, token: str | None = None) -> None:
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - environment failure
            raise ProviderError("tushare is not installed") from exc
        if token:
            ts.set_token(token)
        self.pro = ts.pro_api()

    def fund_nav(self, start_date: date | str, end_date: date | str) -> pd.DataFrame:
        try:
            return self.pro.fund_nav(
                ts_code="160125.SZ",
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
            )
        except Exception as exc:
            raise ProviderError(f"Tushare fund_nav failed: {type(exc).__name__}") from exc

    def hsi(self, start_date: date | str, end_date: date | str) -> pd.DataFrame:
        try:
            return self.pro.index_global(
                ts_code="HSI",
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
            )
        except Exception as exc:
            raise ProviderError(
                f"Tushare index_global(HSI) failed: {type(exc).__name__}"
            ) from exc

    def hk_daily(
        self, symbol: str, start_date: date | str, end_date: date | str
    ) -> pd.DataFrame:
        try:
            return self.pro.hk_daily(
                ts_code=f"{symbol.zfill(5)}.HK",
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
            )
        except Exception as exc:
            raise ProviderError(f"Tushare hk_daily failed: {type(exc).__name__}") from exc

    def usdcnh(self, start_date: date | str, end_date: date | str) -> pd.DataFrame:
        try:
            return self.pro.fx_daily(
                ts_code="USDCNH.FXCM",
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
            )
        except Exception as exc:
            raise ProviderError(f"Tushare fx_daily failed: {type(exc).__name__}") from exc


class AKShareProvider:
    def __init__(self) -> None:
        try:
            import akshare as ak
        except ImportError as exc:  # pragma: no cover - environment failure
            raise ProviderError("akshare is not installed") from exc
        self.ak = ak

    def hsi(self) -> pd.DataFrame:
        try:
            return self.ak.stock_hk_index_daily_sina(symbol="HSI")
        except Exception as exc:
            raise ProviderError(
                f"AKShare stock_hk_index_daily_sina failed: {type(exc).__name__}"
            ) from exc

    def hk_daily(
        self, symbol: str, start_date: date | str, end_date: date | str
    ) -> pd.DataFrame:
        try:
            return self.ak.stock_hk_hist(
                symbol=symbol.zfill(5),
                period="daily",
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
                adjust="",
            )
        except Exception as exc:
            raise ProviderError(f"AKShare stock_hk_hist failed: {type(exc).__name__}") from exc

    def fund_daily_snapshot(self) -> dict[str, float | str | None]:
        try:
            frame = self.ak.fund_open_fund_daily_em()
        except Exception as exc:
            raise ProviderError(
                f"AKShare fund_open_fund_daily_em failed: {type(exc).__name__}"
            ) from exc
        if frame.empty:
            raise ProviderError("AKShare returned an empty fund daily table")

        code_column = next((c for c in frame.columns if "代码" in str(c)), frame.columns[0])
        rows = frame[frame[code_column].astype(str).str.zfill(6) == "160125"]
        if rows.empty:
            raise ProviderError("AKShare fund daily table has no 160125 row")
        row = rows.iloc[0]
        nav_columns = [c for c in frame.columns if "单位净值" in str(c)]
        nav_columns.sort(reverse=True)
        current = _as_float(row[nav_columns[0]]) if nav_columns else None
        previous = _as_float(row[nav_columns[1]]) if len(nav_columns) > 1 else None
        current_date = str(nav_columns[0]).split("-")[0] if nav_columns else None
        previous_date = str(nav_columns[1]).split("-")[0] if len(nav_columns) > 1 else None
        return {
            "current_nav": current,
            "current_date": current_date,
            "previous_nav": previous,
            "previous_date": previous_date,
        }


def previous_calendar_date(as_of: date) -> date:
    return as_of - timedelta(days=1)

