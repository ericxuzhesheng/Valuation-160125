from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, timedelta
from io import StringIO
from math import isfinite
import re
from typing import Any

import pandas as pd
import requests


class ProviderError(RuntimeError):
    """A data provider failed or returned unusable data."""


def _bounded_call(label: str, function: Any, *, timeout: int = 45, attempts: int = 2) -> Any:
    """Run an AKShare call with a bounded wait and one retry."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(function)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover - provider-specific errors
            last_error = exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if attempt + 1 < attempts:
            continue
    raise ProviderError(f"{label} timed out or failed after {attempts} attempts") from last_error


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
    return number if pd.notna(number) and isfinite(number) else None


def tencent_hk_spot(symbols: list[str], timeout: int = 20) -> pd.DataFrame:
    """Fetch latest/previous HK prices in one request as a final fallback."""
    normalized = sorted({str(symbol).split(".")[0].zfill(5) for symbol in symbols})
    if not normalized:
        return pd.DataFrame(columns=["代码", "最新价", "昨收"])
    query = ",".join(f"r_hk{symbol}" for symbol in normalized)
    try:
        response = requests.get(
            "https://qt.gtimg.cn/q=" + query,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        rows: list[dict[str, str]] = []
        for item in response.text.split(";"):
            match = re.search(r'v_r_hk(\d+)="(.*)"', item)
            if not match:
                continue
            fields = match.group(2).split("~")
            if len(fields) < 5:
                continue
            rows.append(
                {
                    "代码": match.group(1).zfill(5),
                    "最新价": fields[3],
                    "昨收": fields[4],
                }
            )
        if not rows:
            raise ProviderError("Tencent HK quote returned no usable rows")
        return pd.DataFrame(rows)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Tencent HK quote failed: {type(exc).__name__}") from exc


class TushareProvider:
    def __init__(self, token: str | None = None) -> None:
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - environment failure
            raise ProviderError("tushare is not installed") from exc
        if not token:
            try:
                token = ts.get_token()
            except Exception:
                token = None
        if not token:
            raise ProviderError("TUSHARE_TOKEN is missing")
        # Do not call set_token(), which persists the token in the user profile.
        self.pro = ts.pro_api(token=token, timeout=30)

    def fund_nav(self, start_date: date | str, end_date: date | str) -> pd.DataFrame:
        try:
            return self.pro.fund_nav(
                ts_code="160125.SZ",
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
            )
        except Exception as exc:
            raise ProviderError(f"Tushare fund_nav failed: {type(exc).__name__}") from exc

    def fund_holdings(self) -> pd.DataFrame:
        try:
            return self.pro.fund_portfolio(ts_code="160125.SZ")
        except Exception as exc:
            raise ProviderError(
                f"Tushare fund_portfolio failed: {type(exc).__name__}"
            ) from exc

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
    def __init__(self, timeout: int = 45) -> None:
        try:
            import akshare as ak
        except ImportError as exc:  # pragma: no cover - environment failure
            raise ProviderError("akshare is not installed") from exc
        self.ak = ak
        self.timeout = timeout

    def hsi(self) -> pd.DataFrame:
        try:
            return _bounded_call(
                "AKShare stock_hk_index_daily_sina",
                lambda: self.ak.stock_hk_index_daily_sina(symbol="HSI"),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ProviderError(
                f"AKShare stock_hk_index_daily_sina failed: {type(exc).__name__}"
            ) from exc

    def hk_daily(
        self, symbol: str, start_date: date | str, end_date: date | str
    ) -> pd.DataFrame:
        try:
            return _bounded_call(
                "AKShare stock_hk_hist",
                lambda: self.ak.stock_hk_hist(
                    symbol=symbol.zfill(5),
                    period="daily",
                    start_date=_yyyymmdd(start_date),
                    end_date=_yyyymmdd(end_date),
                    adjust="",
                ),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ProviderError(f"AKShare stock_hk_hist failed: {type(exc).__name__}") from exc

    def hk_spot(self) -> pd.DataFrame:
        """Fetch the latest trading-day quote for all Hong Kong stocks once."""
        try:
            return _bounded_call(
                "AKShare stock_hk_spot_em",
                self.ak.stock_hk_spot_em,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ProviderError(
                f"AKShare stock_hk_spot_em failed: {type(exc).__name__}"
            ) from exc

    def fund_daily_snapshot(self) -> dict[str, float | str | None]:
        try:
            frame = _bounded_call(
                "AKShare fund_open_fund_daily_em",
                self.ak.fund_open_fund_daily_em,
                timeout=self.timeout,
            )
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

    def fund_holdings(self, year: str = "") -> dict[str, Any]:
        """Fetch the latest public portfolio table with an Eastmoney header.

        AKShare's parser uses the same endpoint, but Eastmoney can reject a
        request without a browser-like Referer/User-Agent.  Keeping the
        request here makes the daily job more reproducible while preserving
        AKShare as the library/data-source fallback.
        """
        url = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
        params = {
            "type": "jjcc",
            "code": "160125",
            "topline": "10000",
            "year": year,
            "month": "",
            "rt": "0.913877030254846",
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://fundf10.eastmoney.com/ccmx_160125.html",
        }
        try:
            response = _bounded_call(
                "AKShare fund_portfolio_hold_em",
                lambda: requests.get(url, params=params, headers=headers, timeout=self.timeout),
                timeout=self.timeout,
            )
            response.raise_for_status()
            match = re.search(r'content:"(.*)",arryear:', response.text, flags=re.S)
            if not match:
                raise ProviderError("Eastmoney portfolio response has no content")
            content = match.group(1)
            soup = self._soup(content)
            labels = [
                item.get_text(" ", strip=True)
                for item in soup.find_all(name="h4", attrs={"class": "t"})
            ]
            tables = pd.read_html(StringIO(content), converters={"股票代码": str})
            if not labels or not tables:
                raise ProviderError("Eastmoney returned no public holding table")
            reports: list[dict[str, Any]] = []
            for label, raw_table in zip(labels, tables):
                table = raw_table.copy()
                weight_column = next(
                    (column for column in table.columns if "占净值" in str(column)), None
                )
                if weight_column is None or "股票代码" not in table.columns:
                    continue
                table["weight_pct"] = pd.to_numeric(
                    table[weight_column].astype(str).str.replace("%", "", regex=False),
                    errors="coerce",
                )
                table["symbol"] = (
                    table["股票代码"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
                )
                table = table.dropna(subset=["symbol", "weight_pct"])
                report_date_match = re.search(r"截止至：\s*(\d{4}-\d{2}-\d{2})", label)
                report_date = report_date_match.group(1) if report_date_match else None
                if report_date and not table.empty:
                    reports.append(
                        {
                            "as_of_date": report_date,
                            "data": table[["symbol", "weight_pct"]].reset_index(drop=True),
                            "raw_label": label,
                        }
                    )
            if not reports:
                raise ProviderError("public holding table has no usable rows")
            return {
                "as_of_date": reports[0]["as_of_date"],
                "data": reports[0]["data"],
                "reports": reports,
                "source": "AKShare/Eastmoney",
                "raw_label": reports[0]["raw_label"],
            }
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"AKShare fund_portfolio_hold_em failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _soup(content: str) -> Any:
        from bs4 import BeautifulSoup

        return BeautifulSoup(content, features="lxml")


def previous_calendar_date(as_of: date) -> date:
    return as_of - timedelta(days=1)
