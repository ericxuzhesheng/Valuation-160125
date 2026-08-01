from datetime import date

import pandas as pd
import pytest

from valuation_160125.cli import _normalise_holdings, _usdcnh_levels


def test_usdcnh_is_converted_to_hkd_cny_in_the_correct_direction():
    frame = pd.DataFrame(
        {
            "trade_date": ["2026-07-30", "2026-07-31"],
            "bid_close": [7.20, 7.18],
        }
    )
    previous, current = _usdcnh_levels(frame, date(2026, 7, 31))
    assert previous == pytest.approx(7.20 / 7.8)
    assert current == pytest.approx(7.18 / 7.8)


def test_normalise_holdings_supports_tushare_and_akshare_columns():
    frame = pd.DataFrame(
        {
            "股票代码": ["700", "00939"],
            "占净值比例": ["4.12%", "3.00%"],
        }
    )
    result = _normalise_holdings(frame)
    assert result["symbol"].tolist() == ["00700", "00939"]
    assert result["weight_pct"].tolist() == pytest.approx([4.12, 3.0])
