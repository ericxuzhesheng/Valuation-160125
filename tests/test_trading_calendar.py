from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from valuation_160125 import cli
from valuation_160125.providers import ProviderError, TushareProvider


@pytest.mark.parametrize("a_open,hk_open", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_both_exchange_calendars_are_queried_for_the_valuation_date(a_open, hk_open):
    provider = TushareProvider.__new__(TushareProvider)
    provider.pro = SimpleNamespace(
        trade_cal=Mock(return_value=pd.DataFrame({"cal_date": ["20261002"], "is_open": [str(a_open)]})),
        hk_tradecal=Mock(return_value=pd.DataFrame({"cal_date": ["20261002"], "is_open": [hk_open]})),
    )

    assert provider.trading_day_status(date(2026, 10, 2)) == {
        "SZSE": bool(a_open), "HKEX": bool(hk_open)
    }
    provider.pro.trade_cal.assert_called_once_with(
        exchange="SZSE", start_date="20261002", end_date="20261002"
    )
    provider.pro.hk_tradecal.assert_called_once_with(start_date="20261002", end_date="20261002")


@pytest.mark.parametrize("invalid", [
    pd.DataFrame(),
    pd.DataFrame({"cal_date": ["20261002"]}),
    pd.DataFrame({"cal_date": ["20261001"], "is_open": [1]}),
    pd.DataFrame({"cal_date": ["20261002"], "is_open": [None]}),
    pd.DataFrame({"cal_date": ["20261002"], "is_open": [2]}),
    pd.DataFrame({"cal_date": ["20261002", "20261002"], "is_open": [0, 1]}),
])
@pytest.mark.parametrize("endpoint", ["trade_cal", "hk_tradecal"])
def test_missing_or_invalid_calendar_is_not_treated_as_a_trading_day(invalid, endpoint):
    valid = pd.DataFrame({"cal_date": ["20261002"], "is_open": [1]})
    provider = TushareProvider.__new__(TushareProvider)
    provider.pro = SimpleNamespace(trade_cal=Mock(return_value=valid), hk_tradecal=Mock(return_value=valid))
    getattr(provider.pro, endpoint).return_value = invalid
    with pytest.raises(ProviderError):
        provider.trading_day_status(date(2026, 10, 2))


@pytest.mark.parametrize("endpoint", ["trade_cal", "hk_tradecal"])
def test_calendar_api_failure_is_reported(endpoint):
    valid = pd.DataFrame({"cal_date": ["20261002"], "is_open": [1]})
    provider = TushareProvider.__new__(TushareProvider)
    provider.pro = SimpleNamespace(trade_cal=Mock(return_value=valid), hk_tradecal=Mock(return_value=valid))
    getattr(provider.pro, endpoint).side_effect = TimeoutError("calendar unavailable")
    with pytest.raises(ProviderError, match="trading calendar"):
        provider.trading_day_status(date(2026, 10, 2))


@pytest.mark.parametrize("as_of,a_open,hk_open", [
    ("2026-10-01", False, False),  # National Day, both closed.
    ("2026-10-02", False, True),   # Mainland holiday, Hong Kong open.
    ("2026-07-01", True, False),   # Hong Kong holiday, mainland open.
    ("2026-10-10", False, False),  # Government make-up workday, exchanges closed.
    ("2026-10-09", True, True),    # Friday's report is sent on Saturday morning.
])
def test_cli_only_estimates_and_emails_on_joint_trading_days(monkeypatch, tmp_path, as_of, a_open, hk_open):
    status = {"SZSE": a_open, "HKEX": hk_open}
    calendar = Mock(return_value=status)
    monkeypatch.setattr(cli, "TushareProvider", lambda **kwargs: SimpleNamespace(trading_day_status=calendar))
    report = cli.build_demo_report()
    report["as_of_date"] = as_of
    build = Mock(return_value=report)
    send = Mock()
    monkeypatch.setattr(cli, "build_live_report", build)
    monkeypatch.setattr(cli, "send_report_email", send)

    assert cli.main(["--date", as_of, "--check-trading-calendar", "--send-email", "--output-dir", str(tmp_path)]) == 0
    calendar.assert_called_once_with(date.fromisoformat(as_of))
    saved = json.loads((tmp_path / f"nav_estimate_{as_of}.json").read_text(encoding="utf-8"))
    assert saved["trading_calendar"] == status
    if a_open and hk_open:
        build.assert_called_once_with(date.fromisoformat(as_of))
        send.assert_called_once()
        assert as_of in send.call_args.args[0]
    else:
        build.assert_not_called()
        send.assert_not_called()
        assert saved["estimate_mode"] == "skipped_non_trading_day"
        assert saved["nav_estimate"] is None


def test_cli_calendar_failure_stops_estimation_and_reports_failure(monkeypatch, tmp_path):
    calendar = Mock(side_effect=ProviderError("trading calendar unavailable"))
    monkeypatch.setattr(cli, "TushareProvider", lambda **kwargs: SimpleNamespace(trading_day_status=calendar))
    build = Mock()
    send = Mock()
    monkeypatch.setattr(cli, "build_live_report", build)
    monkeypatch.setattr(cli, "send_report_email", send)
    assert cli.main(["--date", "2026-10-02", "--check-trading-calendar", "--send-email", "--output-dir", str(tmp_path)]) == 1
    build.assert_not_called()
    send.assert_called_once()
    assert "失败" in send.call_args.args[0]
    saved = json.loads((tmp_path / "nav_estimate_2026-10-02.json").read_text(encoding="utf-8"))
    assert saved["estimate_mode"] == "failed"


def test_offline_demo_does_not_query_trading_calendars(monkeypatch, tmp_path):
    provider = Mock(side_effect=AssertionError("offline mode must not access calendars"))
    monkeypatch.setattr(cli, "TushareProvider", provider)
    assert cli.main(["--offline-demo", "--check-trading-calendar", "--output-dir", str(tmp_path)]) == 0
    provider.assert_not_called()
