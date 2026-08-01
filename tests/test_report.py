from valuation_160125.report import render_markdown


def test_report_contains_estimate_and_data_quality():
    markdown = render_markdown(
        {
            "as_of_date": "2026-07-31",
            "fund_code": "160125",
            "published_nav_date": "2026-07-30",
            "published_nav": 1.5568,
            "nav_estimate": 1.5583,
            "estimate_mode": "benchmark_fallback",
            "confidence": "low",
            "source_checks": {"fund_nav": {"status": "match"}},
            "notes": ["持仓数据暂不可用，使用基准兜底"],
        }
    )

    assert "1.5583" in markdown
    assert "benchmark_fallback" in markdown
    assert "持仓数据暂不可用" in markdown

