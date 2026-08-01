from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 160125 盘后净值估算 — {report.get('as_of_date', 'unknown')}",
        "",
        f"- 估算净值：**{_display(report.get('nav_estimate'))}**",
        f"- 估算模型：`{report.get('estimate_mode', 'unknown')}`",
        f"- 置信等级：`{report.get('confidence', 'unknown')}`",
        f"- 最近公布净值：{_display(report.get('published_nav'))} "
        f"（日期 {report.get('published_nav_date', '—')}）",
        "",
        "## 双源校验",
        "",
    ]
    for name, check in (report.get("source_checks") or {}).items():
        lines.append(
            f"- `{name}`: **{check.get('status', 'unknown')}**; "
            f"Tushare={_display(check.get('left'))}; "
            f"AKShare={_display(check.get('right'))}; "
            f"差异={_display(check.get('delta'))}"
        )
    lines.extend(["", "## 备注", ""])
    notes = report.get("notes") or ["无"]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    as_of = report.get("as_of_date", "unknown")
    json_path = target / f"nav_estimate_{as_of}.json"
    markdown_path = target / f"nav_estimate_{as_of}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path

