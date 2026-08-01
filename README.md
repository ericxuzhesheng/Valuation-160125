# Valuation-160125

每天盘后估算南方香港优选股票（160125）的当日单位净值，并用 Tushare 与 AKShare 做交叉校验。

## 当前模型

当日官方净值通常滞后一个交易日，因此程序使用最近一次公布净值作为基准：

```text
估算净值_t = 已公布净值_(t-1) × (1 + 基准日收益率_t)
```

当前估值模型优先抓取最近披露季度的前十大持仓，按每只港股当日人民币计价涨跌计算；未披露的剩余权重使用业绩比较基准代理。最终结果再与“恒生指数 95% + 现金 5%”基准模型、以及最近约 400 天官方净值相对基准的滚动 beta 校准模型组合。持仓披露是季度频率，不能代表基金当日真实调仓。

程序会输出：

- 当日估算净值、估算模型和置信等级；
- 最近公布净值及日期；
- Tushare/AKShare 的基金净值和恒生指数校验状态；
- 数据失败、汇率缺失和持仓未纳入等明确备注。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) "src")

# 需要本机 Tushare 凭证，建议只放在环境变量或 Tushare 本地凭证存储中
$env:TUSHARE_TOKEN = "<local-only-token>"
python -m valuation_160125.cli --output-dir artifacts

# 不联网的演示与格式检查
python -m valuation_160125.cli --offline-demo --output-dir artifacts
python -m pytest

# 历史回测：只在季度报告日后 22 天启用该披露持仓快照，输出 JSON 和 CSV
python -m valuation_160125.cli --backtest --backtest-start 2025-07-01 --backtest-end 2026-07-31 --output-dir artifacts
```

## GitHub Actions 远程定时邮件

`.github/workflows/daily-estimate.yml` 在工作日 UTC 10:01 运行，即北京时间约 18:01，收件人为 `1874103486@qq.com`。GitHub Actions 的定时任务可能有延迟，邮件应理解为盘后参考，不是交易执行信号。

在仓库的 **Settings → Secrets and variables → Actions** 中添加以下 Repository secrets：

| Secret | 用途 |
|---|---|
| `TUSHARE_TOKEN` | Tushare API 凭证 |
| `SMTP_HOST` | SMTP 服务器地址 |
| `SMTP_PORT` | 通常为 587；465 使用 SSL |
| `SMTP_USERNAME` | SMTP 登录名 |
| `SMTP_PASSWORD` | SMTP 密码或应用专用密码 |
| `MAIL_FROM` | 发件地址 |
| `MAIL_TO` | 收件地址，多个地址用英文逗号分隔 |

配置后可在 **Actions → Daily 160125 post-market NAV estimate → Run workflow** 手动运行一次。成功或失败都会保存报告 artifact；失败时程序会尝试发送失败邮件并以非零状态退出，便于 GitHub Actions 告警。

## 数据源策略

Tushare 用于基金净值、恒生指数和外汇主查询；AKShare 用于基金净值列表、港股/指数行情交叉验证。交叉验证不做无依据的简单平均：两源一致才标记 `match`，一源中断或不一致会降低置信等级并写入报告。

个股持仓不是每日公开数据，且 160125 的最新持仓可能暂时无法从 API 返回。因此没有持仓快照时，程序明确使用基准兜底，而不是伪造持仓权重。
