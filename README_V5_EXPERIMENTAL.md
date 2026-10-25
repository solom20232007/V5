# Binance Early Pump Scanner V5 — Experimental

## What changed from the uploaded V4-derived code
- Keeps the project isolated from V4: V5-specific filenames for the entry-quality module, self-learning script, signal-result CSV, watch log, diagnostics log, and GitHub Actions concurrency group.
- Adds a bounded Smart Money Score using buy pressure, 15m volume ratio, volume acceleration, spot order-book imbalance, and (when a Binance USD-M futures contract exists) open-interest value change.
- Uses Binance spot `/api/v3/depth` for the order book and futures `/futures/data/openInterestHist` for OI. Missing data is treated as unavailable, not as a bullish signal.
- Adds a fast lane with strict checks and no-wash-trading protection.
- Allows analysis from 25 completed candles and uses median volume baselines that can work with shorter histories.
- Raises candidate processing limit to 100 by default; configurable with `MAX_PROCESS_PER_CYCLE`.
- Adds environment switches: `SMART_MONEY_ENABLED=0`, `FAST_EXPLOSIVE_ENABLED=0`, `SMART_MONEY_TOP_N=30`, `SMART_MONEY_WEIGHT=0.30`, `MIN_KLINES=25`.

## Important operating notes
- This is experimental code, not a guarantee of profitable signals. Order-book depth is a snapshot and can be spoofed; OI is futures-only and not available for every spot pair.
- The Smart Money Score is a heuristic ranking feature, not proof of institutional activity.
- Run V4 and V5 separately, preferably in separate repositories and with separate Telegram/automation settings. Do not run both with the same mutable state files.
- Verify GitHub Actions secrets (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) and test with `workflow_dispatch` before leaving it unattended.
- This archive was statically checked for Python syntax. No live exchange scan or profitability backtest was run in this environment.
