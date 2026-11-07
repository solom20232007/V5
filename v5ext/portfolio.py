"""(8) Portfolio Simulator — يحوّل قائمة صفقات (backtest أو إشارات حية) لمنحنى رأس مال واقعي.

قواعد:
- حجم الصفقة = (رأس المال × RISK_PER_TRADE) ÷ مسافة الـ SL، بحد أقصى MAX_POSITION_PCT من رأس المال.
- أقصى عدد صفقات مفتوحة في نفس الوقت MAX_POSITIONS؛ الإشارة الزيادة بتتسجّل "skipped".
- مفيش رافعة: مجموع المراكز المفتوحة ≤ رأس المال المتاح (cash).
- الربح/الخسارة بيتحسب عند الإغلاق بالـ net_pct (بعد الرسوم والانزلاق).
"""
import heapq
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01        # 1% من رأس المال لكل صفقة
    max_position_pct: float = 0.25      # أقصى حجم مركز = 25% من رأس المال
    max_positions: int = 5
    compounding: bool = True


def simulate(trades, cfg=None):
    """trades: DataFrame فيه entry_time, exit_time, symbol, net_pct, sl_pct."""
    cfg = cfg or PortfolioConfig()
    cols = {"entry_time", "exit_time", "net_pct", "sl_pct"}
    if trades is None or not len(trades) or not cols.issubset(trades.columns):
        return pd.DataFrame(columns=["time", "equity"]), _summary([], [], cfg, 0, 0), pd.DataFrame()
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.dropna(subset=["entry_time", "exit_time", "net_pct", "sl_pct"]).sort_values("entry_time")

    cash = cfg.initial_capital
    equity_realized = cfg.initial_capital
    open_pos = []           # heap of (exit_time, idx, notional, net_pct)
    curve = [(t["entry_time"].iloc[0] if len(t) else pd.Timestamp.utcnow(), cfg.initial_capital)]
    log_rows = []
    skipped = 0

    def close_until(ts):
        nonlocal cash, equity_realized
        while open_pos and open_pos[0][0] <= ts:
            et, idx, notional, net = heapq.heappop(open_pos)
            pnl = notional * net / 100.0
            cash += notional + pnl
            equity_realized += pnl
            curve.append((et, equity_realized))
            log_rows[idx]["pnl"] = pnl
            log_rows[idx]["equity_after"] = equity_realized

    for row in t.itertuples(index=False):
        close_until(row.entry_time)
        base = equity_realized if cfg.compounding else cfg.initial_capital
        if len(open_pos) >= cfg.max_positions:
            skipped += 1
            continue
        sl = max(float(row.sl_pct), 0.1)
        notional = min(base * cfg.risk_per_trade / (sl / 100.0), base * cfg.max_position_pct, cash)
        if notional <= 1e-9:
            skipped += 1
            continue
        cash -= notional
        log_rows.append({"entry_time": row.entry_time, "exit_time": row.exit_time,
                         "symbol": getattr(row, "symbol", ""), "notional": notional,
                         "net_pct": float(row.net_pct), "pnl": None, "equity_after": None})
        heapq.heappush(open_pos, (row.exit_time, len(log_rows) - 1, notional, float(row.net_pct)))
    close_until(pd.Timestamp.max.tz_localize("UTC"))

    eq = pd.DataFrame(curve, columns=["time", "equity"]).sort_values("time").reset_index(drop=True)
    pos = pd.DataFrame(log_rows)
    return eq, _summary(eq["equity"].tolist(), pos, cfg, skipped, len(t)), pos


def _summary(equity, pos, cfg, skipped, total):
    out = {"config": asdict(cfg), "signals": int(total), "taken": int(total - skipped),
           "skipped_max_positions": int(skipped)}
    if not len(equity) or pos is None or not len(pos):
        out.update({"final_equity": cfg.initial_capital, "return_pct": 0.0, "max_drawdown_pct": 0.0,
                    "win_rate": 0.0, "profit_factor": 0.0})
        return out
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    pnl = pos["pnl"].astype(float)
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    out.update({
        "final_equity": round(float(eq[-1]), 2),
        "return_pct": round((eq[-1] / cfg.initial_capital - 1) * 100.0, 2),
        "max_drawdown_pct": round(float(dd), 2),
        "win_rate": round(float((pnl > 0).mean()), 4),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
        "avg_pnl": round(float(pnl.mean()), 3),
        "best_trade": round(float(pnl.max()), 2),
        "worst_trade": round(float(pnl.min()), 2),
    })
    try:
        p = pos.dropna(subset=["pnl"]).copy()
        p["month"] = pd.to_datetime(p["exit_time"], utc=True).dt.strftime("%Y-%m")
        out["monthly_pnl"] = {k: round(float(v), 2) for k, v in p.groupby("month")["pnl"].sum().items()}
    except Exception:  # noqa: BLE001
        pass
    return out


def trades_from_live_results(df):
    """صفقات البوت الحية المقفولة (نفس حسابات التقارير) بصيغة الصفقات."""
    from .performance import closed_trades
    t = closed_trades(df)
    if not len(t):
        return pd.DataFrame()
    out = pd.DataFrame({
        "entry_time": t["opened_at"], "exit_time": t["closed_at"], "symbol": t["symbol"],
        "net_pct": t["pnl_pct"], "lane": t["lane"], "outcome": t["exit_reason"],
    })
    # sl_pct لحساب حجم الصفقة
    m = df[["symbol", "final_time_utc", "sl_pct"]].copy()
    m["entry_time"] = pd.to_datetime(m["final_time_utc"], utc=True, errors="coerce")
    out = out.merge(m[["symbol", "entry_time", "sl_pct"]], on=["symbol", "entry_time"], how="left")
    out["sl_pct"] = pd.to_numeric(out["sl_pct"], errors="coerce").fillna(4.0)
    return out.dropna(subset=["entry_time", "exit_time", "net_pct"])
