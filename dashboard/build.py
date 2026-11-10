"""(10) Dashboard — يجمع كل البيانات في JSON واحد ويحقنه في template.html.

- build_static(): يكتب docs/index.html (ينفع GitHub Pages: Settings → Pages → /docs).
- collect(): نفس البيانات لـ dashboard/app.py (Flask) اللي بيقدمها live.
"""
import json
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from v5ext import config as C  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")


def _p(path):
    return os.path.join(ROOT, path)


def _read_json(path):
    try:
        with open(_p(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _read_csv(path, **kw):
    try:
        return pd.read_csv(_p(path), low_memory=False, **kw)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:  # noqa: BLE001
            pass
        if isinstance(v, float) and not math.isfinite(v):
            return None
    return v


def _records(df, cols, limit=None):
    if df is None or not len(df):
        return []
    have = [c for c in cols if c in df.columns]
    d = df[have]
    if limit:
        d = d.tail(limit)
    return [{k: _clean(v) for k, v in row.items()} for row in d.iloc[::-1].to_dict("records")]


def _curve(df, max_points=600):
    if df is None or not len(df) or "equity" not in df:
        return []
    step = max(1, len(df) // max_points)
    d = df.iloc[::step]
    return [{"t": str(t)[:16], "v": round(float(v), 2)} for t, v in zip(d["time"], d["equity"])]


def collect():
    from v5ext import performance as PF
    live = _read_csv(C.TRACK_FILE)
    tables = PF.period_tables(live)
    allst = tables["all"]
    s30 = PF.summarize(PF._since(PF.closed_trades(live), 30))
    kpi = {"signals": int(len(live)), "closed": allst["trades"], "wins": allst["wins"],
           "win_rate": allst["win_rate"], "total_pct": round(allst["total_pct"], 2),
           "usd": round(allst["usd"], 2), "win_rate_30d": s30["win_rate"], "trades_30d": s30["trades"],
           "open": int((live.get("trade_status", pd.Series(dtype=object)).astype(str) == "OPEN").sum()) if len(live) else 0,
           "last_signal": str(live["final_time_utc"].iloc[-1])[:16] if len(live) else None,
           "min_for_rate": PF.MIN_FOR_RATE, "stake": PF.STAKE_USD}

    fng = (_read_json(C.FNG_CACHE_FILE) or {})
    hist = fng.get("history") or {}
    fng_hist = [{"d": k, "v": v} for k, v in sorted(hist.items())[-120:]]

    watch = _read_csv(C.WATCH_LOG_FILE)
    return {
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "kpi": kpi,
        "paper": bool(C.PAPER_MODE),
        "pnl": tables,
        "fng": fng.get("current") or {},
        "fng_history": fng_hist,
        "signals": _records(live, [
            "final_time_utc", "symbol", "horizon", "lane", "score", "entry_price", "sl_pct", "tp1_pct",
            "tp2_pct", "tp3_pct", "trade_status", "tp_hits", "warned", "exit_reason", "trade_pnl_pct",
            "trade_minutes", "outcome", "exit_pct", "mfe_pct", "mae_pct", "status", "obi_1_0pct",
            "whale_net_ratio", "oi_change_1h", "fng"], limit=80),
        "watch": _records(watch, ["logged_time_utc", "symbol", "setup", "score", "gain_15m",
                                  "gain_1h", "volume_15m", "acceleration", "liquidity"], limit=40),
        "live_equity": _curve(_read_csv(C.LIVE_PORTFOLIO_EQUITY_FILE)),
        "live_portfolio": _read_json(C.LIVE_PORTFOLIO_SUMMARY_FILE),
        "bt_equity": _curve(_read_csv(C.PORTFOLIO_EQUITY_FILE)),
        "bt_portfolio": _read_json(C.PORTFOLIO_SUMMARY_FILE),
        "backtest": _read_json(C.BACKTEST_SUMMARY_FILE),
        "model": _read_json(C.MODEL_META_FILE),
        "bayes": _read_json(C.BAYES_STATS_FILE),
        "optimizer": _read_json(C.OPTIMIZER_REPORT_FILE),
    }


def render(data):
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    return html.replace("/*__DATA__*/null", blob)


def build_static(out_dir=None):
    out_dir = out_dir or _p(C.DOCS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(collect()))
    # GitHub Pages: منع Jekyll من تجاهل الملفات
    open(os.path.join(out_dir, ".nojekyll"), "a").close()
    return path


if __name__ == "__main__":
    print(build_static())
