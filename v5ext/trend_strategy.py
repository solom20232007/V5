"""استراتيجية الاتجاه اليومي (Daily Trend Following) — اختبار مستقل.

قواعد بسيطة ومعروفة (Donchian / Turtle-style) — عدد معاملات قليل عن قصد:
  • فلتر السوق: BTC فوق متوسط 50 يوم (اختياري)
  • الدخول: إغلاق يومي فوق أعلى سعر في آخر N يوم، والعملة فوق متوسط 50 يوم
            → الشراء على افتتاح اليوم اللي بعده
  • الوقف الأولي: الدخول − 2 × ATR(14)
  • الخروج: إغلاق يومي تحت أقل سعر في آخر M يوم (وقف متحرك) → البيع على افتتاح اليوم اللي بعده
            أو لمس الوقف الأولي خلال اليوم
  • الرسوم + الانزلاق نفس الـ backtest التاني لكل جهة

الاختيار بين الإعدادات على فترة train، والحكم مرة واحدة على test.
"""
import numpy as np
import pandas as pd

from . import config as C

COST_PCT = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT)


def _atr(h, l, c, n=14):
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def btc_regime(btc_daily, sma=50):
    """{open_time: True/False} — BTC close فوق متوسطه (بيانات اليوم المقفول)."""
    c = btc_daily["close"].to_numpy()
    ma = pd.Series(c).rolling(sma).mean().to_numpy()
    return dict(zip(btc_daily["open_time"].to_numpy().tolist(), (c > ma).tolist()))


def backtest_symbol(symbol, df, entry_n=20, exit_n=10, atr_mult=2.0, trend_sma=50,
                    regime=None, max_hold_days=120):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ot = df["open_time"].to_numpy()
    n = len(c)
    if n < max(entry_n, trend_sma, 15) + 5:
        return []
    hi_n = pd.Series(h).rolling(entry_n).max().shift(1).to_numpy()   # قمة آخر N يوم (قبل النهارده)
    lo_m = pd.Series(l).rolling(exit_n).min().shift(1).to_numpy()
    ma = pd.Series(c).rolling(trend_sma).mean().to_numpy()
    atr = _atr(h, l, c)
    trades = []
    i = max(entry_n, trend_sma, 15)
    while i < n - 1:
        ok = c[i] > hi_n[i] and c[i] > ma[i] and atr[i] == atr[i]
        if ok and regime is not None:
            ok = bool(regime.get(int(ot[i]), False))
        if not ok:
            i += 1
            continue
        ei = i + 1
        entry = o[ei]
        stop = entry - atr_mult * atr[i]
        if stop <= 0 or entry <= 0:
            i += 1
            continue
        sl_pct = (entry - stop) / entry * 100
        exit_px, reason, xi = None, None, None
        for j in range(ei, min(n, ei + max_hold_days)):
            if l[j] <= stop:                                   # الوقف الأولي خلال اليوم
                exit_px, reason, xi = min(stop, o[j]), "STOP", j
                break
            if j + 1 < n and c[j] < lo_m[j]:                   # كسر قاع M يوم → خروج بكرة
                exit_px, reason, xi = o[j + 1], "TRAIL", j + 1
                break
        if exit_px is None:
            if ei + max_hold_days < n:
                xi = ei + max_hold_days
                exit_px, reason = o[xi], "MAX_HOLD"
            else:
                break                                          # الصفقة لسه مفتوحة آخر البيانات
        gross = (exit_px / entry - 1) * 100
        trades.append({
            "symbol": symbol, "entry_time": pd.to_datetime(int(ot[ei]), unit="ms", utc=True),
            "exit_time": pd.to_datetime(int(ot[xi]), unit="ms", utc=True),
            "entry": entry, "exit": exit_px, "sl_pct": sl_pct, "reason": reason,
            "days": int(xi - ei), "gross_pct": gross, "net_pct": gross - COST_PCT,
            "r_multiple": (gross - COST_PCT) / sl_pct,
        })
        i = xi + 1
    return trades


def run(klines_daily, btc_daily, use_regime=True, **params):
    reg = btc_regime(btc_daily) if use_regime and btc_daily is not None and len(btc_daily) else None
    out = []
    for sym, df in klines_daily.items():
        out.extend(backtest_symbol(sym, df, regime=reg, **params))
    t = pd.DataFrame(out)
    return t.sort_values("entry_time").reset_index(drop=True) if len(t) else t


def summary(t):
    if t is None or not len(t):
        return {"trades": 0, "win_rate": 0.0, "avg_net_pct": 0.0, "profit_factor": 0.0,
                "total_pct": 0.0, "avg_days": 0, "best": None, "worst": None}
    p = t["net_pct"].astype(float)
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    return {"trades": int(len(p)), "win_rate": round(float((p > 0).mean()), 4),
            "avg_net_pct": round(float(p.mean()), 3),
            "median_net_pct": round(float(p.median()), 3),
            "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
            "total_pct": round(float(p.sum()), 2), "avg_days": round(float(t["days"].mean()), 1),
            "best": round(float(p.max()), 2), "worst": round(float(p.min()), 2),
            "by_reason": {k: int(v) for k, v in t["reason"].value_counts().items()}}


VARIANTS = {
    "20/10 + BTC filter": dict(entry_n=20, exit_n=10, use_regime=True),
    "20/10 no filter": dict(entry_n=20, exit_n=10, use_regime=False),
    "55/20 + BTC filter": dict(entry_n=55, exit_n=20, use_regime=True),
    "55/20 no filter": dict(entry_n=55, exit_n=20, use_regime=False),
}


def experiment(klines_daily, btc_daily, train_frac=0.6, portfolio_cfg=None):
    from . import portfolio as P
    all_t = {name: run(klines_daily, btc_daily, **kw) for name, kw in VARIANTS.items()}
    starts = [t["entry_time"].min() for t in all_t.values() if len(t)]
    ends = [t["entry_time"].max() for t in all_t.values() if len(t)]
    if not starts:
        return {"error": "no trades"}
    t0, t1 = min(starts), max(ends)
    cut = t0 + (t1 - t0) * train_frac
    cfg = portfolio_cfg or P.PortfolioConfig()
    res = {}
    for name, t in all_t.items():
        tr, te = t[t.entry_time < cut], t[t.entry_time >= cut]
        _, port_te, _ = P.simulate(te, cfg)
        res[name] = {"train": summary(tr), "test": summary(te),
                     "test_portfolio": {k: port_te.get(k) for k in
                                        ("return_pct", "max_drawdown_pct", "taken", "skipped_max_positions")}}
    ok = [n for n in res if res[n]["train"]["trades"] >= 20]
    best = max(ok, key=lambda n: res[n]["train"]["avg_net_pct"]) if ok else None
    # مرجع: شراء BTC والاحتفاظ بيه في فترة test
    bh = None
    b = btc_daily[pd.to_datetime(btc_daily.open_time, unit="ms", utc=True) >= cut]
    if len(b) > 1:
        bh = round((b.close.iloc[-1] / b.open.iloc[0] - 1) * 100, 2)
    verdict = None
    if best:
        te, pt = res[best]["test"], res[best]["test_portfolio"]
        verdict = {
            "chosen_on_train": best, "test_trades": te["trades"], "test_avg_net_pct": te["avg_net_pct"],
            "test_profit_factor": te["profit_factor"], "test_portfolio_return_pct": pt.get("return_pct"),
            "test_portfolio_max_dd_pct": pt.get("max_drawdown_pct"), "btc_buy_hold_test_pct": bh,
            "passes": bool(te["trades"] >= 20 and te["avg_net_pct"] > 0 and (te["profit_factor"] or 0) > 1.2
                           and (pt.get("return_pct") or 0) > 0),
            "rule": "PASS = test >= 20 trades, avg > 0, PF > 1.2, portfolio return > 0",
        }
    return {"test_from": cut.isoformat(), "symbols": len(klines_daily), "results": res,
            "verdict": verdict, "btc_buy_hold_test_pct": bh}
