"""اختبار السرعة: هل لقط النبضة في أول 1–3 دقايق (بدل 10–20 دقيقة) بيكسب؟

على شموع الدقيقة التاريخية (Binance 1m):
  كشف النبضة على الدقيقة t (مقفولة) — كل الشروط لازم تتحقق:
    • حجم آخر 3 دقايق ≥ V × الوسيط لنفس الطول خلال الـ 24 ساعة اللي قبلها
    • السعر طلع ≥ P% في آخر 3 دقايق، والدقيقة الأخيرة خضرا (بيتسارع مش بيهدى)
    • الشراء العدواني (Taker Buy) ≥ T من حجم آخر 3 دقايق
    • عدد الصفقات آخر 3 دقايق ≥ 2.5 × الوسيط (دخول ناس جديدة)
    • العملة ما كانتش طالعة أصلًا: تغيّر الساعة اللي قبل النبضة ≤ 3% (عايزين البداية مش النص)
    • BTC مش نازل: آخر 15 دقيقة > −0.5%
  الدخول: افتتاح الدقيقة t+1 (زي بوت WebSocket بيرد في ثواني)
  الوقف: تحت بداية النبضة (افتتاح t−2) بـ 0.2%، بين 1% و4%
  الأهداف: 1.5R / 3R / 4.5R بنفس إدارة الصفقة (ثلث عند كل هدف + تحريك الوقف) + مهلة

التكلفة متحفظة: رسوم + انزلاق عادي + انزلاق إضافي 0.10% لكل جهة (الدخول وقت الاندفاع أغلى).
"""
import numpy as np
import pandas as pd

from . import config as C
from .trade_manager import new_state, step

EXTRA_IMPACT_PCT = 0.10
COST_PCT = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT + EXTRA_IMPACT_PCT)

DETECTORS = {
    "strict": dict(vol_mult=8.0, min_move=1.5, taker=0.65),
    "medium": dict(vol_mult=5.0, min_move=1.0, taker=0.60),
}
EXITS = {
    "60m": dict(timeout_min=60),
    "20m": dict(timeout_min=20),
}


def features(df, btc_close_by_time=None):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    qv = df["quote_volume"].to_numpy(dtype=float)
    tq = df["taker_buy_quote"].to_numpy(dtype=float)
    tr = df["trades"].to_numpy(dtype=float)
    q3 = pd.Series(qv).rolling(3).sum()
    base_q = q3.shift(3).rolling(1440, min_periods=720).median()
    t3 = pd.Series(tr).rolling(3).sum()
    base_t = t3.shift(3).rolling(1440, min_periods=720).median()
    f = pd.DataFrame({
        "vol_ratio": (q3 / base_q).to_numpy(),
        "trades_ratio": (t3 / base_t).to_numpy(),
        "move3": (c / np.concatenate([[np.nan, np.nan], o[:-2]]) - 1) * 100,
        "last_green": c > o,
        "taker3": (pd.Series(tq).rolling(3).sum() / q3).to_numpy(),
        "pre1h": (pd.Series(c).shift(3) / pd.Series(c).shift(63) - 1).to_numpy() * 100,
    })
    if btc_close_by_time is not None:
        bc = pd.Series(df["open_time"].map(btc_close_by_time).to_numpy(dtype=float))
        f["btc15"] = (bc / bc.shift(15) - 1).to_numpy() * 100
    else:
        f["btc15"] = 0.0
    return f


def detect(f, vol_mult, min_move, taker):
    m = ((f.vol_ratio >= vol_mult) & (f.move3 >= min_move) & f.last_green & (f.taker3 >= taker)
         & (f.trades_ratio >= 2.5) & (f.pre1h <= 3.0) & (f.btc15.fillna(0) > -0.5))
    return np.where(m.to_numpy())[0]


def simulate_symbol(symbol, df, sig_idx, timeout_min, rr=(1.5, 3.0, 4.5), cooldown_min=120):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ot = df["open_time"].to_numpy()
    n = len(c)
    trades, busy_until = [], -1
    for t in sig_idx:
        if t <= busy_until or t + 2 >= n or t < 3:
            continue
        ei = t + 1
        entry = o[ei]
        sl = o[t - 2] * (1 - 0.002)
        sl_pct = (entry - sl) / entry * 100
        if sl_pct < 1.0:
            sl, sl_pct = entry * 0.99, 1.0
        if sl_pct > 4.0:
            continue
        risk = entry - sl
        st = new_state(entry, sl, entry + risk * rr[0], entry + risk * rr[1], entry + risk * rr[2], timeout_min)
        peak, low, j = entry, entry, ei
        for j in range(ei, n):
            peak, low = max(peak, h[j]), min(low, l[j])
            step(st, h[j], l[j], c[j], j - ei + 1)
            if st["closed"]:
                break
        if not st["closed"]:
            break
        net = st["realized_pct"] - COST_PCT
        trades.append({
            "symbol": symbol, "entry_time": pd.to_datetime(int(ot[ei]), unit="ms", utc=True),
            "exit_time": pd.to_datetime(int(ot[j]), unit="ms", utc=True), "entry": entry,
            "sl_pct": sl_pct, "reason": st["exit_reason"], "minutes": j - ei + 1,
            "gross_pct": st["realized_pct"], "net_pct": net, "r_multiple": net / sl_pct,
            "mfe_pct": (peak / entry - 1) * 100, "mae_pct": (low / entry - 1) * 100,
        })
        busy_until = j + cooldown_min
    return trades


def summary(t):
    if t is None or not len(t):
        return {"trades": 0, "win_rate": 0.0, "avg_net_pct": 0.0, "profit_factor": 0.0}
    p = t["net_pct"]
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    srt = p.sort_values(ascending=False)
    return {"trades": int(len(p)), "win_rate": round(float((p > 0).mean()), 4),
            "avg_net_pct": round(float(p.mean()), 3), "median_net_pct": round(float(p.median()), 3),
            "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
            "total_pct": round(float(p.sum()), 2),
            "avg_without_top3": round(float(srt.iloc[3:].mean()), 3) if len(p) > 3 else None,
            "avg_mfe_pct": round(float(t["mfe_pct"].mean()), 2), "avg_minutes": round(float(t["minutes"].mean()), 1),
            "by_reason": {k: int(v) for k, v in t["reason"].value_counts().items()}}


def experiment(klines_1m, btc_1m=None, train_frac=0.6):
    btc_map = dict(zip(btc_1m.open_time, btc_1m.close)) if btc_1m is not None and len(btc_1m) else None
    feats = {s: features(df, btc_map) for s, df in klines_1m.items()}
    all_t = {}
    for dn, dkw in DETECTORS.items():
        sigs = {s: detect(f, **dkw) for s, f in feats.items()}
        for en, ekw in EXITS.items():
            rows = []
            for s, df in klines_1m.items():
                rows += simulate_symbol(s, df, sigs[s], **ekw)
            t = pd.DataFrame(rows)
            all_t[f"{dn} / {en}"] = t.sort_values("entry_time").reset_index(drop=True) if len(t) else t
    starts = [int(df.open_time.iloc[0]) for df in klines_1m.values() if len(df)]
    ends = [int(df.open_time.iloc[-1]) for df in klines_1m.values() if len(df)]
    t0, t1 = pd.to_datetime(min(starts), unit="ms", utc=True), pd.to_datetime(max(ends), unit="ms", utc=True)
    cut = t0 + (t1 - t0) * train_frac
    res = {}
    for name, t in all_t.items():
        tr = t[t.entry_time < cut] if len(t) else t
        te = t[t.entry_time >= cut] if len(t) else t
        res[name] = {"train": summary(tr), "test": summary(te)}
    ok = [n for n in res if res[n]["train"]["trades"] >= 30]
    best = max(ok, key=lambda n: res[n]["train"]["avg_net_pct"]) if ok else None
    verdict = None
    if best:
        te = res[best]["test"]
        verdict = {"chosen_on_train": best, **{f"test_{k}": te.get(k) for k in
                                               ("trades", "avg_net_pct", "profit_factor", "avg_without_top3")},
                   "passes": bool(te["trades"] >= 30 and te["avg_net_pct"] > 0 and (te["profit_factor"] or 0) > 1.2
                                  and (te.get("avg_without_top3") or -1) > 0),
                   "rule": "PASS = test >= 30 trades, avg > 0, PF > 1.2, still > 0 without top 3 trades"}
    return {"test_from": cut.isoformat(), "symbols": len(klines_1m), "cost_pct_round_trip": COST_PCT,
            "results": res, "verdict": verdict}, all_t
