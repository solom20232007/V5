"""(7) Backtesting Engine مستقل.

المرحلة A (مكلفة، مرة واحدة): replay لكل شمعة 5m تاريخية عبر analyze() الأصلية في الماسح
  → نفس الـ score ونفس الـ setup ونفس الحسابات بالظبط (مش نسخة تقريبية).
  الناتج: scan records (كل شمعة score >= BT_LABEL_MIN_SCORE) + dataset للـ ML.

المرحلة B (رخيصة، تتكرر آلاف المرات في الـ optimizer): محاكاة قرار FINAL زي
  maybe_final_signal (تأكيدين خلال 45 دقيقة، cooldown 12 ساعة، BTC guard، lanes)
  + خروج بـ SL/TP1/Timeout على شموع 5m، مع رسوم وانزلاق.

حدود معروفة (مذكورة في التقرير): مفيش order book / whales تاريخي، وفلتر شمعة 1m
والـ cross-exchange مش متطبقين في الـ backtest.
"""
import contextlib
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import sentiment

log = logging.getLogger("v5ext.backtest")

DAY_BARS = 288
KL_LIMIT = 180   # = KLINES_LIMIT في الماسح (آخر صف = شمعة لسه بتتكوّن)

# الحقول اللي بتتحفظ من نتيجة analyze()
RECORD_FIELDS = [
    "symbol", "price", "5m", "15m", "1h", "4h", "24h", "volume_1h", "volume_15m",
    "acceleration", "prior_acceleration", "breakout", "close_location", "score", "setup",
    "quote_volume", "liquidity", "rsi14", "atr_pct", "buy_pressure", "buy_pressure_delta",
    "body_pct", "upper_wick_ratio", "distance_from_1h_high", "volume_persistence",
    "efficiency", "price_confirmation", "continuation_quality", "trend_continuity",
    "green_streak", "is_wash", "funding_rate", "new_listing_mode", "ema9", "ema21", "ema50",
    "trend",
]


# ==================================================================
# Scanner import (بدون تشغيل الحلقة الرئيسية)
# ==================================================================
_scanner = None


def get_scanner():
    global _scanner
    if _scanner is None:
        import importlib
        _scanner = importlib.import_module("binance_early_pump_v5")
    return _scanner


@contextlib.contextmanager
def scanner_params(overrides):
    """يطبّق عتبات مؤقتًا على globals الماسح ويرجّعها بعدين."""
    sc = get_scanner()
    old = {}
    try:
        for k, v in (overrides or {}).items():
            if hasattr(sc, k):
                old[k] = getattr(sc, k)
                setattr(sc, k, type(old[k])(v) if isinstance(old[k], (int, float))
                        and not isinstance(old[k], bool) else v)
        yield sc
    finally:
        for k, v in old.items():
            setattr(sc, k, v)


# ==================================================================
# Stage A: scan history
# ==================================================================
def _btc_context(btc_df):
    if btc_df is None or not len(btc_df):
        return {}
    c = btc_df["close"].to_numpy()
    t = btc_df["close_time"].to_numpy()
    out = {}
    for i in range(12, len(c)):
        out[int(t[i]) // 300_000] = (
            (c[i] / c[i - 3] - 1) * 100.0,
            (c[i] / c[i - 12] - 1) * 100.0,
        )
    return out


def scan_symbol(symbol, df, btc_ctx, fng_hist, min_score=None, stride=1):
    """يرجّع list[dict] لكل شمعة عدّت فلاتر analyze ووصل الـ score فيها لـ min_score."""
    sc = get_scanner()
    min_score = C.BT_LABEL_MIN_SCORE if min_score is None else min_score
    if df is None or len(df) < C.BT_WARMUP_BARS + 5:
        return []
    raw = df[["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]].to_numpy()
    rows = [[int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), int(r[6]),
             str(r[7]), int(r[8]), str(r[9]), str(r[10]), "0"] for r in raw]
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    qv = df["quote_volume"].to_numpy(); ct = df["close_time"].to_numpy()
    qv_cum = np.concatenate([[0.0], np.cumsum(qv)])

    window_holder = {}

    def fake_get_json(path, params=None, retries=3):
        return window_holder["rows"]

    real_get_json = sc.get_json
    real_funding = dict(sc.funding_rates_cache)
    sc.get_json = fake_get_json
    sc.funding_rates_cache.clear()
    out = []
    try:
        start = max(C.BT_WARMUP_BARS, KL_LIMIT)
        for i in range(start, len(df) - 1, stride):
            g15 = (c[i] / c[i - 3] - 1) * 100.0
            g1h = (c[i] / c[i - 12] - 1) * 100.0
            q24 = qv_cum[i + 1] - qv_cum[i + 1 - DAY_BARS]
            ch24 = (c[i] / o[i - DAY_BARS + 1] - 1) * 100.0
            # نفس الرفض المبكر في analyze (أسرع بكتير من استدعائها)
            if (q24 < sc.MIN_24H_QUOTE_VOLUME or ch24 > sc.MAX_24H_GAIN_PCT
                    or not (sc.MIN_15M_GAIN <= g15 <= sc.MAX_15M_GAIN)
                    or not (sc.MIN_1H_GAIN <= g1h <= sc.MAX_1H_GAIN)):
                continue
            # آخر صف = "الشمعة اللي بتتكوّن" (analyze بيتجاهلها) → نحط الشمعة الجاية مكانها
            window_holder["rows"] = rows[i - KL_LIMIT + 2: i + 2]
            res = sc.analyze(symbol, {"change": round(ch24, 3), "quote_volume": q24, "price": c[i]})
            if not res or res.get("_rejected") or res.get("score", 0) < min_score:
                continue
            rec = {k: res.get(k) for k in RECORD_FIELDS}
            ts = int(ct[i])
            b15, b1h = btc_ctx.get(ts // 300_000, (0.0, 0.0))
            rec.update({
                "ts": ts, "bar": i, "btc_15m": b15, "btc_1h": b1h,
                "btc_safe": not (b15 <= sc.BTC_CRASH_THRESHOLD_15M or b1h <= sc.BTC_CRASH_THRESHOLD_1H),
                "fng": sentiment.value_at(fng_hist, ts),
                "hour_utc": (ts // 3_600_000) % 24,
            })
            out.append(rec)
    finally:
        sc.get_json = real_get_json
        sc.funding_rates_cache.update(real_funding)
    return out


def _scan_worker(args):
    symbol, path, btc_ctx, fng_hist, min_score, stride = args
    import logging as _l
    _l.getLogger("scanner").setLevel(_l.WARNING)
    df = pd.read_csv(path)
    return scan_symbol(symbol, df, btc_ctx, fng_hist, min_score, stride)


def scan_history(klines, btc_df, fng_hist, workers=None, min_score=None, stride=1):
    """klines = {symbol: DataFrame}. يرجّع DataFrame للـ records."""
    btc_ctx = _btc_context(btc_df)
    workers = workers or max(1, min(4, (os.cpu_count() or 2)))
    records = []
    if workers > 1 and len(klines) > 1:
        os.makedirs(C.KLINE_CACHE_DIR, exist_ok=True)
        jobs = []
        for sym, df in klines.items():
            p = os.path.join(C.KLINE_CACHE_DIR, f"_bt_{sym}.csv")
            df.to_csv(p, index=False)
            jobs.append((sym, p, btc_ctx, fng_hist, min_score, stride))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for sym_recs in ex.map(_scan_worker, jobs):
                records.extend(sym_recs)
        for j in jobs:
            with contextlib.suppress(OSError):
                os.remove(j[1])
    else:
        for sym, df in klines.items():
            records.extend(scan_symbol(sym, df, btc_ctx, fng_hist, min_score, stride))
    rec = pd.DataFrame(records)
    if len(rec):
        rec = rec.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return rec


# ==================================================================
# Exits
# ==================================================================
def simulate_exit(o, h, l, c, entry_idx, entry, sl, tp1, tp2=None, timeout_bars=None,
                  mode="TP1"):
    """دخول عند open الشمعة entry_idx. SL و TP في نفس الشمعة → SL (متحفّظ زي الماسح).

    mode=TP1  : خروج كامل عند TP1 (نفس تعريف النجاح في الماسح والـ self-learn)
    mode=SPLIT: 50% عند TP1 ثم SL → Entry والباقي لـ TP2 أو Timeout
    """
    timeout_bars = timeout_bars or C.BT_TIMEOUT_BARS
    n = len(c)
    peak, low = entry, entry
    half_done = False
    realized = 0.0
    stop = sl
    last = min(n - 1, entry_idx + timeout_bars - 1)
    for j in range(entry_idx, last + 1):
        hi, lo = h[j], l[j]
        peak, low = max(peak, hi), min(low, lo)
        if lo <= stop:
            px = stop
            gross = realized + (0.5 if half_done else 1.0) * (px / entry - 1)
            oc = "BE" if half_done and stop >= entry else "SL"
            if half_done and oc == "SL":
                oc = "TP1_SL"
            return _exit(oc, gross, j - entry_idx + 1, entry, peak, low)
        if mode == "SPLIT":
            if not half_done and hi >= tp1:
                realized += 0.5 * (tp1 / entry - 1)
                half_done, stop = True, entry
                if tp2 and hi >= tp2:
                    return _exit("TP2", realized + 0.5 * (tp2 / entry - 1), j - entry_idx + 1, entry, peak, low)
                continue
            if half_done and tp2 and hi >= tp2:
                return _exit("TP2", realized + 0.5 * (tp2 / entry - 1), j - entry_idx + 1, entry, peak, low)
        elif hi >= tp1:
            return _exit("TP1", tp1 / entry - 1, j - entry_idx + 1, entry, peak, low)
    px = c[last]
    gross = realized + (0.5 if half_done else 1.0) * (px / entry - 1)
    return _exit("TP1_TIMEOUT" if half_done else "TIMEOUT", gross, last - entry_idx + 1, entry, peak, low)


def _exit(outcome, gross, bars, entry, peak, low):
    cost = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT) / 100.0
    return {"outcome": outcome, "gross_pct": gross * 100.0, "net_pct": (gross - cost) * 100.0,
            "bars_held": bars, "mfe_pct": (peak / entry - 1) * 100.0, "mae_pct": (low / entry - 1) * 100.0}


def simulate_managed(o, h, l, c, entry_idx, risk, timeout_min):
    """نفس إدارة الصفقة الحية (trade_manager) على شموع 5m."""
    from .trade_manager import new_state, step, net_pct
    entry = o[entry_idx]
    st = new_state(entry, risk["sl"], risk["tp1"], risk["tp2"], risk.get("tp3") or risk["tp2"] * 1.5, timeout_min)
    peak = low = entry
    n = len(c)
    j = entry_idx
    for j in range(entry_idx, n):
        peak, low = max(peak, h[j]), min(low, l[j])
        step(st, h[j], l[j], c[j], (j - entry_idx + 1) * 5)
        if st["closed"]:
            break
    if not st["closed"]:
        return None                      # البيانات خلصت قبل ما الصفقة تتقفل
    return {"outcome": st["exit_reason"], "gross_pct": st["realized_pct"], "net_pct": net_pct(st),
            "bars_held": j - entry_idx + 1, "mfe_pct": (peak / entry - 1) * 100.0,
            "mae_pct": (low / entry - 1) * 100.0, "warned": st["warned"], "stage": st["stage"]}


def label_records(records, klines, risk_overrides=None):
    """يضيف label (TP1 قبل SL) و r_multiple لكل record — ده الـ dataset بتاع الـ ML."""
    sc = get_scanner()
    labels, rs, nets = [], [], []
    arrays = {s: (d.open.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy())
              for s, d in klines.items()}
    with scanner_params(risk_overrides or {}):
        for rec in records.itertuples(index=False):
            arr = arrays.get(rec.symbol)
            if arr is None or rec.bar + 1 >= len(arr[0]):
                labels.append(np.nan); rs.append(np.nan); nets.append(np.nan); continue
            o, h, l, c = arr
            entry = o[rec.bar + 1]
            risk = sc.compute_risk_levels(entry, rec.atr_pct, rec.rsi14, rec.liquidity)
            if not risk:
                labels.append(np.nan); rs.append(np.nan); nets.append(np.nan); continue
            ex = simulate_exit(o, h, l, c, rec.bar + 1, entry, risk["sl"], risk["tp1"], risk["tp2"])
            # timeout قبل نهاية البيانات بالكامل = مجهول
            if ex["outcome"] == "TIMEOUT" and rec.bar + 1 + C.BT_TIMEOUT_BARS > len(c):
                labels.append(np.nan); rs.append(np.nan); nets.append(np.nan); continue
            labels.append(1 if ex["outcome"] == "TP1" else 0)
            nets.append(ex["net_pct"])
            rs.append(ex["net_pct"] / risk["sl_pct"])
    out = records.copy()
    out["label"] = labels
    out["r_multiple"] = rs
    out["net_pct"] = nets
    return out


# ==================================================================
# Stage B: simulate FINAL signals
# ==================================================================
def burst_thresholds(records, watch_score, quantile=None):
    """عتبات اندفاعة السوق من توزيع عدد العملات فوق WATCH في كل شمعة (الشموع الفاضية = 0)."""
    quantile = C.BURST_QUANTILE if quantile is None else quantile
    if records is None or not len(records):
        return None
    ts0, ts1 = int(records.ts.min()), int(records.ts.max())
    grid = pd.Series(0, index=np.arange(ts0 // 300_000, ts1 // 300_000 + 1))
    cnt = records[records.score >= watch_score].groupby(records.ts // 300_000).size()
    grid.loc[cnt.index] = cnt.values
    roll = grid.rolling(3, min_periods=1).sum()
    return {"thr1": max(2, int(np.ceil(grid.quantile(quantile)))),
            "thr3": max(3, int(np.ceil(roll.quantile(quantile))))}


def _burst_until_map(records, watch_score, thr):
    """{bar_index: blocked_until_ms}"""
    cnt = records[records.score >= watch_score].groupby(records.ts // 300_000).size()
    if not len(cnt):
        return set()
    full = cnt.reindex(np.arange(cnt.index.min(), cnt.index.max() + 1), fill_value=0)
    roll = full.rolling(3, min_periods=1).sum()
    hits = full.index[(full >= thr["thr1"]) | (roll >= thr["thr3"])]
    blocked = set()
    span = C.BURST_BLOCK_MIN // 5
    for b in hits:
        blocked.update(range(int(b), int(b) + span + 1))
    return blocked


def pullback_entry(o, h, l, c, sig_bar, retrace=0.5, wait_bars=24, rr=(1.5, 3.0, 4.5),
                   min_sl=1.5, max_sl=8.0, timeout_min=480):
    """دخول بعد ارتداد: بدل الشراء وقت الطلوع، استنى السعر يرجع جزء من الموجة ويأكّد.

    الموجة: من أقل سعر في الساعة قبل الطلوع (L) لأعلى سعر وقت الإشارة (H)، R = H − L.
    1) استنى لحد wait_bars شمعة إن السعر يلمس H − retrace×R
    2) لو نزل تحت H − 0.786×R → الإشارة اتلغت (الطلوع فشل)
    3) بعد اللمس: شمعة 5 دقايق خضرا تقفل فوق أعلى سعر في الشمعة اللي قبلها = تأكيد
    4) الدخول على افتتاح الشمعة اللي بعد التأكيد. الوقف تحت أقل سعر في الارتداد.
    يرجّع (entry_idx, risk) أو None.
    """
    i = int(sig_bar)
    if i < 16 or i + 2 >= len(c):
        return None
    H = float(max(h[i - 3:i + 1]))
    L = float(min(l[i - 15:i - 2]))
    R = H - L
    if R <= 0 or R / H < 0.01:
        return None
    zone, invalid = H - retrace * R, H - 0.786 * R
    touched, lowest = False, float("inf")
    last = min(len(c) - 2, i + wait_bars)
    for j in range(i + 1, last + 1):
        if l[j] <= invalid:
            return None
        lowest = min(lowest, l[j])
        if l[j] <= zone:
            touched = True
        if touched and c[j] > o[j] and c[j] > h[j - 1]:
            ei = j + 1
            entry = float(o[ei])
            sl = lowest * (1 - 0.002)
            sl_pct = (entry - sl) / entry * 100.0
            if sl_pct < min_sl:
                sl, sl_pct = entry * (1 - min_sl / 100.0), min_sl
            if sl_pct > max_sl or sl_pct <= 0:
                return None
            risk_abs = entry - sl
            tps = [entry + risk_abs * k for k in rr]
            return ei, {
                "entry": entry, "sl": sl, "sl_pct": sl_pct,
                "tp1": tps[0], "tp1_pct": (tps[0] / entry - 1) * 100, "tp1_rr": rr[0],
                "tp2": tps[1], "tp2_pct": (tps[1] / entry - 1) * 100, "tp2_rr": rr[1],
                "tp3": tps[2], "tp3_pct": (tps[2] / entry - 1) * 100, "tp3_rr": rr[2],
                "timeout_min": timeout_min,
            }
    return None


def simulate_signals(records, klines, params=None, exit_mode="MANAGED", ml_probs=None,
                     ml_min_prob=None, fng_mode=None, burst=None, spacing_min=0,
                     entry_mode="NOW", pullback=None, relaxed_trigger=False):
    """entry_mode: NOW = الدخول فورًا (الماسح الحالي) | PULLBACK = بعد ارتداد (pullback = kwargs).
    relaxed_trigger: يتجاهل final_candidate_eligible (فلاتر FINAL) — الارتداد هو الفلتر."""
    """burst: None = بدون فلتر، أو dict عتبات من burst_thresholds (محسوبة على فترة train).
    spacing_min: أقل مدة بين أي إشارتين (على مستوى المحفظة)."""
    """records: DataFrame من scan_history. params: overrides لعتبات الماسح.

    ml_probs: Series/dict {(symbol, ts): prob} — احتمالات out-of-sample بس.
    لو ml_min_prob متحدد: أي record مالوش احتمال OOS بيتجاهل (عشان مفيش تسريب).
    """
    params = dict(params or {})
    sc = get_scanner()
    arrays = {s: (d.open.to_numpy(), d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(),
                  d.open_time.to_numpy())
              for s, d in klines.items()}
    trades = []
    if records is None or not len(records):
        return pd.DataFrame()
    recs = records.to_dict("records")
    with scanner_params(params):
        expiry = sc.CANDIDATE_EXPIRY_MINUTES * 60_000
        cooldown = sc.FINAL_COOLDOWN_HOURS * 3_600_000
        cand, last_final = {}, {}
        busy_until = {}
        blocked = _burst_until_map(records, sc.WATCH_SCORE, burst) if burst else set()
        last_any = -10 ** 15
        for x in recs:
            sym, ts = x["symbol"], int(x["ts"])
            if not x.get("btc_safe", True):
                continue
            if x["score"] < sc.WATCH_SCORE:
                continue
            extra, blocked, _ = sentiment.score_adjustment(
                None if pd.isna(x.get("fng")) else x.get("fng"), fng_mode)
            if blocked:
                continue
            xx = dict(x)
            for k in ("rsi14", "atr_pct", "ema9", "ema21", "ema50"):
                if isinstance(xx.get(k), float) and math.isnan(xx[k]):
                    xx[k] = None
            xx["score"] = x["score"] - extra
            if not relaxed_trigger and not sc.final_candidate_eligible(xx):
                continue
            # == maybe_final_signal ==
            if sym in last_final and ts - last_final[sym] < cooldown:
                continue
            if ts < busy_until.get(sym, 0):
                continue
            if xx["score"] < sc.MIN_CANDIDATE_SCORE:
                continue
            if blocked and ts // 300_000 in blocked:
                continue
            if spacing_min and ts - last_any < spacing_min * 60_000:
                continue
            c = cand.get(sym)
            if not c or ts - c["last_ts"] > expiry:
                cand[sym] = {"count": 1, "last_ts": ts, "best": xx["score"], "first": xx}
                continue
            c["count"] += 1
            c["last_ts"] = ts
            c["best"] = max(c["best"], xx["score"])
            if c["count"] < sc.CONFIRMATIONS_REQUIRED:
                continue
            first = c["first"]
            if not (xx.get("continuation_quality") and xx["5m"] > -0.3
                    and xx["15m"] >= first["15m"] - 1.0
                    and xx["volume_1h"] >= first["volume_1h"] * 0.70):
                continue
            prob = None
            if ml_probs is not None:
                prob = ml_probs.get((sym, ts))
            if ml_min_prob is not None:
                if prob is None or prob < ml_min_prob:
                    continue
            lane = sc.final_lane(xx)
            arr = arrays.get(sym)
            if arr is None:
                continue
            o, h, l, cl, ot = arr
            ei = int(x["bar"]) + 1
            if ei >= len(o):
                continue
            from .trade_manager import HORIZONS, atr_1h_from_5m, classify_horizon, horizon_risk
            if entry_mode == "PULLBACK":
                pb = pullback_entry(o, h, l, cl, int(x["bar"]), **(pullback or {}))
                # المرشح استُهلك سواء اتدخل ولا اتلغى (مرشح واحد ليه نتيجة واحدة)
                cand.pop(sym, None)
                if pb is None:
                    last_final[sym] = ts
                    continue
                ei, risk = pb
                entry, hz, timeout_min = risk["entry"], "PULLBACK", risk["timeout_min"]
            else:
                entry = o[ei]
                hz = classify_horizon(dict(xx, lane=lane))
                a1h = atr_1h_from_5m(h, l, cl, ei - 1) if hz == "SWING" else None
                risk = horizon_risk(sc.compute_risk_levels, entry, xx, hz, a1h)
                if risk is None:
                    hz = "INTRADAY"
                    risk = horizon_risk(sc.compute_risk_levels, entry, xx, hz)
                if not risk:
                    continue
                timeout_min = HORIZONS[hz][2]
            if exit_mode == "MANAGED":
                ex = simulate_managed(o, h, l, cl, ei, risk, timeout_min)
                if ex is None:
                    continue
            else:
                ex = simulate_exit(o, h, l, cl, ei, entry, risk["sl"], risk["tp1"], risk["tp2"],
                                   timeout_bars=timeout_min // 5, mode=exit_mode)
            last_final[sym] = ts
            last_any = ts
            cand.pop(sym, None)
            exit_ts = int(ot[min(ei + ex["bars_held"] - 1, len(ot) - 1)]) + 300_000
            busy_until[sym] = exit_ts
            trades.append({
                "entry_time": pd.to_datetime(int(ot[ei]), unit="ms", utc=True),
                "exit_time": pd.to_datetime(exit_ts, unit="ms", utc=True),
                "symbol": sym, "lane": lane, "horizon": hz, "setup": x.get("setup"), "score": x["score"],
                "entry": entry, "sl_pct": risk["sl_pct"], "tp1_pct": risk["tp1_pct"],
                "tp1_rr": risk["tp1_rr"], "ml_prob": prob, "fng": x.get("fng"),
                **ex, "r_multiple": ex["net_pct"] / risk["sl_pct"],
            })
    return pd.DataFrame(trades)


# ==================================================================
# Metrics
# ==================================================================
def metrics(trades):
    if trades is None or not len(trades):
        return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "avg_net_pct": 0.0,
                "profit_factor": 0.0, "total_r": 0.0, "max_dd_r": 0.0, "max_losing_streak": 0}
    r = trades["r_multiple"].to_numpy(dtype=float)
    wins = trades["net_pct"] > 0
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max()) if len(eq) else 0.0
    streak = best = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    out = {
        "trades": int(len(r)),
        "win_rate": round(float(wins.mean()), 4),
        "expectancy_r": round(float(r.mean()), 4),
        "avg_net_pct": round(float(trades["net_pct"].mean()), 4),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "total_r": round(float(r.sum()), 3),
        "max_dd_r": round(dd, 3),
        "max_losing_streak": int(best),
        "avg_bars_held": round(float(trades["bars_held"].mean()), 1),
    }
    out["by_lane"] = {
        str(k): {"trades": int(len(g)), "win_rate": round(float((g["net_pct"] > 0).mean()), 3),
                 "expectancy_r": round(float(g["r_multiple"].mean()), 3)}
        for k, g in trades.groupby(trades["lane"].fillna("NONE"))
    }
    if "horizon" in trades:
        out["by_horizon"] = {
            str(k): {"trades": int(len(g)), "win_rate": round(float((g["net_pct"] > 0).mean()), 3),
                     "avg_net_pct": round(float(g["net_pct"].mean()), 3)}
            for k, g in trades.groupby(trades["horizon"].fillna("NONE"))
        }
    out["by_outcome"] = {str(k): int(v) for k, v in trades["outcome"].value_counts().items()}
    return out
