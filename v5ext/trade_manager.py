"""إدارة الصفقة بعد الإشارة — نفس المنطق بالظبط في المتابعة الحية وفي الـ backtest.

الخطة (اللي مكتوبة في رسالة الإشارة):
  • TP1  → بيع ثلث الكمية + نقل الوقف لسعر الدخول (Break-even)
  • TP2  → بيع ثلث تاني + نقل الوقف لـ TP1
  • TP3  → بيع الباقي (الصفقة اتقفلت بكل الأهداف)
  • الوقف → بيع كل الباقي
  • نزول السعر لنص المسافة بين الدخول والوقف (قبل TP1) → تنبيه "الإشارة بتفشل"
  • انتهاء مدة المتابعة → إغلاق الباقي بسعر السوق

لو الوقف والهدف اتلمسوا في نفس الشمعة → الوقف بيتحسب الأول (افتراض متحفّظ).
الربح/الخسارة الصافي = الإجمالي − (رسوم + انزلاق) للدخول والخروج.
"""
from . import config as C

WARN_FRACTION = 0.5          # تنبيه لما السعر يقطع 50% من المسافة للوقف
PART = 1.0 / 3.0


def new_state(entry, sl, tp1, tp2, tp3, timeout_min):
    return {
        "entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2),
        "tp3": float(tp3), "stop": float(sl), "stage": 0, "remaining": 1.0,
        "realized_pct": 0.0, "closed": False, "exit_reason": None, "warned": False,
        "timeout_min": float(timeout_min), "close_min": None, "last_price": float(entry),
    }


def _take(st, frac, price):
    st["realized_pct"] += frac * (price / st["entry"] - 1.0) * 100.0
    st["remaining"] = max(0.0, st["remaining"] - frac)


def _close(st, price, reason, minute):
    if st["remaining"] > 1e-9:
        _take(st, st["remaining"], price)
    st["closed"], st["exit_reason"], st["close_min"] = True, reason, round(minute, 1)
    st["last_price"] = price


def step(st, hi, lo, close, minute):
    """يعالج شمعة واحدة. يرجّع list بالأحداث بالترتيب."""
    if st["closed"]:
        return []
    ev = []
    entry = st["entry"]
    # 1) الوقف أولًا
    if lo <= st["stop"]:
        reason = {0: "SL", 1: "BE", 2: "LOCK_TP1"}.get(st["stage"], "SL")
        _close(st, st["stop"], reason, minute)
        return [reason]
    # 2) الأهداف
    if st["stage"] == 0 and hi >= st["tp1"]:
        _take(st, PART, st["tp1"]); st["stage"] = 1; st["stop"] = entry; ev.append("TP1")
    if st["stage"] == 1 and hi >= st["tp2"]:
        _take(st, PART, st["tp2"]); st["stage"] = 2; st["stop"] = st["tp1"]; ev.append("TP2")
    if st["stage"] == 2 and hi >= st["tp3"]:
        _close(st, st["tp3"], "TP3", minute); ev.append("TP3")
        return ev
    st["last_price"] = close
    # 3) تنبيه الفشل المبكر
    if st["stage"] == 0 and not st["warned"]:
        if close <= entry - WARN_FRACTION * (entry - st["sl"]):
            st["warned"] = True
            ev.append("WARN")
    # 4) انتهاء المدة
    if minute >= st["timeout_min"]:
        _close(st, close, "TIMEOUT", minute)
        ev.append("TIMEOUT")
    return ev


def net_pct(st):
    cost = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT)
    return st["realized_pct"] - cost


def open_pnl_pct(st, price):
    """الربح/الخسارة الحالي (المحقق + غير المحقق) قبل الرسوم."""
    return st["realized_pct"] + st["remaining"] * (price / st["entry"] - 1.0) * 100.0


def atr_pct(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    a = sum(trs[-period:]) / period
    return a / closes[-1] * 100.0 if closes[-1] > 0 else None


def atr_1h_from_5m(h, l, c, end_idx, period=14):
    """ATR% على ساعة محسوب من شموع 5m (للـ backtest). end_idx = آخر شمعة مقفولة."""
    n_bars = (period + 1) * 12
    start = end_idx - n_bars + 1
    if start < 0:
        return None
    hs, ls, cs = [], [], []
    for k in range(start, end_idx + 1, 12):
        hs.append(max(h[k:k + 12])); ls.append(min(l[k:k + 12])); cs.append(c[k + 11])
    return atr_pct(hs, ls, cs, period)


# ------------------------------------------------------------------
# أنواع الفرص — المدة في الرسالة = مهلة المتابعة الفعلية (مفيش تعارض)
# ------------------------------------------------------------------
# key: (الاسم بالعربي، مدة المتابعة، مهلة بالدقائق، فريم ATR، مضاعف ATR، أقل وقف%، أقصى وقف%، RR للأهداف أو None = ثوابت الماسح)
HORIZONS = {
    "SCALPING": ("سريعة (Scalping)", "حتى 90 دقيقة", 90, "5m", 1.5, 2.5, 6.0, (1.0, 1.8, 2.5)),
    "INTRADAY": ("خلال اليوم (Intraday)", "حتى 8 ساعات", 480, "5m", None, None, None, None),
    "SWING": ("متعددة الأيام (Swing)", "حتى 3 أيام", 4320, "1h", 2.5, 5.0, 15.0, (1.5, 3.0, 4.5)),
}


def _f(v, default=0.0):
    try:
        v = float(v)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def classify_horizon(x):
    lane = x.get("lane")
    g15, g4h, g24 = _f(x.get("15m")), _f(x.get("4h")), _f(x.get("24h"))
    accel, atr = _f(x.get("acceleration")), _f(x.get("atr_pct"))
    rsi = _f(x.get("rsi14"), 50)
    if lane in ("EXPLOSIVE", "SQUEEZE") or (g15 >= 3 and accel >= 2.2) or atr >= 5:
        return "SCALPING"
    if (x.get("trend") == "STRONG_UP" and g4h >= 3 and _f(x.get("volume_persistence")) >= 2
            and g24 <= 15 and rsi < 72 and x.get("liquidity") in ("HIGH", "GOOD", "MEDIUM")
            and lane in ("PERSISTENT", "STRICT", "REVERSAL")):
        return "SWING"
    return "INTRADAY"


def horizon_risk(compute_risk_levels, entry, x, horizon, atr_1h=None):
    """يحسب الوقف والأهداف حسب نوع الفرصة باستخدام compute_risk_levels بتاعة الماسح."""
    name, _, _, frame, mult, mn, mx, rr = HORIZONS[horizon]
    atr = x.get("atr_pct")
    if frame == "1h":
        if not atr_1h:
            return None          # من غير ATR ساعة مينفعش نحسب Swing صح → يتحوّل Intraday
        atr = atr_1h
    kw = {}
    if mult is not None:
        kw.update(atr_mult=mult, min_sl=mn, max_sl=mx)
    if rr is not None:
        kw.update(rr1=rr[0], rr2=rr[1], rr3=rr[2])
    return compute_risk_levels(entry, atr, x.get("rsi14"), x.get("liquidity"), **kw)
