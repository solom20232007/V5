"""اختبار حي على الورق لاستراتيجية الاتجاه اليومي (نفس قواعد trend_strategy بالظبط — إعداد 20/10 + فلتر BTC).

بيشتغل مرة واحدة في اليوم بعد قفل الشمعة اليومية (00:00 UTC = 2 الفجر بتوقيت القاهرة):
  1) يقفل الصفقات اللي اتحققت شروط خروجها على الشمعة اللي اتقفلت
  2) يفتح صفقات جديدة على افتتاح اليوم الجديد للعملات اللي كسرت قمة 20 يوم
  3) يبعت ملخص بالأحداث (لو فيه)، وكل سبت ملخص أسبوعي

مفيش فلوس حقيقية — الهدف قياس الأداء على المستقبل (من غير انحياز اختيار العملات اللي في الـ backtest).
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from .binance_data import spot_get, top_usdt_symbols

log = logging.getLogger("v5ext.trend_live")

STATE_FILE = os.path.join(C.STATE_DIR, "trend_paper.json")
TRADES_FILE = os.path.join(C.REPORTS_DIR, "trend_paper_trades.csv")
PARAMS = dict(entry_n=20, exit_n=10, atr_mult=2.0, trend_sma=50, btc_sma=50, max_hold_days=120)
MAX_POSITIONS = 10
RISK_PER_TRADE = 0.01
MAX_POSITION_PCT = 0.15
START_CAPITAL = 1000.0
COST_PCT = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT)
RUN_AFTER_MIN = 10            # استنى 10 دقايق بعد نص الليل UTC عشان الشمعة تتقفل عند Binance
ENTRY_WINDOW_HOURS = 3        # الدخول بس لو التشغيل حصل خلال 3 ساعات من فتح اليوم (غير كده السعر قديم)


def _load():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {"last_day": None, "open": {}, "closed": [], "skipped": 0, "started": None,
                "equity": START_CAPITAL, "last_weekly": None}


def _save(st):
    os.makedirs(C.STATE_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_FILE)
    if st["closed"]:
        os.makedirs(C.REPORTS_DIR, exist_ok=True)
        pd.DataFrame(st["closed"]).to_csv(TRADES_FILE, index=False)


def _daily(symbol, limit=80):
    rows = spot_get("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": limit}, retries=2)
    if not rows:
        return None
    df = pd.DataFrame([r[:6] for r in rows], columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = df["open_time"].astype("int64")
    for k in ("open", "high", "low", "close"):
        df[k] = df[k].astype(float)
    return df


def _atr(h, l, c, n=14):
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def _day(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def universe(scanner):
    allowed = None
    if scanner.SHARIAH_FILTER_ENABLED:
        allowed = set(scanner.SHARIAH_COMPLIANT_BASES) | scanner.load_shariah_cache() | scanner.load_user_shariah_overrides()
    exclude = set(scanner.STABLECOIN_BASES) | set(scanner.MEGA_CAP_BASES) | set(scanner.STABLE_BASES)
    return top_usdt_symbols(n=120, min_quote_volume=scanner.MIN_24H_QUOTE_VOLUME,
                            exclude_bases=exclude, allowed_bases=allowed)


def _fmt(v):
    return f"{float(v):.8g}"


def _close_trade(st, sym, pos, exit_px, reason, exit_day):
    gross = (exit_px / pos["entry"] - 1) * 100
    net = gross - COST_PCT
    notional = pos.get("notional", 0.0)
    pnl = notional * net / 100
    st["equity"] = round(st.get("equity", START_CAPITAL) + pnl, 2)
    rec = {"symbol": sym, "entry_day": pos["entry_day"], "exit_day": exit_day, "entry": pos["entry"],
           "exit": exit_px, "stop": pos["stop"], "sl_pct": pos["sl_pct"], "reason": reason,
           "days": (pd.Timestamp(exit_day) - pd.Timestamp(pos["entry_day"])).days,
           "gross_pct": round(gross, 3), "net_pct": round(net, 3), "notional": round(notional, 2),
           "pnl_usd": round(pnl, 2), "equity_after": st["equity"]}
    st["closed"].append(rec)
    return rec


def process(scanner, now=None, fetch=_daily, symbols=None, entry_price="current"):
    """entry_price: current = آخر سعر وقت التشغيل (اللي إنت فعلًا تقدر تاخده) | open = افتتاح اليوم (للمطابقة مع الـ backtest)."""
    """يرجّع list بالرسايل اللي المفروض تتبعت. بيعدّل الحالة ويحفظها."""
    now = now or datetime.now(timezone.utc)
    st = _load()
    st["started"] = st.get("started") or now.strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")
    if st.get("last_day") == today:
        return []
    if now.hour == 0 and now.minute < RUN_AFTER_MIN:
        return []

    btc = fetch("BTCUSDT", 80)
    if btc is None or len(btc) < PARAMS["btc_sma"] + 2:
        raise RuntimeError("BTC daily data unavailable")
    if _day(int(btc.open_time.iloc[-1])) != today:
        raise RuntimeError("today's daily candle not open yet")
    bc = btc.close.to_numpy()[:-1]
    regime = bool(bc[-1] > np.mean(bc[-PARAMS["btc_sma"]:]))

    events = {"opened": [], "closed": [], "skipped": [], "late": False}
    syms = symbols if symbols is not None else universe(scanner)
    data = {}
    for s in set(syms) | set(st["open"]):
        try:
            df = fetch(s, 80)
            if df is not None and len(df) >= PARAMS["trend_sma"] + 3 and _day(int(df.open_time.iloc[-1])) == today:
                data[s] = df
        except Exception as e:  # noqa: BLE001
            log.debug(f"trend daily {s}: {e}")

    # 1) الخروج (على كل الشموع المقفولة من آخر مرة)
    for s, pos in list(st["open"].items()):
        df = data.get(s)
        if df is None:
            continue
        closed = df.iloc[:-1]
        o, h, l, c = (closed[k].to_numpy() for k in ("open", "high", "low", "close"))
        days = [_day(int(t)) for t in closed.open_time]
        lo_m = pd.Series(l).rolling(PARAMS["exit_n"]).min().shift(1).to_numpy()
        for j, d in enumerate(days):
            if d < pos["entry_day"] or (pos.get("checked_until") and d <= pos["checked_until"]):
                continue
            if l[j] <= pos["stop"]:
                events["closed"].append(_close_trade(st, s, pos, min(pos["stop"], o[j]), "STOP", d))
                del st["open"][s]
                break
            held = (pd.Timestamp(d) - pd.Timestamp(pos["entry_day"])).days
            exit_now = c[j] < lo_m[j] or held + 1 >= PARAMS["max_hold_days"]
            if exit_now:
                nxt = float(df.open.iloc[j + 1])            # افتتاح اليوم اللي بعده
                why = "TRAIL" if c[j] < lo_m[j] else "MAX_HOLD"
                events["closed"].append(_close_trade(st, s, pos, nxt, why, _day(int(df.open_time.iloc[j + 1]))))
                del st["open"][s]
                break
            pos["checked_until"] = d

    # 2) الدخول (بس في أول ENTRY_WINDOW_HOURS ساعات من اليوم الجديد — غير كده السعر مش واقعي)
    late = (now.hour + now.minute / 60) > ENTRY_WINDOW_HOURS and entry_price == "current"
    if late:
        events["late"] = True
    if regime and not late:
        cands = []
        for s, df in data.items():
            if s in st["open"]:
                continue
            closed = df.iloc[:-1]
            h, l, c = (closed[k].to_numpy() for k in ("high", "low", "close"))
            if len(c) < PARAMS["trend_sma"] + 1:
                continue
            hi_n = h[-PARAMS["entry_n"] - 1:-1].max()
            ma = c[-PARAMS["trend_sma"]:].mean()
            atr = _atr(h, l, c)[-1]
            if c[-1] > hi_n and c[-1] > ma and atr == atr and atr > 0:
                px = float(df.close.iloc[-1]) if entry_price == "current" else float(df.open.iloc[-1])
                cands.append((c[-1] / hi_n - 1, s, px, float(atr)))
        cands.sort(reverse=True)
        for _, s, entry, atr in cands:
            if len(st["open"]) >= MAX_POSITIONS:
                events["skipped"].append(s)
                st["skipped"] = st.get("skipped", 0) + 1
                continue
            stop = entry - PARAMS["atr_mult"] * atr
            if stop <= 0:
                continue
            sl_pct = (entry - stop) / entry * 100
            eq = st.get("equity", START_CAPITAL)
            used = sum(p.get("notional", 0) for p in st["open"].values())
            notional = max(0.0, min(eq * RISK_PER_TRADE / (sl_pct / 100), eq * MAX_POSITION_PCT, eq - used))
            st["open"][s] = {"entry_day": today, "entry": entry, "stop": stop, "sl_pct": round(sl_pct, 2),
                             "notional": round(notional, 2)}
            events["opened"].append((s, st["open"][s]))
    st["last_day"] = today
    st["last_regime"] = regime
    _save(st)
    return _messages(st, events, regime, data, now)


def _open_pnl(st, data):
    out = []
    for s, p in st["open"].items():
        df = data.get(s)
        px = float(df.close.iloc[-1]) if df is not None else None
        out.append((s, p, None if px is None else (px / p["entry"] - 1) * 100))
    return out


def _messages(st, ev, regime, data, now, tag="🤖 [V5.1]"):
    msgs = []
    if ev["opened"] or ev["closed"]:
        L = [f"{tag} 📈 الاتجاه اليومي — 🧪 اختبار على الورق ({now:%Y-%m-%d})", ""]
        for s, p in ev["opened"]:
            L.append(f"🟢 دخول تجريبي: {s}")
            L.append(f"   الدخول (السعر وقت الرسالة): {_fmt(p['entry'])} | الوقف: {_fmt(p['stop'])} (−{p['sl_pct']:.1f}%)")
            L.append("   السبب: إغلاق أمس فوق أعلى سعر في 20 يوم + فوق متوسط 50 يوم + BTC فوق متوسطه")
            L.append("   الخروج: لو قفل يوم تحت أقل سعر في 10 أيام، أو لمس الوقف")
        for r in ev["closed"]:
            icon = "✅" if r["net_pct"] > 0 else "❌"
            why = {"STOP": "لمس الوقف", "TRAIL": "كسر قاع 10 أيام", "MAX_HOLD": "أقصى مدة"}.get(r["reason"], r["reason"])
            L.append(f"{icon} خروج تجريبي: {r['symbol']} — {why} بعد {r['days']} يوم: {r['net_pct']:+.2f}% بعد الرسوم")
        if ev["skipped"]:
            L.append(f"⏸ اتجاهلت {len(ev['skipped'])} إشارة (المحفظة التجريبية مليانة {MAX_POSITIONS} صفقات)")
        L += ["", f"المحفظة التجريبية: {st['equity']:.2f}$ (بدأت {START_CAPITAL:.0f}$) | مفتوح: {len(st['open'])}",
              "🧪 اختبار على الورق — مش توصية شراء"]
        msgs.append("\n".join(L))
    # ملخص أسبوعي يوم السبت
    week = now.strftime("%G-W%V")
    if now.weekday() == 5 and st.get("last_weekly") != week:
        msgs.append(weekly_summary(st, data, tag))
        st["last_weekly"] = week
        _save(st)
    return msgs


def weekly_summary(st, data, tag="🤖 [V5.1]"):
    cl = pd.DataFrame(st["closed"])
    L = [f"{tag} 🗓 ملخص أسبوعي — الاتجاه اليومي 🧪 (من {st.get('started')})", ""]
    if len(cl):
        p = cl.net_pct
        gp, gl = p[p > 0].sum(), -p[p < 0].sum()
        L.append(f"صفقات اتقفلت: {len(cl)} | كسبانة: {(p > 0).sum()} ({(p > 0).mean() * 100:.0f}%)")
        L.append(f"متوسط الصفقة: {p.mean():+.2f}% | الأرباح÷الخسائر: {gp / gl:.2f}" if gl > 0 else f"متوسط الصفقة: {p.mean():+.2f}%")
        L.append(f"أفضل: {p.max():+.1f}% | أسوأ: {p.min():+.1f}%")
    else:
        L.append("لسه مفيش صفقات اتقفلت.")
    op = _open_pnl(st, data)
    if op:
        L.append("")
        L.append(f"مفتوح دلوقتي ({len(op)}):")
        for s, p, u in op:
            L.append(f"• {s}: من {p['entry_day']} — " + ("—" if u is None else f"{u:+.1f}%"))
    L += ["", f"المحفظة التجريبية: {st.get('equity', START_CAPITAL):.2f}$ من {START_CAPITAL:.0f}$",
          "الحكم بعد 2–3 شهور: لازم يكون موجب ومش معتمد على صفقة أو اتنين.", "🧪 اختبار على الورق — مش توصية شراء"]
    return "\n".join(L)


def maybe_run(scanner, send_fn, tag="🤖 [V5.1]"):
    """يتنادى كل دورة مسح. بيشتغل فعلًا مرة في اليوم بس."""
    if os.getenv("TREND_PAPER", "1") == "0":
        return []
    st = _load()
    now = datetime.now(timezone.utc)
    if st.get("last_day") == now.strftime("%Y-%m-%d") or (now.hour == 0 and now.minute < RUN_AFTER_MIN):
        return []
    msgs = process(scanner, now)
    for m in msgs:
        send_fn(m)
    return msgs
