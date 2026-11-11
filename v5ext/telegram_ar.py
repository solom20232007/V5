"""رسائل تيليجرام بالعربي (المصطلحات الفنية بين قوسين).

مبدأ: مفيش أي رقم في الرسالة مش متحقق منه.
• نسبة النجاح المعروضة = نتيجة إشارات البوت الفعلية (مش تقدير نظري). لو الصفقات المقفولة أقل من 5
  الرسالة بتقول "لسه بيجمع بيانات".
• تقدير النموذج بيظهر بس لو النموذج معاير (Calibrated) وعدّى اختبار خارج العينة، ومكتوب جنبه "تقديري".
"""
from datetime import datetime

from . import config as C
from .trade_manager import HORIZONS, classify_horizon

PAPER_HEADER = ("🧪 إشارة تجريبية — للمتابعة فقط، مش توصية شراء\n"
                "(الاختبار على آخر 90 يوم طلع خسارة، فالبوت بيسجّل الإشارات لحد ما يثبت إنه بيكسب)")

LANE_AR = {
    "STRICT": "صارم (Strict)",
    "EXPLOSIVE": "انفجاري مبكر (Explosive)",
    "PERSISTENT": "زخم مستمر (Persistent)",
    "REVERSAL": "انعكاس (Reversal)",
    "SQUEEZE": "عصر الشورت (Short Squeeze)",
}


def _f(v, default=None):
    try:
        if v is None:
            return default
        v = float(v)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _p(v):
    return f"{float(v):.8g}"


def orderbook_lines(x):
    obi = _f(x.get("obi_1_0pct"))
    if obi is None:
        obi = _f(x.get("orderbook_imbalance"))
    if obi is None:
        return ["⚖️ ميزان الأوامر (Order Book Imbalance): غير متاح الآن"]
    buy = (1 + obi) / 2 * 100
    if obi >= 0.2:
        verdict = "ضغط شراء واضح 🟢"
    elif obi >= 0.05:
        verdict = "ميل للشراء"
    elif obi > -0.05:
        verdict = "متوازن"
    elif obi > -0.2:
        verdict = "ميل للبيع"
    else:
        verdict = "ضغط بيع واضح 🔴"
    out = ["⚖️ ميزان الأوامر (Order Book Imbalance) ±1%:",
           f"   شراء {buy:.0f}% | بيع {100 - buy:.0f}%  ← {verdict}"]
    bands = []
    for key, lbl in (("obi_0_5pct", "±0.5%"), ("obi_2_0pct", "±2%")):
        v = _f(x.get(key))
        if v is not None:
            bands.append(f"{lbl}: شراء {(1 + v) / 2 * 100:.0f}%")
    if bands:
        out.append("   " + " | ".join(bands))
    out.append("   (لقطة لحظية من دفتر الأوامر — ممكن تتغير بسرعة)")
    return out


def reasons(x, confirmations, cross_label):
    r = []
    v15, v1h = _f(x.get("volume_15m"), 0), _f(x.get("volume_1h"), 0)
    if v15 >= 2:
        r.append(f"حجم تداول آخر 15 دقيقة {v15:.1f} ضعف المعتاد")
    if v1h >= 1.7:
        r.append(f"حجم آخر ساعة {v1h:.1f} ضعف المعتاد")
    bp = _f(x.get("buy_pressure"), 0)
    if bp >= 0.55:
        r.append(f"{bp * 100:.0f}% من الصفقات شراء عدواني (Taker Buy)")
    acc = _f(x.get("acceleration"), 0)
    if acc >= 1.3:
        r.append(f"تسارع الحجم {acc:.1f}x مقارنة بالـ 15 دقيقة السابقة")
    br = _f(x.get("breakout"), -99)
    if br >= 0:
        r.append(f"كسر قمة آخر 4 ساعات (Breakout) بنسبة {br:+.2f}%")
    elif br >= -1:
        r.append("السعر ملاصق لقمة آخر 4 ساعات")
    if x.get("trend") in ("UP", "STRONG_UP"):
        r.append("اتجاه صاعد على المتوسطات (EMA 9 > 21" + (" > 50)" if x.get("trend") == "STRONG_UP" else ")"))
    gs = int(_f(x.get("green_streak"), 0))
    if gs >= 2:
        r.append(f"{gs} شموع خضراء متتالية (5 دقايق)")
    obi = _f(x.get("obi_1_0pct"))
    if obi is not None and obi >= 0.15:
        r.append("دفتر الأوامر يميل للشراء (Order Book)")
    wn = _f(x.get("whale_net_ratio"))
    if wn is not None and wn >= 0.2 and (x.get("whale_count") or 0) > 0:
        r.append(f"الحيتان (Whales) صافي شراء: {x.get('whale_count')} صفقة كبيرة")
    if x.get("oi_regime") == "NEW_LONGS":
        r.append(f"سيولة جديدة في العقود (Open Interest {_f(x.get('oi_change_1h'), 0):+.1f}% خلال ساعة)")
    if cross_label == "CONFIRMED":
        r.append("الحركة مؤكدة على منصات تانية (Bybit/OKX)")
    r.append(f"اتأكدت في {confirmations} مسح متتالي + شمعة دقيقة مقفولة إيجابية")
    return r


def warnings(x, cross_label):
    w = []
    if x.get("ask_wall_near"):
        w.append("جدار بيع (Sell Wall) قريب فوق السعر")
    rsi = _f(x.get("rsi14"))
    if rsi is not None and rsi >= 72:
        w.append(f"تشبّع شراء نسبي (RSI {rsi:.0f})")
    if x.get("liquidity") == "LOW":
        w.append("سيولة منخفضة — قلّل حجم الصفقة")
    if _f(x.get("24h"), 0) > 15:
        w.append(f"العملة صاعدة {_f(x.get('24h'), 0):.1f}% خلال 24 ساعة")
    if x.get("oi_regime") == "SHORT_COVERING":
        w.append("الصعود جاي من تغطية شورت (Short Covering) — غالبًا أقصر عمرًا")
    wn = _f(x.get("whale_net_ratio"))
    if wn is not None and wn <= -0.3 and (x.get("whale_count") or 0) > 0:
        w.append("الحيتان صافي بيع")
    if (x.get("fng_label") or "") in ("EXTREME_FEAR", "EXTREME_GREED"):
        w.append(f"مؤشر الخوف والطمع (Fear & Greed) في منطقة متطرفة: {x.get('fng')}")
    if _f(x.get("spread_pct"), 0) > 0.3:
        w.append(f"فرق سعر واسع (Spread {_f(x.get('spread_pct')):.2f}%)")
    if x.get("is_wash"):
        w.append("نمط يشبه التداول الوهمي (Wash Trading)")
    if cross_label == "DIVERGENCE":
        w.append("المنصات التانية مش ماشية مع الحركة (Divergence)")
    return w


def _track_record(x):
    try:
        from .performance import read_results, track_record_lines
        return track_record_lines(read_results(), x.get("lane"), x.get("horizon"))
    except Exception:  # noqa: BLE001
        return []


def _model_line(x):
    """تقدير النموذج — بس لو معاير ومقبول."""
    p = _f(x.get("ml_prob_cal"))
    if p is None:
        return None
    return f"🔬 تقدير النموذج لهذه الإشارة (تقديري، مش ضمان): {p * 100:.0f}% يوصل TP1 قبل الوقف"


def format_monitor_ar(x, confirmations, cross_label=None, tag="🤖 [V5.1]"):
    """تنبيه رصد: حقائق عن الحركة بس — مفيش دخول/وقف/أهداف ولا توقعات."""
    lines = [f"{tag} 👀 رصد: {x['symbol']} بتتحرك بحجم غير عادي", ""]
    lines.append(f"📉 الحركة: 5د {_f(x.get('5m'), 0):+.2f}% | 15د {_f(x.get('15m'), 0):+.2f}% | "
                 f"ساعة {_f(x.get('1h'), 0):+.2f}% | 24س {_f(x.get('24h'), 0):+.2f}%")
    lines.append(f"💲 السعر الآن: {_p(x.get('price') or 0)}")
    lines.append(f"💧 السيولة 24 ساعة: ${_f(x.get('quote_volume'), 0):,.0f} ({x.get('liquidity')})")
    lines += ["", "━━━ 🔎 اللي لفت النظر ━━━"]
    lines += [f"• {t}" for t in reasons(x, confirmations, cross_label)]
    lines.append("")
    lines += orderbook_lines(x)
    if (x.get("whale_count") or 0) > 0:
        lines.append(f"🐋 صفقات كبيرة آخر {_f(x.get('whale_window_min'), 0):.0f} دقيقة: "
                     f"شراء ${_f(x.get('whale_buy_usd'), 0):,.0f} / بيع ${_f(x.get('whale_sell_usd'), 0):,.0f}")
    if x.get("fng") is not None:
        lines.append(f"😨 الخوف والطمع (Fear & Greed): {x.get('fng')}")
    warn = warnings(x, cross_label)
    if warn:
        lines += ["", "━━━ ⚠️ انتبه ━━━"] + [f"• {t}" for t in warn]
    lines += ["", "ℹ️ تنبيه رصد فقط — مش إشارة شراء. الاختبار على 6 شهور: الشراء وقت الحركة دي كان بيخسر بعد الرسوم في المتوسط."]
    return "\n".join(lines)


def format_final_ar(x, confirmations, cross_label=None, cross_detail=None, tag="🤖 [V5.1]"):
    if C.MONITOR_MODE:
        return format_monitor_ar(x, confirmations, cross_label, tag)
    hz = x.get("horizon") or classify_horizon(x)
    hz_name, hz_time, hz_timeout = HORIZONS[hz][:3]
    x["horizon"] = hz
    x.setdefault("timeout_min", hz_timeout)
    r = x.get("risk") or {}

    if C.PAPER_MODE:
        lines = [f"{tag} 🧪 {x['symbol']} — صعود مبكر (متابعة)", PAPER_HEADER, ""]
    else:
        lines = [f"{tag} 🔥 فرصة صعود مبكر — {x['symbol']}", ""]
    lines.append(f"⏱ نوع الفرصة: {hz_name}")
    lines.append(f"⌛ مدة المتابعة: {hz_time} (البوت هيتابع ويبعتلك النتيجة)")
    lines.append(f"🛤 المسار (Lane): {LANE_AR.get(x.get('lane'), x.get('lane') or '-')} | درجة الإشارة (Score): {x.get('score')}/100")
    lines += _track_record(x)
    ml = _model_line(x)
    if ml:
        lines.append(ml)
    lines.append("")

    if r:
        lines += [
            "━━━ 📐 المستويات اللي البوت هيقيس عليها ━━━" if C.PAPER_MODE else "━━━ 💰 خطة الصفقة ━━━",
            f"سعر الدخول (Entry): {_p(r['entry'])}",
            f"وقف الخسارة (Stop Loss): {_p(r['sl'])}  ({-r['sl_pct']:+.2f}%)",
            f"الهدف الأول (TP1): {_p(r['tp1'])}  ({r['tp1_pct']:+.2f}%)",
            f"الهدف الثاني (TP2): {_p(r['tp2'])}  ({r['tp2_pct']:+.2f}%)",
        ]
        if r.get("tp3"):
            lines.append(f"الهدف الثالث (TP3): {_p(r['tp3'])}  ({r['tp3_pct']:+.2f}%)")
        if C.PAPER_MODE:
            lines.append("• البوت بيحسب النتيجة كأن ثلث اتقفل عند كل هدف والوقف اتحرك")
        else:
            lines += [
                "• عند TP1: بيع ثلث الكمية وحرّك الوقف لسعر الدخول",
                "• عند TP2: بيع ثلث تاني وحرّك الوقف لـ TP1",
                "• عند TP3: بيع الباقي",
            ]
            for n in r.get("notes", []):
                lines.append(f"• {n}")
        lines.append("")

    lines.append("━━━ 🔎 سبب ظهور الإشارة ━━━")
    lines += [f"✓ {t}" for t in reasons(x, confirmations, cross_label)]
    lines.append("")
    lines += orderbook_lines(x)
    if x.get("whale_count") == 0:
        lines.append("🐋 الحيتان (Whales): مفيش صفقات كبيرة في آخر ربع ساعة")
    elif x.get("whale_count") is not None:
        lines.append(f"🐋 الحيتان (Whales) آخر {_f(x.get('whale_window_min'), 0):.0f} دقيقة: "
                     f"شراء ${_f(x.get('whale_buy_usd'), 0):,.0f} / بيع ${_f(x.get('whale_sell_usd'), 0):,.0f}")
    if _f(x.get("oi_change_1h")) is not None:
        lines.append(f"📈 العقود المفتوحة (Open Interest): {_f(x.get('oi_change_1h')):+.2f}% خلال ساعة")
    if x.get("fng") is not None:
        lines.append(f"😨 الخوف والطمع (Fear & Greed): {x.get('fng')}")
    lines.append(f"📉 الحركة: 5د {_f(x.get('5m'), 0):+.2f}% | 15د {_f(x.get('15m'), 0):+.2f}% | "
                 f"ساعة {_f(x.get('1h'), 0):+.2f}% | 24س {_f(x.get('24h'), 0):+.2f}%")
    lines.append(f"💧 السيولة 24 ساعة: ${_f(x.get('quote_volume'), 0):,.0f} ({x.get('liquidity')})")

    warn = warnings(x, cross_label)
    if warn:
        lines += ["", "━━━ ⚠️ تحذيرات ━━━"] + [f"• {t}" for t in warn]

    if C.PAPER_MODE:
        lines += ["", "🧪 للمتابعة والتسجيل فقط — مش توصية شراء."]
    else:
        lines += ["", "⚠️ إشارة تحليلية وليست توصية شراء. لا تخاطر بأكثر من 1-2% من رأس مالك في الصفقة."]
    return "\n".join(lines)


# ------------------------------------------------------------------
# تنبيهات المتابعة
# ------------------------------------------------------------------
def _when(ts_ms):
    try:
        from .performance import TZ
        return datetime.fromtimestamp(ts_ms / 1000, tz=TZ).strftime("%H:%M")
    except Exception:  # noqa: BLE001
        return ""


def _dur(minutes):
    m = int(_f(minutes, 0))
    if m < 60:
        return f"{m} دقيقة"
    if m < 1440:
        return f"{m // 60} ساعة و{m % 60} دقيقة"
    return f"{m // 1440} يوم و{(m % 1440) // 60} ساعة"


def format_trade_alert(track, alert, tag="🤖 [V5.1]"):
    from .trade_manager import net_pct
    sym = track["symbol"]
    st = track.get("mgmt") or {}
    ev, price = alert["event"], _f(alert.get("price"), 0)
    entry = _f(track.get("entry_price"), 0)
    move = (price / entry - 1) * 100 if entry else 0
    late = ""
    import time as _t
    if alert.get("ts") and _t.time() * 1000 - alert["ts"] > 10 * 60_000:
        late = f"\n(الحدث حصل الساعة {_when(alert['ts'])} بتوقيت القاهرة)"
    closing = ev in ("SL", "BE", "LOCK_TP1", "TP3", "TIMEOUT")
    net = net_pct(st) if (st.get("closed") and closing) else None
    net_line = f"\n💰 النتيجة الصافية للصفقة: {net:+.2f}% (بعد الرسوم) — بعد {_dur(st.get('close_min'))}" \
        if net is not None else ""
    hz_time = HORIZONS.get(track.get("horizon") or "", ("", "", 0))[1]

    if ev == "TP1":
        body = (f"🎯 {sym}: تحقق الهدف الأول (TP1) عند {_p(price)} ({move:+.2f}%)\n"
                f"👈 بيع ثلث الكمية، وحرّك وقف الخسارة لسعر الدخول {_p(entry)}")
    elif ev == "TP2":
        body = (f"🎯🎯 {sym}: تحقق الهدف الثاني (TP2) عند {_p(price)} ({move:+.2f}%)\n"
                f"👈 بيع ثلث تاني، وحرّك الوقف للهدف الأول {_p(st.get('tp1', 0))}")
    elif ev == "TP3":
        body = f"🏆 {sym}: تحققت كل الأهداف (TP3) عند {_p(price)} ({move:+.2f}%)\n👈 بيع الباقي — الصفقة خلصت"
    elif ev == "WARN":
        body = (f"🚨 تنبيه: إشارة {sym} بتفشل\n"
                f"السعر نزل لـ {_p(price)} ({move:+.2f}%) وقطع نص المسافة لوقف الخسارة.\n"
                f"👈 يُفضّل البيع فورًا لتقليل الخسارة (الوقف الأصلي {_p(track.get('sl', 0))})")
    elif ev == "SL":
        body = (f"❌ {sym}: ضرب وقف الخسارة (Stop Loss) عند {_p(price)} ({move:+.2f}%)\n"
                f"👈 لو لسه معاك الصفقة: بيع فورًا")
    elif ev == "BE":
        body = (f"🔔 {sym}: السعر رجع لسعر الدخول بعد الهدف الأول\n"
                f"👈 اقفل الباقي على التعادل (Break-even)")
    elif ev == "LOCK_TP1":
        body = (f"🔔 {sym}: السعر رجع للهدف الأول بعد الهدف الثاني\n"
                f"👈 اقفل الباقي — الربح محفوظ")
    elif ev == "TIMEOUT":
        body = (f"⏱ {sym}: انتهت مدة المتابعة ({hz_time}) والسعر {_p(price)} ({move:+.2f}%)\n"
                f"👈 يُفضّل قفل الباقي بسعر السوق")
    else:
        body = f"{sym}: {ev} @ {_p(price)}"
    if C.PAPER_MODE:
        body = _paper_body(ev, sym, price, move, st, track, hz_time)
        return f"{tag} 🧪 {body}{late}{net_line}"
    return f"{tag} {body}{late}{net_line}"


def _paper_body(ev, sym, price, move, st, track, hz_time):
    """نفس الأحداث بصيغة متابعة (من غير أوامر بيع/شراء)."""
    return {
        "TP1": f"{sym}: وصل الهدف الأول (TP1) عند {_p(price)} ({move:+.2f}%) — متابعة تجريبية",
        "TP2": f"{sym}: وصل الهدف الثاني (TP2) عند {_p(price)} ({move:+.2f}%) — متابعة تجريبية",
        "TP3": f"{sym}: وصل الهدف الثالث (TP3) عند {_p(price)} ({move:+.2f}%) — الإشارة اتقفلت",
        "WARN": f"{sym}: الإشارة بتفشل — السعر {_p(price)} ({move:+.2f}%) قطع نص المسافة للوقف",
        "SL": f"{sym}: ضربت وقف الخسارة عند {_p(price)} ({move:+.2f}%) — الإشارة اتقفلت",
        "BE": f"{sym}: رجعت لسعر الدخول بعد الهدف الأول — الإشارة اتقفلت",
        "LOCK_TP1": f"{sym}: رجعت للهدف الأول بعد الهدف الثاني — الإشارة اتقفلت",
        "TIMEOUT": f"{sym}: انتهت مدة المتابعة ({hz_time}) والسعر {_p(price)} ({move:+.2f}%)",
    }.get(ev, f"{sym}: {ev} @ {_p(price)}")
