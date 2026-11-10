"""البوت بيحسب أداءه بنفسه من الإشارات اللي بعتها فعلًا (early_pump_signal_results_v5.csv).

• النجاح = الصفقة اتقفلت بصافي ربح > 0 بعد الرسوم (حسب خطة الأهداف الجزئية في trade_manager).
• التقارير: يومي (كل يوم)، أسبوعي (الأسبوع من السبت للجمعة)، شهري — بتوقيت القاهرة.
• "لو دخلت بـ 100$ في كل إشارة" = مجموع النسب المئوية (حجم ثابت، بدون تراكم).
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config as C

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Africa/Cairo")
except Exception:  # noqa: BLE001
    TZ = timezone(timedelta(hours=2))

MIN_FOR_RATE = 5          # أقل عدد صفقات مقفولة عشان نعرض نسبة نجاح
STAKE_USD = 100.0
COST_PCT = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT)

EXIT_AR = {
    "TP3": "كل الأهداف", "LOCK_TP1": "TP2 ثم رجوع لـ TP1", "BE": "TP1 ثم تعادل",
    "SL": "وقف خسارة", "TIMEOUT": "انتهاء المدة", "TP1": "TP1", "LEGACY": "قديم",
}


def read_results(path=None):
    try:
        return pd.read_csv(path or C.TRACK_FILE, low_memory=False)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def closed_trades(df):
    """صفقات مقفولة: (symbol, lane, horizon, pnl_pct صافي, closed_at UTC, exit_reason, opened_at)."""
    if df is None or not len(df):
        return pd.DataFrame(columns=["symbol", "lane", "horizon", "pnl_pct", "closed_at", "exit_reason", "opened_at"])
    d = df.copy()

    def col(name):
        return d[name] if name in d.columns else pd.Series([None] * len(d), index=d.index, dtype=object)

    opened = pd.to_datetime(col("final_time_utc"), utc=True, errors="coerce")
    new = col("trade_status").astype(str) == "CLOSED"
    pnl = pd.to_numeric(col("trade_pnl_pct"), errors="coerce")
    closed_at = pd.to_datetime(col("closed_time_utc"), utc=True, errors="coerce")
    reason = col("exit_reason").astype(object)
    # إشارات قبل نظام إدارة الصفقة: نستخدم نتيجة TP1/SL/TIMEOUT
    legacy = (~new) & col("outcome").isin(["TP1", "SL", "TIMEOUT"])
    if legacy.any():
        pnl = pnl.where(~legacy, pd.to_numeric(col("exit_pct"), errors="coerce") - COST_PCT)
        mins = pd.to_numeric(col("outcome_min"), errors="coerce").fillna(0)
        closed_at = closed_at.where(~legacy, opened + pd.to_timedelta(mins, unit="m"))
        reason = reason.where(~legacy, col("outcome"))
    keep = (new | legacy) & pnl.notna() & closed_at.notna()
    out = pd.DataFrame({
        "symbol": col("symbol"), "lane": col("lane"), "horizon": col("horizon"),
        "pnl_pct": pnl, "closed_at": closed_at, "exit_reason": reason, "opened_at": opened,
    })[keep]
    return out.sort_values("closed_at").reset_index(drop=True)


def summarize(t):
    n = int(len(t))
    if not n:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "total_pct": 0.0,
                "avg_pct": None, "best": None, "worst": None, "usd": 0.0}
    p = t["pnl_pct"].astype(float)
    return {
        "trades": n, "wins": int((p > 0).sum()), "losses": int((p <= 0).sum()),
        "win_rate": float((p > 0).mean()), "total_pct": float(p.sum()), "avg_pct": float(p.mean()),
        "best": float(p.max()), "worst": float(p.min()), "usd": float(p.sum() / 100.0 * STAKE_USD),
    }


def _since(t, days):
    if not len(t):
        return t
    return t[t["closed_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]


def track_record_lines(df, lane=None, horizon=None):
    """سطور "سجل البوت" اللي بتظهر في كل إشارة جديدة."""
    t = closed_trades(df)
    out = []
    all_s = summarize(t)
    if all_s["trades"] < MIN_FOR_RATE:
        out.append(f"📒 سجل إشارات البوت: لسه بيجمع بيانات ({all_s['trades']} صفقة مقفولة) — "
                   f"نسبة النجاح هتظهر بعد {MIN_FOR_RATE} صفقات")
        return out
    s30 = summarize(_since(t, 30))
    base = s30 if s30["trades"] >= MIN_FOR_RATE else all_s
    period = "آخر 30 يوم" if base is s30 else "من البداية"
    out.append(f"📒 سجل إشارات البوت ({period}): {base['wins']} ناجحة من {base['trades']} "
               f"= {base['win_rate'] * 100:.0f}% | صافي {base['total_pct']:+.1f}%")
    if horizon:
        h = summarize(t[t["horizon"].astype(str) == str(horizon)])
        if h["trades"] >= MIN_FOR_RATE:
            out.append(f"   نفس النوع: {h['wins']}/{h['trades']} = {h['win_rate'] * 100:.0f}% | متوسط {h['avg_pct']:+.2f}%")
    if lane:
        l = summarize(t[t["lane"].astype(str) == str(lane)])
        if l["trades"] >= MIN_FOR_RATE:
            out.append(f"   نفس المسار: {l['wins']}/{l['trades']} = {l['win_rate'] * 100:.0f}% | متوسط {l['avg_pct']:+.2f}%")
    return out


# ------------------------------------------------------------------
# تقارير يومي / أسبوعي / شهري
# ------------------------------------------------------------------
def _local(ts):
    return ts.tz_convert(TZ) if hasattr(ts, "tz_convert") else ts.astimezone(TZ)


def period_bounds(kind, now_local):
    """حدود الفترة اللي خلصت (بتوقيت القاهرة) → (start, end, label)."""
    today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if kind == "daily":
        start = today - timedelta(days=1)
        return start, today, f"تقرير يوم {start:%Y-%m-%d}"
    if kind == "weekly":
        # الأسبوع بيبدأ السبت (weekday=5)
        this_week = today - timedelta(days=(today.weekday() - 5) % 7)
        start = this_week - timedelta(days=7)
        return start, this_week, f"تقرير الأسبوع {start:%Y-%m-%d} → {this_week - timedelta(days=1):%Y-%m-%d}"
    this_month = today.replace(day=1)
    start = (this_month - timedelta(days=1)).replace(day=1)
    return start, this_month, f"تقرير شهر {start:%Y-%m}"


def period_key(kind, now_local):
    return period_bounds(kind, now_local)[1].strftime("%Y-%m-%d")


def build_report(df, kind, now_utc=None, tag="🤖 [V5.1]"):
    now_utc = now_utc or datetime.now(timezone.utc)
    start, end, label = period_bounds(kind, now_utc.astimezone(TZ))
    t = closed_trades(df)
    if len(t):
        cl = t["closed_at"].dt.tz_convert(TZ)
        t = t[(cl >= pd.Timestamp(start)) & (cl < pd.Timestamp(end))]
    s = summarize(t)
    opened = 0
    if df is not None and len(df) and "final_time_utc" in df:
        op = pd.to_datetime(df["final_time_utc"], utc=True, errors="coerce").dt.tz_convert(TZ)
        opened = int(((op >= pd.Timestamp(start)) & (op < pd.Timestamp(end))).sum())
    still_open = int((df.get("trade_status", pd.Series(dtype=object)).astype(str) == "OPEN").sum()) \
        if df is not None and len(df) else 0
    icon = {"daily": "📅", "weekly": "🗓", "monthly": "📆"}[kind]
    lines = [f"{tag} {icon} {label}" + (" — 🧪 إشارات تجريبية" if C.PAPER_MODE else ""), ""]
    lines.append(f"إشارات جديدة: {opened} | صفقات اتقفلت: {s['trades']} | لسه مفتوحة الآن: {still_open}")
    if s["trades"]:
        lines.append(f"✅ ناجحة: {s['wins']} | ❌ خاسرة: {s['losses']} | نسبة النجاح: {s['win_rate'] * 100:.0f}%")
        sign = "🟢 ربح" if s["total_pct"] > 0 else ("🔴 خسارة" if s["total_pct"] < 0 else "⚪ تعادل")
        lines.append(f"{sign} صافي: {s['total_pct']:+.2f}%  (متوسط الصفقة {s['avg_pct']:+.2f}%)")
        lines.append(f"💵 لو كان حد دخل بـ {STAKE_USD:.0f}$ في كل إشارة: {s['usd']:+.2f}$")
        lines.append(f"أفضل صفقة: {s['best']:+.2f}% | أسوأ صفقة: {s['worst']:+.2f}%")
        if kind != "daily":
            by = t.groupby(t["horizon"].fillna("-").astype(str))["pnl_pct"].agg(["count", "sum"])
            parts = [f"{k}: {int(r['count'])} صفقة {r['sum']:+.1f}%" for k, r in by.iterrows()]
            lines.append("حسب النوع: " + " | ".join(parts))
        if s["trades"] <= 12:
            lines.append("")
            for r in t.itertuples(index=False):
                lines.append(f"• {r.symbol}: {r.pnl_pct:+.2f}% ({EXIT_AR.get(str(r.exit_reason), r.exit_reason)})")
    else:
        lines.append("مفيش صفقات اتقفلت في الفترة دي.")
    lines += ["", "النسب بعد الرسوم والانزلاق، على أساس خطة الأهداف المكتوبة في كل إشارة."]
    return "\n".join(lines), s


def maybe_send_reports(meta, send_fn, df_loader=read_results, now_utc=None, tag="🤖 [V5.1]"):
    """يتنادى كل دورة. يبعت تقرير الفترة اللي خلصت مرة واحدة بس. أول تشغيل: يسجّل بس."""
    now_utc = now_utc or datetime.now(timezone.utc)
    now_local = now_utc.astimezone(TZ)
    sent = []
    df = None
    for kind in ("daily", "weekly", "monthly"):
        key = period_key(kind, now_local)
        mk = f"report_{kind}"
        if meta.get(mk) is None:
            meta[mk] = key          # أول تشغيل: مانبعتش تقرير لفترة قبل ما البوت يشتغل
            continue
        if meta[mk] == key:
            continue
        if df is None:
            df = df_loader()
        msg, _ = build_report(df, kind, now_utc, tag)
        send_fn(msg)
        meta[mk] = key
        sent.append(kind)
    return sent


def period_tables(df):
    """للـ dashboard: جداول يومي/أسبوعي/شهري (بتوقيت القاهرة)."""
    t = closed_trades(df)
    if not len(t):
        return {"daily": [], "weekly": [], "monthly": [], "all": summarize(t)}
    loc = t["closed_at"].dt.tz_convert(TZ)
    day = loc.dt.normalize()
    keys = {
        "daily": day.dt.strftime("%Y-%m-%d"),
        "weekly": (day - pd.to_timedelta((day.dt.weekday - 5) % 7, unit="D")).dt.strftime("%Y-%m-%d"),
        "monthly": loc.dt.strftime("%Y-%m"),
    }
    out = {"all": summarize(t)}
    for k, key in keys.items():
        rows = []
        for period, g in t.groupby(key):
            s = summarize(g)
            rows.append({"period": period, "trades": s["trades"], "wins": s["wins"],
                         "win_rate": s["win_rate"], "total_pct": round(s["total_pct"], 3),
                         "usd": round(s["usd"], 2)})
        out[k] = sorted(rows, key=lambda r: r["period"], reverse=True)[:60]
    return out
