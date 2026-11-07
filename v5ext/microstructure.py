"""(1) Order Book Imbalance  (2) Open Interest  (3) Whale Transactions  + Smart Money Score.

كل دالة بترجع dict فيه قيم أو None. None = "البيانات مش متاحة" ولا تُعتبر أبدًا إشارة صعود.
الدوال النقية (compute_*) منفصلة عن جلب البيانات عشان تتختبر بدون إنترنت.
"""
import logging
import math
import statistics
import time

from . import config as C
from .binance_data import futures_get, spot_get

log = logging.getLogger("v5ext.micro")
_cache = {}


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


# ==================================================================
# 1) Order Book Imbalance
# ==================================================================
def compute_orderbook_features(depth, bands_pct=C.OBI_BANDS_PCT, wall_mult=C.OBI_WALL_MULT):
    """depth = رد /api/v3/depth.

    - obi_<band>: (bids - asks) / (bids + asks) بالقيمة الدولارية داخل ±band% من الـ mid
    - obi_weighted: كل مستوى وزنه بيقل أُسّيًا مع البعد عن الـ mid (أصعب في الـ spoofing البعيد)
    - ask_wall_ratio / bid_wall_ratio: أكبر مستوى ÷ وسيط المستويات (جدار بيع/شراء)
    - spread_pct
    """
    bids = [(float(p), float(q)) for p, q in depth.get("bids", []) if float(q) > 0]
    asks = [(float(p), float(q)) for p, q in depth.get("asks", []) if float(q) > 0]
    out = {"obi_weighted": None, "spread_pct": None, "ask_wall_ratio": None,
           "bid_wall_ratio": None, "ask_wall_near": False}
    for b in bands_pct:
        out[f"obi_{str(b).replace('.', '_')}pct"] = None
    if not bids or not asks:
        return out
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return out
    mid = (best_bid + best_ask) / 2.0
    out["spread_pct"] = (best_ask - best_bid) / mid * 100.0

    for b in bands_pct:
        lo, hi = mid * (1 - b / 100.0), mid * (1 + b / 100.0)
        bv = sum(p * q for p, q in bids if p >= lo)
        av = sum(p * q for p, q in asks if p <= hi)
        key = f"obi_{str(b).replace('.', '_')}pct"
        out[key] = (bv - av) / (bv + av) if bv + av > 0 else None

    # وزن أُسّي: مستوى على بعد 1% وزنه e^-1
    def wsum(levels):
        return sum(p * q * math.exp(-abs(p / mid - 1.0) * 100.0) for p, q in levels)
    wb, wa = wsum(bids), wsum(asks)
    out["obi_weighted"] = (wb - wa) / (wb + wa) if wb + wa > 0 else None

    bid_notional = [p * q for p, q in bids]
    ask_notional = [p * q for p, q in asks]
    if len(ask_notional) >= 5:
        med = statistics.median(ask_notional) or 1e-12
        mx = max(ask_notional)
        out["ask_wall_ratio"] = mx / med
        idx = ask_notional.index(mx)
        # جدار بيع قريب (أقل من 1.5% فوق السعر) = مقاومة فورية
        out["ask_wall_near"] = bool(out["ask_wall_ratio"] >= wall_mult
                                    and asks[idx][0] <= mid * 1.015)
    if len(bid_notional) >= 5:
        med = statistics.median(bid_notional) or 1e-12
        out["bid_wall_ratio"] = max(bid_notional) / med
    return out


def fetch_orderbook(symbol):
    depth = spot_get("/api/v3/depth", {"symbol": symbol, "limit": C.OBI_DEPTH_LIMIT}, retries=1)
    return compute_orderbook_features(depth)


# ==================================================================
# 2) Open Interest
# ==================================================================
def compute_oi_features(rows, price_change_1h=None):
    """rows = /futures/data/openInterestHist (period=5m) مرتبة من الأقدم للأحدث."""
    out = {"oi_change_15m": None, "oi_change_1h": None, "oi_regime": None, "oi_value_usd": None}
    vals = []
    for r in rows or []:
        try:
            vals.append(float(r.get("sumOpenInterestValue") or 0))
        except (TypeError, ValueError, AttributeError):
            continue
    vals = [v for v in vals if v > 0]
    if len(vals) < 2:
        return out
    out["oi_value_usd"] = vals[-1]
    if len(vals) >= 4:
        out["oi_change_15m"] = (vals[-1] / vals[-4] - 1.0) * 100.0
    else:
        out["oi_change_15m"] = (vals[-1] / vals[0] - 1.0) * 100.0
    out["oi_change_1h"] = (vals[-1] / vals[0] - 1.0) * 100.0
    if price_change_1h is not None:
        oi = out["oi_change_1h"]
        if price_change_1h > 0 and oi > 1.0:
            out["oi_regime"] = "NEW_LONGS"         # سعر ↑ + OI ↑ = فلوس جديدة داخلة
        elif price_change_1h > 0 and oi < -1.0:
            out["oi_regime"] = "SHORT_COVERING"    # سعر ↑ + OI ↓ = تغطية شورت (أقصر عمرًا)
        elif price_change_1h < 0 and oi > 1.0:
            out["oi_regime"] = "NEW_SHORTS"
        else:
            out["oi_regime"] = "NEUTRAL"
    return out


def fetch_open_interest(symbol, price_change_1h=None):
    rows = futures_get("/futures/data/openInterestHist",
                       {"symbol": symbol, "period": C.OI_PERIOD, "limit": C.OI_POINTS})
    if not isinstance(rows, list):
        rows = []
    return compute_oi_features(rows, price_change_1h)


# ==================================================================
# 3) Whale Transactions (spot aggTrades)
# ==================================================================
def whale_threshold_usd(quote_volume_24h):
    adaptive = (quote_volume_24h or 0) * C.WHALE_QV_FRACTION
    return _clip(adaptive, C.WHALE_MIN_USD, C.WHALE_MAX_USD)


def compute_whale_features(trades, threshold_usd, now_ms=None, window_min=C.WHALE_WINDOW_MIN):
    """trades = /api/v3/aggTrades. m=True يعني البائع هو الـ taker (بيع عدواني)."""
    out = {"whale_threshold_usd": threshold_usd, "whale_buy_usd": None, "whale_sell_usd": None,
           "whale_net_ratio": None, "whale_count": None, "whale_buy_share": None,
           "whale_window_min": None}
    if not trades:
        return out
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - window_min * 60_000
    buy = sell = total = 0.0
    count = 0
    first_t = None
    for t in trades:
        try:
            ts = int(t["T"])
            if ts < since:
                continue
            notional = float(t["p"]) * float(t["q"])
        except (KeyError, TypeError, ValueError):
            continue
        first_t = ts if first_t is None else min(first_t, ts)
        total += notional
        if notional >= threshold_usd:
            count += 1
            if t.get("m"):
                sell += notional
            else:
                buy += notional
    if first_t is None:
        return out
    out.update({
        "whale_buy_usd": buy, "whale_sell_usd": sell, "whale_count": count,
        "whale_net_ratio": (buy - sell) / (buy + sell) if buy + sell > 0 else 0.0,
        "whale_buy_share": buy / total if total > 0 else 0.0,
        "whale_window_min": round((now_ms - first_t) / 60_000, 1),
    })
    return out


def fetch_whales(symbol, quote_volume_24h):
    trades = spot_get("/api/v3/aggTrades", {"symbol": symbol, "limit": 1000}, retries=1)
    return compute_whale_features(trades, whale_threshold_usd(quote_volume_24h))


# ==================================================================
# Smart Money Score (0..100, 50 = محايد)
# ==================================================================
def smart_money_score(x):
    """متوسط مكونات محدودة؛ المكونات الناقصة بتتشال بدل ما تتحسب صعودية."""
    comp = [
        _clip((float(x.get("buy_pressure", 0.5)) - 0.45) / 0.25 * 100.0, 0, 100),
        _clip(float(x.get("volume_15m", 1.0)) / 8.0 * 100.0, 0, 100),
        _clip(float(x.get("acceleration", 1.0)) / 3.0 * 100.0, 0, 100),
    ]
    obi = x.get("obi_1_0pct")
    if obi is None:
        obi = x.get("orderbook_imbalance")
    if obi is not None:
        comp.append(_clip((obi + 0.2) / 0.6 * 100.0, 0, 100))
    if x.get("obi_weighted") is not None:
        comp.append(_clip((x["obi_weighted"] + 0.2) / 0.6 * 100.0, 0, 100))
    oi = x.get("oi_change_1h")
    if oi is not None:
        comp.append(_clip((oi + 1.0) / 6.0 * 100.0, 0, 100))
    wn = x.get("whale_net_ratio")
    if wn is not None and (x.get("whale_count") or 0) > 0:
        comp.append(_clip((wn + 1.0) / 2.0 * 100.0, 0, 100))
    score = sum(comp) / len(comp)
    if x.get("ask_wall_near"):
        score -= 5
    return round(_clip(score, 0, 100), 2)


def fetch_all(symbol, price_change_1h=None, quote_volume_24h=0.0):
    """يجيب الثلاثة مع كاش MICRO_CACHE_TTL ثانية. أي جزء يفشل يرجع None."""
    hit = _cache.get(symbol)
    if hit and time.time() - hit["_ts"] < C.MICRO_CACHE_TTL:
        return {k: v for k, v in hit.items() if k != "_ts"}
    data = {}
    try:
        data.update(fetch_orderbook(symbol))
    except Exception as e:  # noqa: BLE001
        log.debug("OBI unavailable %s: %s", symbol, e)
    try:
        data.update(fetch_open_interest(symbol, price_change_1h))
    except Exception as e:  # noqa: BLE001
        log.debug("OI unavailable %s: %s", symbol, e)
    if C.WHALE_ENABLED:
        try:
            data.update(fetch_whales(symbol, quote_volume_24h))
        except Exception as e:  # noqa: BLE001
            log.debug("whales unavailable %s: %s", symbol, e)
    # توافق خلفي مع أعمدة V5 القديمة
    data["orderbook_imbalance"] = data.get("obi_1_0pct")
    data["oi_change_pct"] = data.get("oi_change_15m")
    _cache[symbol] = dict(data, _ts=time.time())
    return data
