"""HTTP + تحميل الشموع التاريخية من Binance مع كاش محلي تراكمي."""
import logging
import os
import time

import pandas as pd
import requests

from . import config as C

log = logging.getLogger("v5ext.data")

_session = requests.Session()
_session.headers.update({"User-Agent": "Binance-Early-Pump-Scanner-V5.1/research"})

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


def spot_get(path, params=None, retries=3, timeout=10):
    last = None
    for attempt in range(retries):
        for base in C.SPOT_BASE_URLS:
            try:
                r = _session.get(base + path, params=params, timeout=timeout)
                if r.status_code == 429 or r.status_code == 418:
                    time.sleep(2 + attempt * 3)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def futures_get(path, params=None, timeout=8):
    r = _session.get(C.FUTURES_BASE_URL + path, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_get_json(url, params=None, timeout=10):
    r = _session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------
# Historical klines
# ------------------------------------------------------------------
def _cache_path(symbol, interval):
    return os.path.join(C.KLINE_CACHE_DIR, f"{symbol}_{interval}.csv.gz")


def fetch_klines_range(symbol, interval, start_ms, end_ms, pause=0.05):
    """يجيب كل الشموع المقفولة بين start_ms و end_ms (صفحات 1000)."""
    step = INTERVAL_MS[interval]
    rows, cur = [], int(start_ms)
    while cur < end_ms:
        batch = spot_get("/api/v3/klines", {"symbol": symbol, "interval": interval,
                                            "startTime": cur, "endTime": int(end_ms), "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + step
        if nxt <= cur or len(batch) < 1000:
            break
        cur = nxt
        time.sleep(pause)
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return normalize_klines(df)


def normalize_klines(df):
    df = df.copy()
    for c in ("open_time", "close_time", "trades"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["ignore"] = 0
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[KLINE_COLUMNS]


def load_klines(symbol, interval, days, now_ms=None, use_cache=True):
    """يرجّع DataFrame للشموع المقفولة لآخر `days` يوم، ويحدّث الكاش بالجديد بس."""
    now_ms = now_ms or int(time.time() * 1000)
    step = INTERVAL_MS[interval]
    end_ms = (now_ms // step) * step - 1          # آخر شمعة مقفولة
    start_ms = end_ms - int(days * 86_400_000)
    path = _cache_path(symbol, interval)
    cached = None
    if use_cache and os.path.exists(path):
        try:
            cached = normalize_klines(pd.read_csv(path))
        except Exception as e:  # noqa: BLE001
            log.warning("cache read failed %s: %s", path, e)
    parts = []
    if cached is not None and len(cached):
        have_start, have_end = int(cached.open_time.iloc[0]), int(cached.open_time.iloc[-1])
        if start_ms < have_start:
            parts.append(fetch_klines_range(symbol, interval, start_ms, have_start - 1))
        parts.append(cached)
        if have_end + step <= end_ms:
            parts.append(fetch_klines_range(symbol, interval, have_end + step, end_ms))
    else:
        parts.append(fetch_klines_range(symbol, interval, start_ms, end_ms))
    df = normalize_klines(pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)) \
        if any(len(p) for p in parts if p is not None) else pd.DataFrame(columns=KLINE_COLUMNS)
    df = df[df.close_time <= end_ms]
    if use_cache and len(df):
        os.makedirs(C.KLINE_CACHE_DIR, exist_ok=True)
        df.to_csv(path, index=False)
    return df[df.open_time >= start_ms].reset_index(drop=True)


def top_usdt_symbols(n=60, min_quote_volume=2_000_000, exclude_bases=(), allowed_bases=None):
    """أعلى أزواج USDT حسب حجم 24h (بدون stablecoins)."""
    info = spot_get("/api/v3/exchangeInfo", {"permissions": "SPOT"})
    tradable = {s["symbol"] for s in info.get("symbols", [])
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}
    tickers = spot_get("/api/v3/ticker/24hr")
    out = []
    for t in tickers:
        sym = t.get("symbol", "")
        if sym not in tradable:
            continue
        base = sym[:-4]
        if base in exclude_bases:
            continue
        if allowed_bases is not None and base not in allowed_bases:
            continue
        qv = float(t.get("quoteVolume") or 0)
        if qv >= min_quote_volume:
            out.append((qv, sym))
    out.sort(reverse=True)
    return [s for _, s in out[:n]]
