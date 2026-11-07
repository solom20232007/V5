"""بيانات صناعية للاختبار بدون إنترنت (random walk + موجات pump مزروعة). مش للتداول."""
import numpy as np
import pandas as pd

from .binance_data import KLINE_COLUMNS, normalize_klines


def make_klines(symbol, days=20, seed=0, start_ms=1_760_000_000_000, base_price=1.0,
                base_qv=40_000.0, pumps_per_day=0.6):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) + seed)
    n = int(days * 288)
    ret = rng.normal(0, 0.0035, n)
    vol_mult = np.ones(n)
    buy_share = np.clip(rng.normal(0.5, 0.04, n), 0.2, 0.8)
    n_pumps = rng.poisson(pumps_per_day * days)
    for _ in range(n_pumps):
        s = int(rng.integers(300, n - 60))
        L = int(rng.integers(6, 18))
        success = rng.random() < 0.5
        drift = rng.uniform(0.003, 0.009)
        ret[s:s + L] += drift
        vol_mult[s:s + L] *= rng.uniform(3, 9)
        buy_share[s:s + L] = np.clip(buy_share[s:s + L] + 0.15, 0, 0.9)
        if success:
            ret[s + L:s + L + 12] += drift * 0.8           # استمرار
        else:
            ret[s + L:s + L + 12] -= drift * 1.3           # انعكاس
    close = base_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[base_price], close[:-1]])
    spread = np.abs(rng.normal(0, 0.002, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    qv = base_qv * vol_mult * np.exp(rng.normal(0, 0.3, n))
    ot = start_ms + np.arange(n) * 300_000
    df = pd.DataFrame({
        "open_time": ot, "open": open_, "high": high, "low": low, "close": close,
        "volume": qv / close, "close_time": ot + 299_999, "quote_volume": qv,
        "trades": (qv / 50).astype(int), "taker_buy_base": qv * buy_share / close,
        "taker_buy_quote": qv * buy_share, "ignore": 0,
    })
    return normalize_klines(df[KLINE_COLUMNS])


def make_fng_history(start_ms, days):
    rng = np.random.default_rng(1)
    v, out = 50.0, {}
    for d in range(-2, days + 2):
        v = float(np.clip(v + rng.normal(0, 6), 5, 95))
        day = pd.to_datetime(start_ms + d * 86_400_000, unit="ms", utc=True).strftime("%Y-%m-%d")
        out[day] = int(v)
    return out
