"""(4) Fear & Greed Index (alternative.me) — قيمة حالية + تاريخ يومي للـ backtest والـ ML."""
import json
import logging
import os
import time
from datetime import datetime, timezone

from . import config as C
from .binance_data import http_get_json

log = logging.getLogger("v5ext.fng")


def classify(value):
    if value is None:
        return "UNKNOWN"
    if value <= C.FNG_EXTREME_FEAR:
        return "EXTREME_FEAR"
    if value < 45:
        return "FEAR"
    if value <= 55:
        return "NEUTRAL"
    if value < C.FNG_EXTREME_GREED:
        return "GREED"
    return "EXTREME_GREED"


def _read_cache():
    try:
        with open(C.FNG_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _write_cache(data):
    try:
        os.makedirs(os.path.dirname(C.FNG_CACHE_FILE), exist_ok=True)
        tmp = C.FNG_CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, C.FNG_CACHE_FILE)
    except Exception as e:  # noqa: BLE001
        log.debug("fng cache write failed: %s", e)


def get_current(force=False):
    """{'value': int|None, 'label': str, 'ts': epoch}. لو الـ API واقع نرجع آخر قيمة (لو أحدث من 48 ساعة)."""
    cache = _read_cache()
    cur = cache.get("current") or {}
    if not force and cur and time.time() - cur.get("fetched", 0) < C.FNG_CACHE_TTL:
        return cur
    try:
        data = http_get_json(C.FNG_URL, {"limit": 1, "format": "json"})
        row = (data.get("data") or [{}])[0]
        value = int(row["value"])
        cur = {"value": value, "label": classify(value), "ts": int(row.get("timestamp", time.time())),
               "fetched": time.time()}
        cache["current"] = cur
        _write_cache(cache)
        return cur
    except Exception as e:  # noqa: BLE001
        log.warning("Fear&Greed fetch failed: %s", e)
        if cur and time.time() - cur.get("fetched", 0) < 48 * 3600:
            return cur
        return {"value": None, "label": "UNKNOWN", "ts": None, "fetched": time.time()}


def get_history():
    """{ 'YYYY-MM-DD': value } لكل الأيام المتاحة."""
    try:
        data = http_get_json(C.FNG_URL, {"limit": 0, "format": "json"}, timeout=20)
        hist = {}
        for row in data.get("data", []):
            day = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d")
            hist[day] = int(row["value"])
        cache = _read_cache()
        cache["history"] = hist
        _write_cache(cache)
        return hist
    except Exception as e:  # noqa: BLE001
        log.warning("Fear&Greed history fetch failed: %s", e)
        return _read_cache().get("history", {})


def value_at(history, ts_ms):
    """قيمة اليوم السابق المقفول (عشان مفيش look-ahead: قيمة اليوم بتتنشر بعد بدايته)."""
    if not history:
        return None
    day = datetime.fromtimestamp(ts_ms / 1000 - 86400, tz=timezone.utc).strftime("%Y-%m-%d")
    return history.get(day)


def score_adjustment(value, mode=None):
    """يرجّع (extra_required_score, blocked, reason).

    adjust: في الخوف الشديد أو الطمع الشديد ترفع العتبة المطلوبة FNG_SCORE_PENALTY نقطة.
    block:  يمنع الإشارات الجديدة كليًا في المنطقتين المتطرفتين.
    off:    لا يغير شيئًا (القيمة تتسجل بس).
    """
    mode = (mode or C.FNG_MODE).lower()
    label = classify(value)
    if mode == "off" or value is None or label not in ("EXTREME_FEAR", "EXTREME_GREED"):
        return 0, False, label
    if mode == "block":
        return 0, True, label
    return C.FNG_SCORE_PENALTY, False, label
