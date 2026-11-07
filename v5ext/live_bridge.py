"""الجسر بين الماسح الحي والموديولات الجديدة. كل دالة هنا fail-safe."""
import json
import logging
import os

from . import config as C
from . import microstructure, sentiment
from .bayes import BayesianAdjuster
from .ml_model import LiveModel

log = logging.getLogger("v5ext.live")

TRACK_EXTRA_COLUMNS = [
    "ml_prob", "ml_prob_cal", "bayes_prob", "expected_r", "smart_money_score", "final_rank",
    "obi_1_0pct", "obi_weighted", "ask_wall_near", "spread_pct",
    "whale_net_ratio", "whale_count", "whale_buy_usd", "whale_sell_usd",
    "oi_change_15m", "oi_change_1h", "oi_regime", "fng", "fng_label", "btc_15m", "btc_1h",
]

_model = LiveModel()
_bayes = None
_ctx = {"fng": None, "fng_label": "UNKNOWN", "btc_15m": 0.0, "btc_1h": 0.0}


def _get_bayes():
    global _bayes
    if _bayes is None:
        _bayes = BayesianAdjuster()
    return _bayes


def start_cycle(btc_15m, btc_1h):
    """يتنادى مرة في أول كل دورة مسح."""
    _ctx["btc_15m"], _ctx["btc_1h"] = float(btc_15m or 0), float(btc_1h or 0)
    try:
        f = sentiment.get_current()
        _ctx["fng"], _ctx["fng_label"] = f.get("value"), f.get("label", "UNKNOWN")
    except Exception as e:  # noqa: BLE001
        log.debug("fng: %s", e)
    return dict(_ctx)


def context():
    return dict(_ctx)


def attach_context(x):
    """حقول رخيصة لكل نتيجة (مطلوبة كـ features للـ ML)."""
    from datetime import datetime, timezone
    x.setdefault("btc_15m", _ctx["btc_15m"])
    x.setdefault("btc_1h", _ctx["btc_1h"])
    x.setdefault("fng", _ctx["fng"])
    x.setdefault("fng_label", _ctx["fng_label"])
    x.setdefault("hour_utc", datetime.now(timezone.utc).hour)
    return x


def enrich(x, smart_weight=0.30, fetch_micro=True):
    """Microstructure + Smart Money + ML + Bayesian. يرجّع x بعد التعديل."""
    attach_context(x)
    if fetch_micro and not x.get("_v5x_micro"):
        try:
            x.update(microstructure.fetch_all(x["symbol"], x.get("1h"), x.get("quote_volume", 0)))
        except Exception as e:  # noqa: BLE001
            log.debug("micro %s: %s", x.get("symbol"), e)
        x["_v5x_micro"] = True
    x["smart_money_score"] = microstructure.smart_money_score(x)

    ml_prob = None
    if C.ML_ENABLED and _model.usable:
        ml_prob = _model.predict(x)
    x["ml_prob"] = None if ml_prob is None else round(ml_prob, 4)
    cal = _model.calibrate(ml_prob) if (ml_prob is not None and _model.calibrated) else None
    x["ml_prob_cal"] = None if cal is None else round(cal, 4)

    if C.BAYES_ENABLED:
        try:
            # الـ prior = الاحتمال المعاير بس (الخام مش احتمال حقيقي) وإلا نتايج الـ lane
            x.update(_get_bayes().adjust(x, cal))
        except Exception as e:  # noqa: BLE001
            log.debug("bayes %s: %s", x.get("symbol"), e)

    base = max(0.0, min(100.0, float(x.get("score", 0))))
    smart = float(x.get("smart_money_score", 50))
    rank = (1 - smart_weight) * base + smart_weight * smart
    if x.get("bayes_score") is not None:
        rank = 0.6 * rank + 0.4 * float(x["bayes_score"])
    x["final_rank"] = round(rank, 2)
    return x


def fng_extra_score(x=None):
    extra, blocked, label = sentiment.score_adjustment(_ctx["fng"])
    return extra, blocked, label


def gate(x, lane_fn):
    """قرار إضافي قبل إرسال FINAL. lane_fn = final_lane من الماسح.

    يرجّع (ok, reason)."""
    try:
        if not x.get("_v5x_micro"):
            enrich(x)
        extra, blocked, label = fng_extra_score()
        if blocked:
            return False, f"FNG_BLOCK_{label}"
        if extra:
            probe = dict(x)
            probe["score"] = x.get("score", 0) - extra
            if lane_fn(probe) is None:
                return False, f"FNG_{label}_NEEDS_+{extra}_SCORE"
        if C.ML_ENABLED and C.ML_GATE and _model.helps and x.get("ml_prob") is not None:
            thr = _model.threshold
            if thr is not None and x["ml_prob"] < thr:
                return False, f"ML_PROB_{x['ml_prob']:.2f}<{thr:.2f}"
        if C.BAYES_ENABLED and C.BAYES_GATE and x.get("expected_r") is not None:
            if x["expected_r"] < C.BAYES_MIN_EXPECTED_R:
                return False, f"BAYES_EXP_R_{x['expected_r']:+.2f}"
        if x.get("ask_wall_near") and (x.get("obi_1_0pct") or 0) < -0.3:
            return False, "SELL_WALL_AND_NEGATIVE_OBI"
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        log.warning("v5x gate error (allowed): %s", e)
        return True, "GATE_ERROR_ALLOWED"


def track_fields(x):
    out = {}
    for k in TRACK_EXTRA_COLUMNS:
        v = x.get(k)
        if isinstance(v, bool):
            v = int(v)
        out[k] = v
    return out


def load_optimized(globals_dict, allowed, logger=None):
    """يحمّل state/optimized_thresholds.json على globals الماسح (قائمة بيضاء + أنواع آمنة)."""
    path = C.OPTIMIZED_THRESHOLDS_FILE
    if os.getenv("USE_OPTIMIZED", "1") == "0" or not os.path.exists(path):
        return []
    applied = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k == "_fng_mode" and isinstance(v, str) and "FNG_MODE" not in os.environ:
                C.FNG_MODE = v
                applied.append(f"FNG_MODE={v}")
                continue
            if k not in allowed or k not in globals_dict:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            old = globals_dict[k]
            globals_dict[k] = int(round(v)) if isinstance(old, int) else float(v)
            applied.append(f"{k}: {old} -> {globals_dict[k]}")
        if applied and logger:
            logger.info("Optimizer thresholds: " + " | ".join(applied))
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning(f"optimized thresholds load failed: {e}")
    return applied
