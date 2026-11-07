"""(9) Auto Parameter Optimization — walk-forward (train → test) فوق الـ backtest.

1) random search + تحسين محلي حوالين أفضل النتائج على فترة الـ train فقط.
2) أفضل إعداد (حسب الـ train) يتقيّم مرة واحدة على فترة الـ test اللي ما اتشافتش.
3) يتقبل بس لو: test expectancy > 0 و test objective ≥ الإعدادات الحالية على نفس الـ test
   وعدد صفقات الـ test ≥ MIN_TEST_TRADES. غير كده الإعدادات الحالية تفضل زي ما هي.
4) لو اتقبل و --apply: يكتب state/optimized_thresholds.json (الماسح يقراه تلقائيًا).
"""
import json
import logging
import math
import os
import random
from datetime import datetime, timezone

import numpy as np

from . import config as C
from .backtest_engine import get_scanner, metrics, simulate_signals

log = logging.getLogger("v5ext.opt")

# (min, max, type)
SPACE = {
    # مساحة صغيرة عن قصد: كل معامل زيادة على بيانات قليلة = فرصة أكبر لحفظ الماضي (Overfitting)
    "FINAL_SCORE": (74, 90, int),
    "PERSISTENT_FINAL_SCORE": (66, 84, int),
    "FINAL_MIN_BUY_PRESSURE": (0.52, 0.66, float),
    "FINAL_MIN_VOLUME_15M": (2.5, 7.0, float),
    "RISK_ATR_MULT_SL": (1.0, 2.5, float),
    "RISK_RR_TP1": (1.0, 2.5, float),
}
FNG_MODES = ["adjust", "off", "block"]
OPTIMIZABLE = set(SPACE)            # المسموح للماسح يقراه من optimized_thresholds.json


def objective(m, min_trades):
    """SQN-like: متوسط R ÷ انحرافه × √n (n محدود بـ 100) − عقوبة drawdown."""
    n = m.get("trades", 0)
    if n < min_trades:
        return -1e9
    r_mean = m["expectancy_r"]
    sd = m.get("_r_std") or 1.0
    sqn = r_mean / max(sd, 1e-6) * math.sqrt(min(n, 100))
    return sqn - 0.02 * m.get("max_dd_r", 0.0)


def _eval(records, klines, params, fng_mode, min_trades):
    tr = simulate_signals(records, klines, params, fng_mode=fng_mode)
    m = metrics(tr)
    m["_r_std"] = float(tr["r_multiple"].std()) if len(tr) > 1 else 1.0
    return objective(m, min_trades), m


def _sample(rng):
    p = {}
    for k, (lo, hi, t) in SPACE.items():
        p[k] = rng.randint(lo, hi) if t is int else round(rng.uniform(lo, hi), 3)
    return p


def _perturb(p, rng, scale=0.15):
    q = dict(p)
    for k, (lo, hi, t) in SPACE.items():
        if rng.random() < 0.5:
            span = (hi - lo) * scale
            v = q[k] + rng.uniform(-span, span)
            v = min(hi, max(lo, v))
            q[k] = int(round(v)) if t is int else round(v, 3)
    return q


def current_params():
    sc = get_scanner()
    return {k: getattr(sc, k) for k in SPACE}


def optimize(records, klines, trials=200, refine=60, train_frac=0.7, min_train_trades=25,
             min_test_trades=30, seed=42, apply=False):
    rng = random.Random(seed)
    ts = records["ts"].to_numpy()
    cut = float(np.quantile(ts, train_frac))
    train = records[records.ts < cut]
    test = records[records.ts >= cut + 86_400_000 / 2]       # embargo نص يوم
    base = current_params()

    results = []
    cands = [(base, "adjust")] + [(_sample(rng), rng.choice(FNG_MODES)) for _ in range(trials)]
    for p, fm in cands:
        obj, m = _eval(train, klines, p, fm, min_train_trades)
        results.append((obj, p, fm, m))
    results.sort(key=lambda r: r[0], reverse=True)
    for _ in range(refine):
        _, p0, fm0, _ = results[rng.randrange(min(5, len(results)))]
        p = _perturb(p0, rng)
        fm = fm0 if rng.random() < 0.8 else rng.choice(FNG_MODES)
        obj, m = _eval(train, klines, p, fm, min_train_trades)
        results.append((obj, p, fm, m))
        results.sort(key=lambda r: r[0], reverse=True)

    best_obj, best_p, best_fm, best_train_m = results[0]
    base_obj_train, base_train_m = next(((o, m) for o, p, fm, m in results
                                         if p == base and fm == "adjust"), (None, None))
    test_obj, test_m = _eval(test, klines, best_p, best_fm, min_test_trades)
    base_test_obj, base_test_m = _eval(test, klines, base, "adjust", min_test_trades)

    accepted = bool(best_obj > -1e8 and test_m["trades"] >= min_test_trades
                    and test_m["expectancy_r"] > 0 and test_obj >= base_test_obj)
    reason = "ACCEPTED" if accepted else (
        "TOO_FEW_TEST_TRADES" if test_m["trades"] < min_test_trades else
        "NEGATIVE_TEST_EXPECTANCY" if test_m["expectancy_r"] <= 0 else
        "NOT_BETTER_THAN_CURRENT" if test_obj < base_test_obj else "NO_VALID_TRAIN_RESULT")

    def clean(m):
        return {k: v for k, v in (m or {}).items() if not k.startswith("_")}

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "trials": len(results), "train_frac": train_frac,
        "train_records": int(len(train)), "test_records": int(len(test)),
        "best_params": best_p, "best_fng_mode": best_fm,
        "best_train": clean(best_train_m), "best_train_objective": round(best_obj, 4),
        "best_test": clean(test_m), "best_test_objective": round(test_obj, 4),
        "current_params": base, "current_train": clean(base_train_m),
        "current_test": clean(base_test_m), "current_test_objective": round(base_test_obj, 4),
        "accepted": accepted, "decision": reason, "applied": False,
        "top10": [{"objective": round(o, 4), "params": p, "fng_mode": fm, "trades": m["trades"],
                   "expectancy_r": m["expectancy_r"]} for o, p, fm, m in results[:10]],
    }
    if accepted and apply:
        os.makedirs(C.STATE_DIR, exist_ok=True)
        with open(C.OPTIMIZED_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
            json.dump({**best_p, "_fng_mode": best_fm, "_updated": report["run_at"]}, f, indent=2)
        report["applied"] = True
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    with open(C.OPTIMIZER_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return report

