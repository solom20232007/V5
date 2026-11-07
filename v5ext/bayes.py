"""(6) Bayesian Score Adjustment.

الفكرة:
  prior  = احتمال نجاح الإشارة (TP1 قبل SL):
           • احتمال نموذج الـ ML لو موجود وصالح، وإلا
           • Beta posterior للـ lane:  Beta(1 + w·k_bt + w_live, 1 + l·k_bt + l_live)
             (الـ backtest وزنه محدود بـ BAYES_PRIOR_STRENGTH عشان الأداء الحي يغلب بسرعة)
  evidence = نسب أرجحية (Likelihood Ratios) متعلّمة من النتائج الحية لكل bucket من:
           Order-book imbalance (±1% بس — النسخة الموزونة مرتبطة بيها فبتتشال عشان ما تتعدّش مرتين)،
           Whale net flow، OI change، Fear&Greed
           LR بيتسحب ناحية 1 لما العينات قليلة ومحصور في [0.5, 2].
  posterior odds = prior odds × Π LR
  expected_R = p·RR_TP1 − (1−p)·1 − cost_R
"""
import json
import logging
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from . import sentiment

log = logging.getLogger("v5ext.bayes")

EVIDENCE = {
    "obi": ("obi_1_0pct", [-0.2, 0.2]),
    "whale": ("whale_net_ratio", [-0.3, 0.3]),
    "oi": ("oi_change_1h", [-1.0, 1.0]),
    "fng": ("fng", [C.FNG_EXTREME_FEAR, 45, 55, C.FNG_EXTREME_GREED - 0.5]),
}


def bucket(value, edges):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return str(int(np.searchsorted(edges, v, side="right")))


def _wins_losses(df, ret_col):
    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    return int((r > 0).sum()), int((r <= 0).sum())


def build_stats(backtest_trades=None, live_results=None):
    cost = 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT)
    lanes = {}
    bt = backtest_trades if backtest_trades is not None and len(backtest_trades) else pd.DataFrame()
    lv = pd.DataFrame()
    if live_results is not None and len(live_results):
        lv = live_results[live_results["outcome"].isin(["TP1", "SL", "TIMEOUT"])].copy()
        lv["net_pct"] = pd.to_numeric(lv["exit_pct"], errors="coerce") - cost

    def lane_frames(name):
        b = bt if name == "ALL" or not len(bt) else bt[bt["lane"].fillna("NONE") == name]
        l = lv if name == "ALL" or not len(lv) else lv[lv["lane"].fillna("NONE") == name]
        return b, l

    names = {"ALL"}
    if len(bt):
        names |= set(bt["lane"].fillna("NONE").astype(str))
    if len(lv):
        names |= set(lv["lane"].fillna("NONE").astype(str))
    for name in sorted(names):
        b, l = lane_frames(name)
        bw, bl = _wins_losses(b, "net_pct") if len(b) else (0, 0)
        lw, ll = _wins_losses(l, "net_pct") if len(l) else (0, 0)
        n_bt = bw + bl
        k = min(1.0, C.BAYES_PRIOR_STRENGTH / n_bt) if n_bt else 0.0
        a = 1 + bw * k + lw
        bb = 1 + bl * k + ll
        mean = a / (a + bb)
        var = a * bb / ((a + bb) ** 2 * (a + bb + 1))
        lanes[name] = {"alpha": round(a, 3), "beta": round(bb, 3), "mean": round(mean, 4),
                       "lower_80": round(max(0.0, mean - 1.2816 * math.sqrt(var)), 4),
                       "bt_wins": bw, "bt_losses": bl, "live_wins": lw, "live_losses": ll}

    # Likelihood ratios — من الحي (والـ fng كمان من الـ backtest لأن تاريخه متاح)
    lrs = {}
    for key, (col, edges) in EVIDENCE.items():
        frames = [lv] + ([bt] if key == "fng" and len(bt) else [])
        rows = []
        for f in frames:
            if len(f) and col in f.columns:
                rows.append(pd.DataFrame({"b": f[col].map(lambda v: bucket(v, edges)),
                                          "win": pd.to_numeric(f["net_pct"], errors="coerce") > 0}))
        if not rows:
            continue
        e = pd.concat(rows).dropna(subset=["b"])
        if len(e) < C.BAYES_LR_MIN_SAMPLES or e["win"].nunique() < 2:
            continue
        W, L = int(e.win.sum()), int((~e.win).sum())
        nb = len(edges) + 1
        table = {}
        for b, g in e.groupby("b"):
            w, l = int(g.win.sum()), int((~g.win).sum())
            lr = ((w + 1) / (W + nb)) / ((l + 1) / (L + nb))
            shrink = len(g) / (len(g) + C.BAYES_LR_MIN_SAMPLES)
            lr = float(np.clip(lr ** shrink, *C.BAYES_LR_CLIP))
            table[b] = {"lr": round(lr, 4), "n": int(len(g)), "wins": w}
        lrs[key] = table

    stats = {"updated": datetime.now(timezone.utc).isoformat(), "lanes": lanes, "lr": lrs,
             "cost_pct": cost}
    os.makedirs(C.STATE_DIR, exist_ok=True)
    with open(C.BAYES_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    return stats


class BayesianAdjuster:
    def __init__(self, stats=None):
        self.stats = stats
        if self.stats is None:
            try:
                with open(C.BAYES_STATS_FILE, "r", encoding="utf-8") as f:
                    self.stats = json.load(f)
            except Exception:  # noqa: BLE001
                self.stats = {"lanes": {}, "lr": {}, "cost_pct": 2 * (C.BT_FEE_PCT + C.BT_SLIPPAGE_PCT)}

    def lane_prior(self, lane):
        lanes = self.stats.get("lanes", {})
        s = lanes.get(str(lane)) or lanes.get("ALL")
        return (s or {}).get("mean", 0.45), s

    def adjust(self, x, ml_prob=None):
        lane = x.get("lane") or "ALL"
        lane_p, lane_stats = self.lane_prior(lane)
        prior = ml_prob if ml_prob is not None else lane_p
        prior = float(np.clip(prior, 0.01, 0.99))
        logit = math.log(prior / (1 - prior))
        used = {}
        for key, (col, edges) in EVIDENCE.items():
            if key == "fng" and ml_prob is not None:
                continue   # الـ fng جوه features الـ ML بالفعل → منع العد المزدوج
            b = bucket(x.get(col), edges)
            entry = (self.stats.get("lr", {}).get(key) or {}).get(b) if b is not None else None
            if entry:
                logit += math.log(entry["lr"])
                used[key] = entry["lr"]
        p = 1 / (1 + math.exp(-logit))
        risk = x.get("risk") or {}
        rr = float(risk.get("tp1_rr") or 1.5)
        sl_pct = float(risk.get("sl_pct") or 4.0)
        cost_r = self.stats.get("cost_pct", 0.3) / max(sl_pct, 0.1)
        exp_r = p * rr - (1 - p) * 1.0 - cost_r
        return {
            "bayes_prob": round(p, 4),
            "bayes_score": int(round(p * 100)),
            "bayes_prior": round(prior, 4),
            "bayes_prior_src": "ML" if ml_prob is not None else f"LANE:{lane}",
            "bayes_lr": used,
            "expected_r": round(exp_r, 3),
            "lane_lower_80": (lane_stats or {}).get("lower_80"),
        }


def fng_penalty_label(value):
    return sentiment.classify(value)
