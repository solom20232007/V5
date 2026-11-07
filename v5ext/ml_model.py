"""(5) XGBoost Model — يتنبأ باحتمال "TP1 قبل SL" لكل مرشّح.

- الـ features بتتاخد من نتيجة analyze() نفسها → نفس الحسابات في الـ backtest والـ live.
- التدريب walk-forward بفترات زمنية متتالية + embargo يوم (مفيش خلط مستقبل بماضي).
- العتبة المقترحة بتتختار على folds بدري وتتقيّم على آخر fold (مش على نفس البيانات).
- لو xgboost مش متسطب → HistGradientBoosting من sklearn (نفس الواجهة).
"""
import json
import logging
import math
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("v5ext.ml")

FEATURES = [
    "score", "5m", "15m", "1h", "4h", "24h", "volume_1h", "volume_15m", "acceleration",
    "prior_acceleration", "breakout", "close_location", "rsi14", "atr_pct", "buy_pressure",
    "buy_pressure_delta", "body_pct", "upper_wick_ratio", "distance_from_1h_high",
    "volume_persistence", "efficiency", "trend_continuity", "green_streak", "is_wash",
    "log_quote_volume", "ema9_dist", "ema21_dist", "ema50_dist", "btc_15m", "btc_1h", "fng",
    "hour_sin", "hour_cos", "new_listing_mode",
]


def _num(v):
    if v is None or isinstance(v, str):
        return np.nan
    try:
        f = float(v)
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def build_features(rows):
    """rows: DataFrame أو list[dict] بنفس مفاتيح analyze() + btc_15m/btc_1h/fng/hour_utc."""
    df = pd.DataFrame(rows).copy() if not isinstance(rows, pd.DataFrame) else rows.copy()
    out = pd.DataFrame(index=df.index)
    for f in FEATURES:
        if f in df.columns:
            out[f] = df[f].map(_num) if df[f].dtype == object else pd.to_numeric(df[f], errors="coerce")
    price = pd.to_numeric(df.get("price"), errors="coerce") if "price" in df else np.nan
    for e in ("ema9", "ema21", "ema50"):
        if e in df.columns:
            ev = pd.to_numeric(df[e], errors="coerce")
            out[f"{e}_dist"] = (price / ev - 1.0) * 100.0
    if "quote_volume" in df.columns:
        out["log_quote_volume"] = np.log10(pd.to_numeric(df["quote_volume"], errors="coerce").clip(lower=1))
    if "hour_utc" in df.columns:
        h = pd.to_numeric(df["hour_utc"], errors="coerce")
        out["hour_sin"] = np.sin(2 * np.pi * h / 24)
        out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    for b in ("is_wash", "new_listing_mode"):
        if b in df.columns:
            out[b] = df[b].map(lambda v: 1.0 if v in (True, 1, "True", "true", 1.0) else 0.0)
    for f in FEATURES:
        if f not in out.columns:
            out[f] = np.nan
    return out[FEATURES].astype(float)


# ------------------------------------------------------------------
def _new_model(n_pos, n_neg):
    try:
        import xgboost as xgb
        return "xgboost", xgb.XGBClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0, gamma=0.5,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            early_stopping_rounds=60, n_jobs=-1,
            # بدون scale_pos_weight: بيرفع الاحتمالات فوق الحقيقة (كان سبب أرقام مضللة)
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        log.warning("xgboost غير متسطب — استخدام HistGradientBoosting كبديل")
        return "sklearn_hgb", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            min_samples_leaf=40, early_stopping=True, validation_fraction=0.15, random_state=7)


def _fit(X, y):
    backend, m = _new_model(int(y.sum()), int(len(y) - y.sum()))
    if backend == "xgboost":
        cut = int(len(X) * 0.85)                 # validation = آخر 15% زمنيًا
        m.fit(X.iloc[:cut], y.iloc[:cut], eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
    else:
        m.fit(X, y)
    return backend, m


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    except ValueError:
        return float("nan")


def choose_threshold(prob, r, min_count=30):
    """العتبة اللي بتعظّم متوسط R للصفقات اللي فوقها (بحد أدنى عدد صفقات)."""
    prob, r = np.asarray(prob, float), np.asarray(r, float)
    best_t, best_e = None, -1e9
    for q in np.linspace(0.0, 0.9, 19):
        t = float(np.quantile(prob, q))
        sel = r[prob >= t]
        if len(sel) < max(min_count, int(0.05 * len(r))):
            continue
        e = float(np.nanmean(sel))
        if e > best_e:
            best_t, best_e = t, e
    return best_t, best_e


def train(dataset, folds=3, embargo_ms=86_400_000):
    """dataset: ناتج label_records. يرجّع (meta, oos_predictions DataFrame)."""
    d = dataset.dropna(subset=["label"]).sort_values("ts").reset_index(drop=True)
    if len(d) < 300 or d["label"].nunique() < 2:
        raise ValueError(f"بيانات غير كافية للتدريب: {len(d)} صف")
    X, y, ts, r = build_features(d), d["label"].astype(int), d["ts"].to_numpy(), d["r_multiple"]

    bounds = np.linspace(0.4, 1.0, folds + 1)
    oos = []
    fold_stats = []
    for k in range(folds):
        start_t = np.quantile(ts, bounds[k])
        end_t = np.quantile(ts, bounds[k + 1]) if k < folds - 1 else ts.max() + 1
        tr = ts < start_t - embargo_ms
        te = (ts >= start_t) & (ts < end_t)
        if tr.sum() < 200 or te.sum() < 30 or y[tr].nunique() < 2:
            continue
        backend, m = _fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oos.append(pd.DataFrame({"symbol": d.loc[te, "symbol"].values, "ts": ts[te], "prob": p,
                                 "label": y[te].values, "r_multiple": r[te].values, "fold": k}))
        fold_stats.append({"fold": k, "train": int(tr.sum()), "test": int(te.sum()),
                           "auc": round(_auc(y[te], p), 4), "base_rate": round(float(y[te].mean()), 4)})
    if not oos:
        raise ValueError("مفيش folds صالحة — زوّد عدد الأيام أو العملات")
    oos = pd.concat(oos, ignore_index=True)

    # العتبة: من folds بدري → تقييم على آخر fold
    last = oos["fold"].max()
    early, late = oos[oos.fold < last], oos[oos.fold == last]
    thr, thr_e = choose_threshold(early.prob, early.r_multiple) if len(early) else (None, None)
    if thr is None:
        thr, thr_e = choose_threshold(oos.prob, oos.r_multiple)
    late_sel = late[late.prob >= thr] if thr is not None else late
    holdout = {
        "threshold": None if thr is None else round(thr, 4),
        "threshold_expectancy_r_selection": None if thr_e is None else round(thr_e, 4),
        "holdout_trades_all": int(len(late)),
        "holdout_expectancy_r_all": round(float(late.r_multiple.mean()), 4) if len(late) else None,
        "holdout_trades_filtered": int(len(late_sel)),
        "holdout_expectancy_r_filtered": round(float(late_sel.r_multiple.mean()), 4) if len(late_sel) else None,
    }

    # معايرة الاحتمالات (Isotonic) على تنبؤات خارج العينة: "60%" لازم تعني ~60% فعلًا
    calibration, brier_raw, brier_cal, calibrated = None, None, None, False
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(early.prob if len(early) >= 200 else oos.prob, early.label if len(early) >= 200 else oos.label)
        grid = np.linspace(0, 1, 101)
        calibration = {"x": grid.round(4).tolist(), "y": np.round(iso.predict(grid), 4).tolist()}
        ev = late if len(early) >= 200 else oos           # تقييم المعايرة على بيانات ما اتعايرش عليها
        cal_p = np.interp(ev.prob, grid, iso.predict(grid))
        brier_raw = float(np.mean((ev.prob - ev.label) ** 2))
        brier_cal = float(np.mean((cal_p - ev.label) ** 2))
        reliability = []
        for lo in np.arange(0, 1, 0.2):
            m = (cal_p >= lo) & (cal_p < lo + 0.2)
            if m.sum() >= 10:
                reliability.append({"bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": int(m.sum()),
                                    "predicted": round(float(cal_p[m].mean()), 3),
                                    "actual": round(float(ev.label.values[m].mean()), 3)})
        # معايَر = تحسّن الـ Brier + كل نطاق فيه 30 إشارة أو أكتر الفرق بين المتوقع والفعلي ≤ 10 نقاط
        gaps = [abs(r["predicted"] - r["actual"]) for r in reliability if r["n"] >= 30]
        calibrated = bool(len(oos) >= 300 and brier_cal <= brier_raw + 0.005 and gaps and max(gaps) <= 0.10)
    except Exception as e:  # noqa: BLE001
        log.warning("calibration failed: %s", e)
        reliability = []

    # النموذج النهائي على كل البيانات (للاستخدام الحي)
    backend, final = _fit(X, y)
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    if backend == "xgboost":
        final.save_model(C.MODEL_FILE)
        imp = dict(zip(FEATURES, map(float, final.feature_importances_)))
        if os.path.exists(C.MODEL_FALLBACK_FILE):
            os.remove(C.MODEL_FALLBACK_FILE)
    else:
        with open(C.MODEL_FALLBACK_FILE, "wb") as f:
            pickle.dump(final, f)
        imp = {}
        if os.path.exists(C.MODEL_FILE):
            os.remove(C.MODEL_FILE)
    meta = {
        "backend": backend, "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(d)), "base_rate": round(float(y.mean()), 4),
        "data_from": pd.to_datetime(int(ts.min()), unit="ms", utc=True).isoformat(),
        "data_to": pd.to_datetime(int(ts.max()), unit="ms", utc=True).isoformat(),
        "auc_oos": round(_auc(oos.label, oos.prob), 4),
        "folds": fold_stats, **holdout,
        "calibrated": calibrated, "calibration": calibration,
        "brier_raw": None if brier_raw is None else round(brier_raw, 4),
        "brier_calibrated": None if brier_cal is None else round(brier_cal, 4),
        "reliability": reliability,
        "feature_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15]),
    }
    with open(C.MODEL_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta, oos


# ------------------------------------------------------------------
class LiveModel:
    """تحميل كسول + كل الأعطال بترجع None (الماسح يكمّل عادي)."""

    def __init__(self):
        self.model, self.meta, self.loaded = None, {}, False

    def load(self):
        if self.loaded:
            return self.model is not None
        self.loaded = True
        try:
            with open(C.MODEL_META_FILE, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            if self.meta.get("backend") == "xgboost" and os.path.exists(C.MODEL_FILE):
                import xgboost as xgb
                m = xgb.XGBClassifier()
                m.load_model(C.MODEL_FILE)
                self.model = m
            elif os.path.exists(C.MODEL_FALLBACK_FILE):
                with open(C.MODEL_FALLBACK_FILE, "rb") as f:
                    self.model = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("ML model not loaded: %s", e)
            self.model = None
        return self.model is not None

    @property
    def usable(self):
        if not self.load():
            return False
        auc = self.meta.get("auc_oos")
        return auc is not None and auc == auc and auc >= C.ML_MIN_AUC_TO_USE

    @property
    def threshold(self):
        if C.ML_MIN_PROB >= 0:
            return C.ML_MIN_PROB
        return self.meta.get("threshold")

    @property
    def helps(self):
        """الفلترة بالنموذج اتثبت إنها حسّنت النتيجة على آخر فترة ما اتشافتش (holdout)."""
        if not self.usable:
            return False
        a, f = self.meta.get("holdout_expectancy_r_all"), self.meta.get("holdout_expectancy_r_filtered")
        n = self.meta.get("holdout_trades_filtered") or 0
        return a is not None and f is not None and n >= 30 and f > a and f > 0

    @property
    def calibrated(self):
        return bool(self.usable and self.meta.get("calibrated") and self.meta.get("calibration"))

    def calibrate(self, p):
        cal = self.meta.get("calibration")
        if p is None or not cal:
            return None
        return float(np.interp(p, cal["x"], cal["y"]))

    def predict(self, x):
        if not self.load():
            return None
        try:
            return float(self.model.predict_proba(build_features([x]))[:, 1][0])
        except Exception as e:  # noqa: BLE001
            log.debug("ML predict failed: %s", e)
            return None
