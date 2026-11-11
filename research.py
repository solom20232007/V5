#!/usr/bin/env python3
"""V5.1 Research CLI — Backtest / XGBoost / Bayesian / Portfolio / Optimizer / Dashboard.

أمثلة:
  python research.py all --days 30 --top 60            # الخط الكامل (يحتاج إنترنت لـ Binance)
  python research.py backtest --days 45 --symbols INJUSDT,APTUSDT
  python research.py train
  python research.py optimize --trials 300 --apply
  python research.py portfolio --capital 1000 --risk 0.01 --max-positions 5
  python research.py dashboard                          # يبني docs/index.html
  python research.py all --synthetic                   # اختبار كامل بدون إنترنت (بيانات وهمية)
"""
import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

from v5ext import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("research")
logging.getLogger("scanner").setLevel(logging.WARNING)


def _json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def resolve_universe(args):
    from v5ext.backtest_engine import get_scanner
    from v5ext.binance_data import top_usdt_symbols
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    sc = get_scanner()
    allowed = None
    if sc.SHARIAH_FILTER_ENABLED:
        allowed = set(sc.SHARIAH_COMPLIANT_BASES) | sc.load_shariah_cache() | sc.load_user_shariah_overrides()
    exclude = set(sc.STABLECOIN_BASES) | set(sc.MEGA_CAP_BASES) | set(sc.STABLE_BASES)
    syms = top_usdt_symbols(n=args.top + args.skip_top, min_quote_volume=sc.MIN_24H_QUOTE_VOLUME,
                            exclude_bases=exclude, allowed_bases=allowed)
    return syms[args.skip_top:]


def load_data(args):
    """يرجّع (klines dict, btc_df, fng_history)."""
    from v5ext import sentiment, synthetic
    from v5ext.binance_data import load_klines
    if args.synthetic:
        syms = [f"SYN{i}USDT" for i in range(args.top)]
        kl = {s: synthetic.make_klines(s, days=args.days, seed=i) for i, s in enumerate(syms)}
        btc = synthetic.make_klines("BTCUSDT", days=args.days, seed=999, base_price=60000, pumps_per_day=0)
        fng = synthetic.make_fng_history(int(btc.open_time.iloc[0]), args.days)
        return kl, btc, fng
    syms = resolve_universe(args)
    log.info(f"Universe: {len(syms)} symbols, {args.days} days")
    now_ms = int(time.time() * 1000)
    kl = {}
    for i, s in enumerate(syms, 1):
        try:
            df = load_klines(s, "5m", args.days + 1, now_ms)
            if len(df) > C.BT_WARMUP_BARS + 50:
                kl[s] = df
            log.info(f"[{i}/{len(syms)}] {s}: {len(df)} bars")
        except Exception as e:  # noqa: BLE001
            log.warning(f"{s}: klines failed: {e}")
    btc = load_klines("BTCUSDT", "5m", args.days + 1, now_ms)
    fng = sentiment.get_history()
    return kl, btc, fng


def load_cached_klines(symbols=None):
    out = {}
    if not os.path.isdir(C.KLINE_CACHE_DIR):
        return out
    from v5ext.binance_data import normalize_klines
    for fn in os.listdir(C.KLINE_CACHE_DIR):
        if not fn.endswith("_5m.csv.gz"):
            continue
        sym = fn[:-len("_5m.csv.gz")]
        if sym == "BTCUSDT" or (symbols and sym not in symbols):
            continue
        out[sym] = normalize_klines(pd.read_csv(os.path.join(C.KLINE_CACHE_DIR, fn)))
    return out


# ------------------------------------------------------------------
# Steps
# ------------------------------------------------------------------
def step_backtest(args, kl, btc, fng):
    from v5ext.backtest_engine import label_records, metrics, scan_history, simulate_signals
    t0 = time.time()
    log.info(f"Stage A: replaying {len(kl)} symbols through analyze() ...")
    records = scan_history(kl, btc, fng, workers=args.workers, stride=args.stride)
    log.info(f"  {len(records)} scan records in {time.time() - t0:.0f}s")
    if not len(records):
        log.warning("لا يوجد records — جرّب أيام أكثر أو عملات أكثر")
        return None, None
    dataset = label_records(records, kl)
    os.makedirs(C.DATA_DIR, exist_ok=True)
    dataset.to_csv(C.SCAN_RECORDS_FILE, index=False)
    dataset.to_csv(C.ML_DATASET_FILE, index=False)
    trades = simulate_signals(dataset, kl, exit_mode=args.exit_mode)
    m = metrics(trades)
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    trades.to_csv(C.BACKTEST_TRADES_FILE, index=False)
    lbl = dataset["label"].dropna()
    summary = {
        "run_at": pd.Timestamp.now("UTC").isoformat(), "synthetic": bool(args.synthetic),
        "symbols": len(kl), "days": args.days, "exit_mode": args.exit_mode,
        "fee_pct_per_side": C.BT_FEE_PCT, "slippage_pct_per_side": C.BT_SLIPPAGE_PCT,
        "scan_records": int(len(dataset)), "labeled_rows": int(len(lbl)),
        "label_base_rate": round(float(lbl.mean()), 4) if len(lbl) else None,
        "rules": m,
        "limitations": [
            "لا يوجد order book / whales تاريخي → الـ backtest يختبر قواعد الـ score + ML + F&G فقط",
            "فلتر شمعة 1m وتأكيد Bybit/OKX غير مطبّقين في الـ backtest",
            "SL و TP في نفس شمعة 5m → يُحتسب SL (افتراض متحفّظ)",
            "تنبيه الفشل المبكر (بيع فورًا) مش محسوب كخروج — الـ backtest بيفترض إنك ملتزم بالوقف",
            "اختيار العملات حسب حجم اليوم → انحياز بقاء (survivorship bias)",
        ],
    }
    _json_dump(summary, C.BACKTEST_SUMMARY_FILE)
    log.info(f"Backtest: {m['trades']} trades | win {m['win_rate']:.1%} | E[R] {m['expectancy_r']:+.3f} | PF {m['profit_factor']}")
    return dataset, trades


def step_train(args, dataset=None):
    from v5ext import ml_model
    if dataset is None:
        dataset = pd.read_csv(C.ML_DATASET_FILE)
    meta, oos = ml_model.train(dataset, folds=args.folds)
    oos.to_csv(os.path.join(C.DATA_DIR, "ml_oos_predictions.csv.gz"), index=False)
    log.info(f"ML ({meta['backend']}): OOS AUC={meta['auc_oos']} | threshold={meta['threshold']} | "
             f"holdout E[R] all={meta['holdout_expectancy_r_all']} filtered={meta['holdout_expectancy_r_filtered']}")
    return meta, oos


def step_ml_backtest(args, dataset, kl, oos):
    """مقارنة عادلة: القواعد وحدها vs القواعد + ML على نفس الفترة OOS."""
    from v5ext.backtest_engine import metrics, simulate_signals
    if oos is None or not len(oos):
        return None
    probs = {(r.symbol, int(r.ts)): float(r.prob) for r in oos.itertuples(index=False)}
    meta = json.load(open(C.MODEL_META_FILE, encoding="utf-8"))
    thr = meta.get("threshold")
    period = dataset[dataset.ts >= oos.ts.min()]
    base = metrics(simulate_signals(period, kl, exit_mode=args.exit_mode))
    ml_tr = simulate_signals(period, kl, exit_mode=args.exit_mode, ml_probs=probs, ml_min_prob=thr)
    ml = metrics(ml_tr)
    s = json.load(open(C.BACKTEST_SUMMARY_FILE, encoding="utf-8"))
    s["oos_comparison"] = {"period_from": pd.to_datetime(int(oos.ts.min()), unit="ms", utc=True).isoformat(),
                           "ml_threshold": thr, "rules_only": base, "rules_plus_ml": ml}
    _json_dump(s, C.BACKTEST_SUMMARY_FILE)
    log.info(f"OOS rules-only E[R]={base['expectancy_r']:+.3f} ({base['trades']}) vs +ML E[R]={ml['expectancy_r']:+.3f} ({ml['trades']})")
    return s["oos_comparison"]


def step_tp_experiment(args, dataset, kl, scales=(0.67, 1.0, 1.33)):
    """تجربة واحدة مضبوطة: مسافة الأهداف بس (كل حاجة تانية ثابتة).
    الحكم على آخر 30% من الفترة (test). مفيش أي تطبيق أوتوماتيك."""
    from v5ext import trade_manager as TM
    from v5ext.backtest_engine import metrics, simulate_signals
    cut = float(dataset["ts"].quantile(0.7))
    parts = {"train": dataset[dataset.ts < cut], "test": dataset[dataset.ts >= cut]}
    keep = ["trades", "win_rate", "expectancy_r", "avg_net_pct", "profit_factor", "total_r", "max_dd_r", "by_outcome"]
    res = {}
    try:
        for sc_ in scales:
            TM.TP_SCALE = sc_
            res[f"x{sc_}"] = {name: {k: metrics(simulate_signals(d, kl, exit_mode=args.exit_mode)).get(k) for k in keep}
                              for name, d in parts.items()}
            t = res[f"x{sc_}"]
            log.info(f"TP x{sc_}: train {t['train']['trades']} trades avg {t['train']['avg_net_pct']:+.2f}% | "
                     f"test {t['test']['trades']} trades avg {t['test']['avg_net_pct']:+.2f}% win {t['test']['win_rate']:.0%}")
    finally:
        TM.TP_SCALE = 1.0
    s = json.load(open(C.BACKTEST_SUMMARY_FILE, encoding="utf-8"))
    s["tp_experiment"] = {"note": "x1.0 = الحالي. أقل من 1 = أهداف أقرب. الحكم على test.",
                          "test_from": pd.to_datetime(int(cut), unit="ms", utc=True).isoformat(), "results": res}
    _json_dump(s, C.BACKTEST_SUMMARY_FILE)
    return res


def step_burst_experiment(args, dataset, kl):
    """هل فلتر اندفاعة السوق وتباعد الإشارات بيحسّنوا النتيجة؟ العتبات من train، الحكم على test."""
    from v5ext.backtest_engine import burst_thresholds, get_scanner, metrics, simulate_signals
    sc = get_scanner()
    cut = float(dataset["ts"].quantile(0.7))
    parts = {"train": dataset[dataset.ts < cut], "test": dataset[dataset.ts >= cut]}
    thr = burst_thresholds(parts["train"], sc.WATCH_SCORE)
    keep = ["trades", "win_rate", "expectancy_r", "avg_net_pct", "profit_factor", "total_r", "max_dd_r", "max_losing_streak"]
    variants = {"none": (None, 0), "spacing_60m": (None, 60), "burst": (thr, 0), "burst+spacing": (thr, 60)}
    res = {}
    for name, (b, sp) in variants.items():
        res[name] = {p: {k: metrics(simulate_signals(d, kl, exit_mode=args.exit_mode, burst=b, spacing_min=sp)).get(k)
                         for k in keep} for p, d in parts.items()}
        t = res[name]["test"]
        log.info(f"Filter {name:14}: test {t['trades']} trades | win {t['win_rate']:.0%} | avg {t['avg_net_pct']:+.2f}% | "
                 f"maxDD {t['max_dd_r']}R | streak {t['max_losing_streak']}")
    s = json.load(open(C.BACKTEST_SUMMARY_FILE, encoding="utf-8"))
    s["burst_experiment"] = {"thresholds_from_train": thr, "symbols": len(kl), "results": res}
    _json_dump(s, C.BACKTEST_SUMMARY_FILE)
    return res


def step_pullback_experiment(args, dataset, kl):
    """الدخول بعد ارتداد مقابل الدخول الفوري. نفس المحفّزات، نفس إدارة الصفقة، نفس الرسوم.
    الاختيار على train (20 صفقة على الأقل)، والحكم على test. مفيش تطبيق أوتوماتيك."""
    from v5ext.backtest_engine import metrics, simulate_signals
    cut = float(dataset["ts"].quantile(0.7))
    parts = {"train": dataset[dataset.ts < cut], "test": dataset[dataset.ts >= cut]}
    keep = ["trades", "win_rate", "expectancy_r", "avg_net_pct", "profit_factor", "total_r",
            "max_dd_r", "max_losing_streak", "by_outcome"]
    base_name = "now (current)"
    variants = {
        base_name: dict(entry_mode="NOW"),
        "pullback 38%/2h": dict(entry_mode="PULLBACK", pullback=dict(retrace=0.382, wait_bars=24)),
        "pullback 50%/2h": dict(entry_mode="PULLBACK", pullback=dict(retrace=0.5, wait_bars=24)),
        "pullback 50%/4h": dict(entry_mode="PULLBACK", pullback=dict(retrace=0.5, wait_bars=48)),
        "pullback 50%/2h relaxed": dict(entry_mode="PULLBACK", pullback=dict(retrace=0.5, wait_bars=24),
                                        relaxed_trigger=True),
    }
    res = {}
    for name, kw in variants.items():
        res[name] = {}
        for p, d in parts.items():
            t = simulate_signals(d, kl, exit_mode=args.exit_mode, **kw)
            res[name][p] = {k: metrics(t).get(k) for k in keep}
        tr, te = res[name]["train"], res[name]["test"]
        log.info(f"Entry {name:24}: train {tr['trades']:3} avg {tr['avg_net_pct']:+.2f}% | "
                 f"test {te['trades']:3} avg {te['avg_net_pct']:+.2f}% win {te['win_rate']:.0%} PF {te['profit_factor']}")
    pick = [n for n in variants if n != base_name and res[n]["train"]["trades"] >= 20]
    best = max(pick, key=lambda n: res[n]["train"]["avg_net_pct"]) if pick else None
    verdict = None
    if best:
        te, base = res[best]["test"], res[base_name]["test"]
        verdict = {
            "chosen_on_train": best,
            "test_trades": te["trades"], "test_avg_net_pct": te["avg_net_pct"],
            "test_profit_factor": te["profit_factor"],
            "baseline_test_avg_net_pct": base["avg_net_pct"],
            "passes": bool(te["trades"] >= 20 and te["avg_net_pct"] > 0 and (te["profit_factor"] or 0) > 1.1
                           and te["avg_net_pct"] > base["avg_net_pct"]),
            "rule": "PASS = test >= 20 trades, avg > 0, PF > 1.1, and better than current entry",
        }
        log.info(f"Pullback verdict: {best} -> test avg {te['avg_net_pct']:+.2f}% ({te['trades']} trades) "
                 f"vs now {base['avg_net_pct']:+.2f}% -> {'PASS' if verdict['passes'] else 'FAIL'}")
    else:
        log.info("Pullback verdict: no variant reached 20 train trades -> FAIL")
    s = json.load(open(C.BACKTEST_SUMMARY_FILE, encoding="utf-8"))
    s["pullback_experiment"] = {"results": res, "verdict": verdict,
                                "test_from": pd.to_datetime(int(cut), unit="ms", utc=True).isoformat()}
    _json_dump(s, C.BACKTEST_SUMMARY_FILE)
    return res


def _synthetic_daily(symbol, days, seed, drift=0.0005):
    import numpy as np
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.04, days)
    # موجات اتجاه
    for _ in range(days // 120):
        s = int(rng.integers(0, days - 40)); r[s:s + 30] += rng.choice([-1, 1]) * 0.012
    c = 10 * np.exp(np.cumsum(r)); o = np.concatenate([[10], c[:-1]])
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.015, days)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.015, days)))
    t0 = 1_700_000_000_000 // 86_400_000 * 86_400_000
    ot = t0 + np.arange(days) * 86_400_000
    return pd.DataFrame({"open_time": ot, "open": o, "high": h, "low": l, "close": c,
                         "volume": 1.0, "close_time": ot + 86_399_999, "quote_volume": 1e7})


def step_trend(args):
    """استراتيجية الاتجاه اليومي — شموع يومية لآخر trend_days يوم."""
    from v5ext import trend_strategy as TS
    from v5ext import portfolio as P
    if args.synthetic:
        kl = {f"SYN{i}USDT": _synthetic_daily(f"SYN{i}", args.trend_days, i) for i in range(args.top)}
        btc = _synthetic_daily("BTC", args.trend_days, 999, drift=0.001)
    else:
        from v5ext.binance_data import load_klines
        syms = resolve_universe(args)
        now_ms = int(time.time() * 1000)
        kl = {}
        for s in syms:
            try:
                df = load_klines(s, "1d", args.trend_days, now_ms)
                if len(df) >= 120:
                    kl[s] = df
            except Exception as e:  # noqa: BLE001
                log.warning(f"{s}: daily klines failed: {e}")
        btc = load_klines("BTCUSDT", "1d", args.trend_days, now_ms)
    log.info(f"Trend: {len(kl)} symbols, {args.trend_days} days")
    # الاتجاه اليومي بيمسك الصفقات الكسبانة أسابيع → 5 خانات بتتملي وتضيّع إشارات. 10 خانات × 15% حد أقصى.
    cfg = P.PortfolioConfig(initial_capital=args.capital, risk_per_trade=args.risk,
                            max_position_pct=min(args.max_position, 0.15),
                            max_positions=max(args.max_positions, 10))
    rep = TS.experiment(kl, btc, portfolio_cfg=cfg)
    for name, v in (rep.get("results") or {}).items():
        tr, te, pt = v["train"], v["test"], v["test_portfolio"]
        log.info(f"Trend {name:20}: train {tr['trades']:3} avg {tr['avg_net_pct']:+.2f}% | test {te['trades']:3} "
                 f"avg {te['avg_net_pct']:+.2f}% PF {te['profit_factor']} | portfolio {pt.get('return_pct')}% "
                 f"DD {pt.get('max_drawdown_pct')}%")
    v = rep.get("verdict")
    if v:
        log.info(f"Trend verdict: {v['chosen_on_train']} -> test avg {v['test_avg_net_pct']:+.2f}% "
                 f"({v['test_trades']} trades), portfolio {v['test_portfolio_return_pct']}% vs BTC hold "
                 f"{v['btc_buy_hold_test_pct']}% -> {'PASS' if v['passes'] else 'FAIL'}")
    rep["synthetic"] = bool(args.synthetic)
    rep["days"] = args.trend_days
    rep["run_at"] = pd.Timestamp.now("UTC").isoformat()
    _json_dump(rep, os.path.join(C.REPORTS_DIR, "trend_experiment.json"))
    best = (v or {}).get("chosen_on_train")
    if best:
        kw = dict(TS.VARIANTS[best])
        TS.run(kl, btc, **kw).to_csv(os.path.join(C.REPORTS_DIR, "trend_trades.csv"), index=False)
    return rep


def read_live_results():
    try:
        return pd.read_csv(C.TRACK_FILE, low_memory=False)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def step_bayes(args):
    from v5ext.bayes import build_stats
    bt = pd.read_csv(C.BACKTEST_TRADES_FILE) if os.path.exists(C.BACKTEST_TRADES_FILE) else None
    stats = build_stats(bt, read_live_results())
    for lane, s in stats["lanes"].items():
        log.info(f"Bayes {lane:11}: mean={s['mean']:.3f} lower80={s['lower_80']:.3f} "
                 f"(bt {s['bt_wins']}/{s['bt_wins'] + s['bt_losses']}, live {s['live_wins']}/{s['live_wins'] + s['live_losses']})")
    return stats


def step_portfolio(args):
    from v5ext import portfolio as P
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    cfg = P.PortfolioConfig(initial_capital=args.capital, risk_per_trade=args.risk,
                            max_position_pct=args.max_position, max_positions=args.max_positions)
    out = {}
    if os.path.exists(C.BACKTEST_TRADES_FILE):
        bt = pd.read_csv(C.BACKTEST_TRADES_FILE)
        eq, summ, _ = P.simulate(bt, cfg)
        eq.to_csv(C.PORTFOLIO_EQUITY_FILE, index=False)
        _json_dump(summ, C.PORTFOLIO_SUMMARY_FILE)
        out["backtest"] = summ
        log.info(f"Portfolio (backtest): {summ.get('return_pct')}% | maxDD {summ.get('max_drawdown_pct')}% | taken {summ.get('taken')}/{summ.get('signals')}")
    live = P.trades_from_live_results(read_live_results())
    eq, summ, _ = P.simulate(live, cfg)
    eq.to_csv(C.LIVE_PORTFOLIO_EQUITY_FILE, index=False)
    _json_dump(summ, C.LIVE_PORTFOLIO_SUMMARY_FILE)
    out["live"] = summ
    log.info(f"Portfolio (live signals): {summ.get('return_pct')}% over {summ.get('taken')} trades")
    return out


def step_optimize(args, dataset=None, kl=None):
    from v5ext.optimizer import optimize
    if dataset is None:
        dataset = pd.read_csv(C.SCAN_RECORDS_FILE)
    if kl is None:
        kl = load_cached_klines(set(dataset.symbol.unique()))
    rep = optimize(dataset, kl, trials=args.trials, refine=args.refine, apply=args.apply,
                   min_train_trades=args.min_train_trades, min_test_trades=args.min_test_trades)
    log.info(f"Optimizer: {rep['decision']} | test E[R] best={rep['best_test'].get('expectancy_r')} "
             f"vs current={rep['current_test'].get('expectancy_r')} | applied={rep['applied']}")
    return rep


def step_dashboard(args):
    from dashboard.build import build_static
    path = build_static()
    log.info(f"Dashboard → {path}")
    return path


# ------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "backtest", "train", "bayes", "portfolio",
                                        "optimize", "dashboard", "trend", "all"])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--top", type=int, default=60, help="عدد العملات (حسب حجم 24h)")
    ap.add_argument("--skip-top", type=int, default=0, help="تخطي أعلى N عملة حجمًا")
    ap.add_argument("--symbols", default="", help="قائمة مفصولة بفواصل بدل --top")
    ap.add_argument("--synthetic", action="store_true", help="بيانات وهمية للاختبار بدون إنترنت")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="1 = كل شمعة (زي الماسح كل 5 دقايق)")
    ap.add_argument("--exit-mode", choices=["MANAGED", "TP1", "SPLIT"], default="MANAGED",
                    help="MANAGED = نفس خطة الأهداف الحية (ثلث عند كل هدف + تحريك الوقف)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--refine", type=int, default=60)
    ap.add_argument("--min-train-trades", type=int, default=40)
    ap.add_argument("--min-test-trades", type=int, default=30)
    ap.add_argument("--apply", action="store_true", help="اكتب العتبات المحسّنة لو اتقبلت")
    ap.add_argument("--trend-days", type=int, default=730, help="أيام الشموع اليومية لاستراتيجية الاتجاه")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--max-position", type=float, default=0.25)
    ap.add_argument("--max-positions", type=int, default=5)
    args = ap.parse_args(argv)
    args.workers = args.workers or None

    cmd = args.command
    if cmd == "fetch":
        kl, _, _ = load_data(args)
        log.info(f"cached {len(kl)} symbols")
    elif cmd == "backtest":
        kl, btc, fng = load_data(args)
        step_backtest(args, kl, btc, fng)
    elif cmd == "train":
        step_train(args)
    elif cmd == "bayes":
        step_bayes(args)
    elif cmd == "portfolio":
        step_portfolio(args)
    elif cmd == "optimize":
        if args.synthetic:
            kl, _, _ = load_data(args)
            step_optimize(args, kl=kl)
        else:
            step_optimize(args)
    elif cmd == "dashboard":
        step_dashboard(args)
    elif cmd == "trend":
        step_trend(args)
    elif cmd == "all":
        kl, btc, fng = load_data(args)
        dataset, _ = step_backtest(args, kl, btc, fng)
        if dataset is None:
            return 1
        step_tp_experiment(args, dataset, kl)
        step_burst_experiment(args, dataset, kl)
        step_pullback_experiment(args, dataset, kl)
        try:
            step_trend(args)
        except Exception as e:  # noqa: BLE001
            log.warning(f"trend experiment failed: {e}")
        try:
            meta, oos = step_train(args, dataset)
            step_ml_backtest(args, dataset, kl, oos)
        except ValueError as e:
            log.warning(f"ML training skipped: {e}")
        step_bayes(args)
        step_optimize(args, dataset, kl)
        step_portfolio(args)
        step_dashboard(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
