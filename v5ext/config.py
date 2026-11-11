"""إعدادات مركزية — كلها قابلة للتغيير من متغيرات البيئة بدون تعديل كود."""
import os


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name, default):
    return os.getenv(name, "1" if default else "0").strip() not in ("0", "false", "False", "")


# ---------- Paths ----------
STATE_DIR = "state"
DATA_DIR = os.getenv("V5_DATA_DIR", "data")
MODELS_DIR = "models"
REPORTS_DIR = "reports"
DOCS_DIR = "docs"

KLINE_CACHE_DIR = os.path.join(DATA_DIR, "klines")
SCAN_RECORDS_FILE = os.path.join(DATA_DIR, "scan_records.csv.gz")
ML_DATASET_FILE = os.path.join(DATA_DIR, "ml_dataset.csv.gz")

BACKTEST_TRADES_FILE = os.path.join(REPORTS_DIR, "backtest_trades.csv")
BACKTEST_SUMMARY_FILE = os.path.join(REPORTS_DIR, "backtest_summary.json")
PORTFOLIO_EQUITY_FILE = os.path.join(REPORTS_DIR, "portfolio_equity.csv")
PORTFOLIO_SUMMARY_FILE = os.path.join(REPORTS_DIR, "portfolio_summary.json")
LIVE_PORTFOLIO_SUMMARY_FILE = os.path.join(REPORTS_DIR, "live_portfolio_summary.json")
LIVE_PORTFOLIO_EQUITY_FILE = os.path.join(REPORTS_DIR, "live_portfolio_equity.csv")
OPTIMIZER_REPORT_FILE = os.path.join(REPORTS_DIR, "optimizer_report.json")

MODEL_FILE = os.path.join(MODELS_DIR, "pump_model.json")          # XGBoost native format
MODEL_FALLBACK_FILE = os.path.join(MODELS_DIR, "pump_model.pkl")  # sklearn fallback
MODEL_META_FILE = os.path.join(MODELS_DIR, "pump_model_meta.json")

BAYES_STATS_FILE = os.path.join(STATE_DIR, "bayes_stats.json")
FNG_CACHE_FILE = os.path.join(STATE_DIR, "fear_greed_cache.json")
OPTIMIZED_THRESHOLDS_FILE = os.path.join(STATE_DIR, "optimized_thresholds.json")
TRACK_FILE = "early_pump_signal_results_v5.csv"
WATCH_LOG_FILE = "early_watch_log_v5.csv"

# ---------- Endpoints ----------
SPOT_BASE_URLS = ["https://data-api.binance.vision", "https://api.binance.com"]
FUTURES_BASE_URL = "https://fapi.binance.com"
FNG_URL = "https://api.alternative.me/fng/"

# ---------- 1) Order Book Imbalance ----------
OBI_DEPTH_LIMIT = _env_int("OBI_DEPTH_LIMIT", 100)
OBI_BANDS_PCT = (0.5, 1.0, 2.0)           # نطاقات حول الـ mid price
OBI_WALL_MULT = _env_float("OBI_WALL_MULT", 5.0)  # مستوى > 5x الوسيط = جدار

# ---------- 2) Open Interest ----------
OI_PERIOD = "5m"
OI_POINTS = 13                             # 13 نقطة × 5m = ساعة

# ---------- 3) Whale Transactions ----------
WHALE_ENABLED = _env_bool("WHALE_ENABLED", True)
WHALE_MIN_USD = _env_float("WHALE_MIN_USD", 5_000)
WHALE_MAX_USD = _env_float("WHALE_MAX_USD", 50_000)
# آخر 1000 صفقة بتغطي دقايق بس في العملات الكبيرة → الحد القديم (0.2% من حجم اليوم، لحد 250k$) كان دايمًا بيطلع صفر
WHALE_QV_FRACTION = _env_float("WHALE_QV_FRACTION", 0.0003)
WHALE_WINDOW_MIN = _env_int("WHALE_WINDOW_MIN", 15)

MICRO_CACHE_TTL = _env_int("MICRO_CACHE_TTL", 240)

# ---------- 4) Fear & Greed ----------
FNG_MODE = os.getenv("FNG_MODE", "adjust").lower()   # adjust | block | off
FNG_EXTREME_FEAR = _env_int("FNG_EXTREME_FEAR", 20)
FNG_EXTREME_GREED = _env_int("FNG_EXTREME_GREED", 80)
FNG_SCORE_PENALTY = _env_int("FNG_SCORE_PENALTY", 3)
FNG_CACHE_TTL = _env_int("FNG_CACHE_TTL", 3600)

# ---------- 5) ML ----------
ML_ENABLED = _env_bool("ML_ENABLED", True)
ML_GATE = _env_bool("ML_GATE", True)                 # يمنع الإشارة لو احتمال النموذج أقل من العتبة
ML_MIN_PROB = _env_float("ML_MIN_PROB", -1)          # -1 = استخدم العتبة اللي النموذج اقترحها
ML_MIN_AUC_TO_USE = _env_float("ML_MIN_AUC_TO_USE", 0.55)  # نموذج أضعف من كده يتجاهل

# ---------- 6) Bayesian ----------
BAYES_ENABLED = _env_bool("BAYES_ENABLED", True)
BAYES_GATE = _env_bool("BAYES_GATE", False)          # افتراضيًا: ترتيب وعرض فقط
BAYES_MIN_EXPECTED_R = _env_float("BAYES_MIN_EXPECTED_R", 0.0)
BAYES_PRIOR_STRENGTH = _env_float("BAYES_PRIOR_STRENGTH", 30)  # أقصى وزن للـ backtest كـ prior
BAYES_LR_MIN_SAMPLES = _env_int("BAYES_LR_MIN_SAMPLES", 20)
BAYES_LR_CLIP = (0.5, 2.0)

# ---------- Backtest ----------
BT_FEE_PCT = _env_float("BT_FEE_PCT", 0.10)         # لكل جهة (taker)
BT_SLIPPAGE_PCT = _env_float("BT_SLIPPAGE_PCT", 0.05)
BT_TIMEOUT_BARS = _env_int("BT_TIMEOUT_BARS", 48)    # 48 × 5m = 4 ساعات (زي TRADE_TIMEOUT_MIN)
BT_WARMUP_BARS = 289                                 # 24 ساعة + شمعة لحساب تغير 24h
BT_LABEL_MIN_SCORE = _env_int("BT_LABEL_MIN_SCORE", 55)  # يسجّل dataset للـ ML من score كده وفوق

# ---------- Paper mode ----------
# الـ backtest (90 يوم، 87 إشارة) طلع خسارة: −0.36% للصفقة بعد الرسوم.
# لحد ما استراتيجية تثبت ربح على فترة اختبار: الإشارات للمتابعة والتسجيل بس.
PAPER_MODE = _env_bool("PAPER_MODE", True)

# ---------- Monitor mode ----------
# الاستراتيجية (شراء بعد طلوع 15 دقيقة) خسرت على 180 يوم بكل التعديلات.
# الرسايل بقت تنبيهات رصد: "العملة بتتحرك بحجم غير عادي" — من غير دخول ولا وقف ولا أهداف.
# المتابعة والتسجيل شغالين في الخلفية للقياس، لكن تنبيهات TP/SL مش بتتبعت.
MONITOR_MODE = _env_bool("MONITOR_MODE", True)

# ---------- Market burst / correlated signals ----------
# اندفاعة سوق عامة = عدد العملات اللي عدّت WATCH في نفس الدورة في أعلى BURST_QUANTILE من التاريخ
BURST_FILTER = _env_bool("BURST_FILTER", True)
BURST_QUANTILE = _env_float("BURST_QUANTILE", 0.97)
BURST_MIN_COUNT = _env_int("BURST_MIN_COUNT", 6)        # حد أدنى ثابت لحد ما يتجمع تاريخ كفاية
BURST_BLOCK_MIN = _env_int("BURST_BLOCK_MIN", 30)
SIGNAL_SPACING_MIN = _env_int("SIGNAL_SPACING_MIN", 60) # إشارة جديدة واحدة بس كل 60 دقيقة (0 = بدون)
