# Binance Early-Pump Scanner — V5.1

ماسح للعملات اللي بتبدأ موجة صعود مبكرة على Binance Spot، مع طبقة ذكاء جديدة فوق قواعد V5.

> ⚠️ أداة تحليلية تجريبية — مش توصية شراء. كل الأرقام في الـ backtest والمحاكاة لا تضمن نتائج مستقبلية.

## الجديد في V5.1

| # | الميزة | الملف | بيعمل إيه |
|---|---|---|---|
| 0 | **Order Book Imbalance** | `v5ext/microstructure.py` | عدم توازن الأوامر في نطاقات ±0.5% / ±1% / ±2% من السعر + نسخة موزونة أُسّيًا (أصعب في الـ spoofing البعيد) + كشف جدار بيع قريب + الـ spread |
| 1 | **Open Interest** | `v5ext/microstructure.py` | تغيّر الـ OI على 15m و 1h + تصنيف: `NEW_LONGS` (سعر↑ + OI↑) / `SHORT_COVERING` / `NEW_SHORTS` |
| 2 | **Whale Transactions** | `v5ext/microstructure.py` | من `aggTrades`: صفقات أكبر من حد متكيّف (0.2% من حجم 24h، بين 10k و 250k$) — شراء/بيع/صافي آخر 15 دقيقة |
| 3 | **Fear & Greed Filter** | `v5ext/sentiment.py` | من alternative.me. في الخوف الشديد/الطمع الشديد: `adjust` يرفع العتبة 3 نقاط، `block` يمنع، `off` تسجيل فقط |
| 4 | **XGBoost Model** | `v5ext/ml_model.py` | احتمال "TP1 قبل SL" من 34 feature (نفس حسابات `analyze()`). تدريب walk-forward + embargo. لو الـ AUC خارج العينة < 0.55 الماسح بيتجاهله تلقائيًا |
| 5 | **Bayesian Score Adjustment** | `v5ext/bayes.py` | Beta posterior لكل lane (backtest كـ prior محدود الوزن + النتائج الحية) × Likelihood Ratios متعلّمة لـ OBI/Whales/OI/F&G → احتمال نهائي + Expected R |
| 6 | **Dashboard Web** | `dashboard/` | Flask محلي **أو** صفحة ثابتة `docs/index.html` لـ GitHub Pages |
| 7 | **Backtesting Engine مستقل** | `v5ext/backtest_engine.py` | يعيد تشغيل كل شمعة 5m تاريخية عبر `analyze()` الأصلية + نفس التأكيدين والـ cooldown والـ BTC guard + **نفس أنواع الفرص وإدارة الأهداف الحية** + رسوم وانزلاق |
| 8 | **Portfolio Simulator** | `v5ext/portfolio.py` | حجم صفقة حسب المخاطرة ومسافة الـ SL، حد أقصى للمراكز المتزامنة، منحنى رأس مال، Max DD، أرباح شهرية |
| 9 | **Auto Parameter Optimization** | `v5ext/optimizer.py` | 6 معاملات بس (عشان ما يحفظش الماضي). Random search على فترة train، وتقييم مرة واحدة على فترة test (30 صفقة على الأقل). بيتطبّق بس لو كسب الإعدادات الحالية و `--apply` |

### رسالة الإشارة (عربي)
نوع الفرصة: سريعة (Scalping) / خلال اليوم (Intraday) / متعددة الأيام (Swing)، سعر الدخول ووقف الخسارة و3 أهداف، مدة المتابعة، أسباب الظهور، ميزان الأوامر كنسبة شراء/بيع، وتحذيرات. الملف: `v5ext/telegram_ar.py`.

| النوع | مدة المتابعة (= مهلة التتبع الفعلية) | الوقف |
|---|---|---|
| سريعة | حتى 90 دقيقة | ATR 5m × 1.5 (2.5%–6%) |
| خلال اليوم | حتى 8 ساعات | ثوابت الماسح (ATR 5m) |
| متعددة الأيام | حتى 3 أيام | **ATR ساعة** × 2.5 (5%–15%) |

### نسبة النجاح — من غير أرقام مضللة
- النسبة المعروضة في كل إشارة = **نتيجة إشارات البوت الفعلية** (كام صفقة كسبت من كام اتقفلت، آخر 30 يوم + لنفس النوع والمسار). قبل 5 صفقات مقفولة بيكتب "لسه بيجمع بيانات".
- اتشال: "ثقة عالية"، الـ Expected R، والتقدير من الـ Score.
- تقدير النموذج بيظهر **بس** لو اتعاير (Isotonic) وعدّى اختبار المعايرة على بيانات ما اتشافتش (فرق ≤ 10 نقط بين المتوقع والفعلي)، ومكتوب جنبه "تقديري".
- النموذج اتشال منه الإعداد اللي كان بيرفع الاحتمالات (`scale_pos_weight`).

### متابعة الصفقات والتنبيهات (`v5ext/trade_manager.py`)
البوت بيتابع كل عملة بعتها على شموع الدقيقة لحد ما الصفقة تتقفل، وبيبعت:
- 🎯 **TP1** → بيع ثلث وحرّك الوقف للدخول · 🎯🎯 **TP2** → ثلث تاني والوقف لـ TP1 · 🏆 **TP3** → بيع الباقي
- 🚨 **الإشارة بتفشل** — السعر قطع نص المسافة للوقف قبل TP1 → "يُفضّل البيع فورًا"
- ❌ **ضرب الوقف** → "بيع فورًا" · 🔔 رجوع للتعادل / لـ TP1 · ⏱ انتهاء المدة
- كل رسالة إغلاق فيها النتيجة الصافية بعد الرسوم.
- مفيش إشارة جديدة لنفس العملة طول ما صفقتها مفتوحة.

### تقارير المكسب والخسارة (`v5ext/performance.py`)
تيليجرام تلقائيًا بتوقيت القاهرة: **يومي** (أول دورة بعد نص الليل)، **أسبوعي** (السبت، عن الأسبوع اللي فات)، **شهري** (أول الشهر). فيه: عدد الإشارات، الناجحة/الخاسرة، نسبة النجاح، صافي %، وقيمتها لو دخلت بـ 100$ في كل إشارة، وأفضل/أسوأ صفقة. ونفس الجداول في الـ Dashboard.
الحساب بيفترض إنك نفّذت الخطة المكتوبة (ثلث عند كل هدف + تحريك الوقف). لو بعت بدري على تنبيه الفشل، نتيجتك الفعلية هتبقى أحسن من المسجّلة.

الماسح الأصلي `binance_early_pump_v5.py` اتعدّل بأقل قدر: لو مجلد `v5ext` فيه أي مشكلة الماسح بيكمّل زي V5 بالظبط.

## التشغيل

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt   # XGBoost (اختياري — من غيره بيستخدم بديل scikit-learn)

# الماسح (زي قبل)
python binance_early_pump_v5.py --once

# خط البحث الكامل: تحميل بيانات → backtest → تدريب XGBoost → Bayesian → optimizer → محفظة → dashboard
python research.py all --days 30 --top 60

# خطوات منفصلة
python research.py backtest --days 45 --symbols INJUSDT,APTUSDT,FETUSDT
python research.py train
python research.py optimize --trials 300 --apply
python research.py portfolio --capital 1000 --risk 0.01 --max-positions 5
python research.py dashboard

# اختبار كامل بدون إنترنت (بيانات وهمية — الأرقام مالهاش أي معنى)
python research.py all --synthetic --days 20 --top 10
```

### الـ Dashboard
- **محلي:** `pip install -r requirements-dashboard.txt` ثم `python dashboard/app.py` وافتح `http://127.0.0.1:8050`
- **أونلاين مجانًا:** Settings → Pages → Branch: `main` / Folder: `/docs`. الماسح بيحدّث الصفحة تقريبًا كل 30 دقيقة.

### GitHub Actions
- `scanner.yml` — زي ما هو + خطوة تحدّث إحصائيات Bayesian والمحفظة الحية والـ dashboard كل ~30 دقيقة.
- `research.yml` — **جديد**: أسبوعيًا (الأحد) أو يدويًا من Actions → Run workflow. فيه خيار `apply` لتطبيق العتبات المحسّنة.

## متغيرات البيئة الجديدة

| المتغير | الافتراضي | المعنى |
|---|---|---|
| `FNG_MODE` | `adjust` | `adjust` / `block` / `off` |
| `FNG_EXTREME_FEAR` / `FNG_EXTREME_GREED` | 20 / 80 | حدود المناطق المتطرفة |
| `ML_ENABLED` / `ML_GATE` | 1 / 1 | تشغيل النموذج / منع الإشارات تحت العتبة |
| `ML_MIN_PROB` | -1 | عتبة يدوية (-1 = العتبة اللي اختارها التدريب) |
| `ML_MIN_AUC_TO_USE` | 0.55 | أقل AUC عشان النموذج يتستخدم |
| `BAYES_GATE` | 0 | لو 1: يمنع لو Expected R < `BAYES_MIN_EXPECTED_R` |
| `WHALE_ENABLED` | 1 | |
| `USE_OPTIMIZED` | 1 | تحميل `state/optimized_thresholds.json` |
| `BT_FEE_PCT` / `BT_SLIPPAGE_PCT` | 0.10 / 0.05 | لكل جهة في الـ backtest |

ترتيب تحميل العتبات: قيم الكود ← `optimized_thresholds.json` (Optimizer) ← `auto_thresholds.json` (Self-learning بخطوات صغيرة).

## حدود لازم تعرفها
- **مفيش order book أو whales تاريخي مجاني** → الـ backtest والـ XGBoost مبنيين على features الشموع + BTC + Fear&Greed. الـ OBI/Whales/OI بيتسجلوا مع كل إشارة حية، والـ Bayesian بيتعلّم أوزانهم تدريجيًا من النتائج الحية (بعد 20 إشارة محسومة على الأقل لكل مؤشر). قبل كده تأثيرهم = عرض فقط + منع حالة واحدة واضحة (جدار بيع قريب مع OBI < −0.3).
- فلتر شمعة الـ 1m وتأكيد Bybit/OKX مش متطبقين في الـ backtest.
- SL و TP في نفس شمعة 5m ← بيتحسب SL (متحفّظ).
- اختيار العملات حسب حجم النهارده ← survivorship bias. استخدم `--skip-top` أو `--symbols` لتنويع العينة.
- order book لحظة واحدة ممكن يتلاعب فيه (spoofing)؛ عشان كده الوزن الأُسّي ومنع "جدار + OBI سالب" بس.
