def scenarios(",
    "def build_milp(",
    "def solve_with_gap(",
    "def report(",
]

for fn in os.listdir(ROOT):
    if not fn.endswith(".ipynb"):
        continue

    path = os.path.join(ROOT, fn)

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code = ""

        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                code += "".join(cell["source"]) + "\n"

        score = sum(t in code for t in targets)

        if score > 0:
            print("="*80)
            print(fn)
            print("matches:", score, "/", len(targets))
            print("="*80)

    except Exception as e:
        pass

!fuse-overlayfs -u /content/drive || umount -f /content/drive || true
!rm -rf /content/drive

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# =====================================================================
# الخلية 1 — التهيئة والمعاملات
# =====================================================================
!pip install -q pulp

from google.colab import drive
drive.mount('/content/drive')

import os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

OUT = "results"
os.makedirs(OUT, exist_ok=True)
FORCE = False          # True = أعد حساب كل مرحلة متجاهلاً الملفات المحفوظة

CFG = dict(
    scada_csv       = "/content/drive/MyDrive/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv",
    hachmann_csv    = "/content/drive/MyDrive/hachmann_digitized.csv",
    governorates    = ["Anbar", "Babil", "Kirkuk", "Najaf", "Wasit"],
    thermal_regimes = [10.5, 17.5, 25.5, 32.8, 40.5, 46.4],   # يُعاد حسابها في الخلية 6
    regime_pcts     = [10, 30, 50, 70, 90, 99],
    configs         = ["S1", "S2", "S3", "S4", "S5"],
)

# --- معاملات النموذج (المعادلات 1-18) -------------------------------
T_HORIZON     = 24         # أفق التخطيط بالساعات
N_SCENARIOS   = 20         # |Omega|
K_SIGMA       = 1.0        # P_max(f) = mu + k*sigma                    (1)
C_CURT        = 1.2        # كلفة القطع                                 (2)
C_RAMP        = 0.4        # كلفة التغير                                (2)
C_DISC        = 0.2        # سعر عدم الراحة                             (2, 18)
EPS_FAIR      = 0.05       # تسامح العدالة                              (16-17)
TEMP_BINS     = [-np.inf, 30, 35, 40, 45, np.inf]    # حدود الحاويات الحرارية
LAMBDA_BY_BIN = [0.30, 0.45, 0.60, 0.70, 0.80]       # lambda_b -> 1.30x .. 1.80x

# --- السولفر ---------------------------------------------------------
TIME_LIMIT    = 300        # ثانية لكل نموذج
GAP_REL       = 0.03       # فجوة MIP نسبية 3%

# --- التنبؤ ----------------------------------------------------------
LOOKBACK      = 24
TEST_DAYS     = 30
SEED          = 42

LAMBDA_RANGES = {
    "conservative": [0.10, 0.20, 0.30, 0.40, 0.50],   # 1.10x - 1.50x
    "baseline":     LAMBDA_BY_BIN,                    # 1.30x - 1.80x
    "aggressive":   [0.50, 0.60, 0.75, 0.90, 1.00],   # 1.50x - 2.00x
}
SCENARIO_GRID = [10, 20, 30, 50]
EPS_GRID      = [0.03, 0.05, 0.10]

rng_global = np.random.default_rng(SEED)

def stage(name, outfile):
    """يتخطى المرحلة إن كان ملفها موجوداً، ما لم يكن FORCE = True"""
    path = f"{OUT}/{outfile}"
    def deco(fn):
        def wrapped(*a, **kw):
            if os.path.exists(path) and not FORCE:
                print(f"[تخطٍ] {name} -- {path} موجود")
                return pd.read_csv(path)
            print(f"[تشغيل] {name}")
            t0 = time.time()
            df = fn(*a, **kw)
            df.to_csv(path, index=False)
            print(f"[تم] {name} في {time.time()-t0:,.0f}ث -> {path}")
            return df
        return wrapped
    return deco

print("تم تحميل الإعدادات")
print(f"  السيناريوهات  = {N_SCENARIOS}")
print(f"  الحد الزمني   = {TIME_LIMIT} ثانية")
print(f"  فجوة MIP      = {GAP_REL}")
print(f"  ملف البيانات  = {CFG['scada_csv']}")
print(f"  موجود؟        = {os.path.exists(CFG['scada_csv'])}")

# =====================================================================
# الخلية 2 — تحميل سجل SCADA والتحقق من سلامته
# =====================================================================
# =====================================================================
def load_scada():
    raw = pd.read_csv(CFG["scada_csv"])
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    counts = raw.Governorate.value_counts()
    problems = []
    if len(raw) != 206040:
        problems.append(f"المتوقع 206,040 صفاً، الموجود {len(raw):,}")
    if len(counts) != 5:
        problems.append(f"المتوقع 5 محافظات، الموجود {len(counts)}")
    if len(counts) and counts.min() != 41208:
        problems.append(f"المتوقع 41,208 صفاً لكل محافظة، الأدنى {counts.min():,}")
    dups = raw.duplicated(subset=["Governorate", "Timestamp"]).sum()
    if dups:
        problems.append(f"{dups} صفاً مكرراً في (المحافظة، الوقت)")
    if problems:
        raise ValueError("ملف SCADA لم يجتز فحص السلامة:\n  - " +
                         "\n  - ".join(problems) +
                         f"\nالأعداد: {counts.to_dict()}")
    df = raw.rename(columns={"Governorate": "gov", "Timestamp": "ts",
                             "Demand_Value": "demand_mw", "temperature_2m": "temp_c"})
    return df.sort_values(["gov", "ts"]).reset_index(drop=True)

SCADA = load_scada()

print(f"عدد الصفوف        : {len(SCADA):,}")
print(f"المحافظات         : {sorted(SCADA.gov.unique())}")
print(f"الفترة            : {SCADA.ts.min()}  ->  {SCADA.ts.max()}")
print(f"الحرارة (°م)      : {SCADA.temp_c.min():.1f} .. {SCADA.temp_c.max():.1f}"
      f"   ({SCADA.temp_c.nunique():,} قيمة مميزة)")
print(f"الطلب (MW)        : {SCADA.demand_mw.min():.1f} .. {SCADA.demand_mw.max():.1f}")
print(f"الأعمدة           : {list(SCADA.columns)}")
print(f"قيم مفقودة        : {SCADA[['demand_mw','temp_c','Supply_Hours']].isna().sum().to_dict()}")

print("\n--- إحصاءات لكل محافظة ---")
display(SCADA.groupby("gov").agg(n=("demand_mw","size"),
                                 mu=("demand_mw","mean"),
                                 sigma=("demand_mw","std"),
                                 t_min=("temp_c","min"),
                                 t_max=("temp_c","max"),
                                 supply_h=("Supply_Hours","mean")).round(2))

print("\n--- المئينات الحرارية (منها تُشتق الأنظمة الستة) ---")
for p, v in zip(CFG["regime_pcts"], np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])):
    print(f"  المئين {p:>2} : {v:6.2f} °م")

# =====================================================================# 2. Data# =====================================================================def load_scada():    raw = pd.read_csv(CFG["scada_csv"])    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])    counts = raw.Governorate.value_counts()    problems = []    if len(raw) != 206040:        problems.append(f"expected 206,040 rows, got {len(raw):,}")    if len(counts) != 5:        problems.append(f"expected 5 governorates, got {len(counts)}")    if len(counts) and counts.min() != 41208:        problems.append(f"expected 41,208 rows per governorate, min is {counts.min():,}")    dups = raw.duplicated(subset=["Governorate", "Timestamp"]).sum()    if dups:        problems.append(f"{dups} duplicate (Governorate, Timestamp) rows")    if problems:        raise ValueError("SCADA file failed integrity checks:\n  - " +                         "\n  - ".join(problems) +                         f"\nCounts: {counts.to_dict()}")    df = raw.rename(columns={"Governorate": "gov", "Timestamp": "ts",                             "Demand_Value": "demand_mw", "temperature_2m": "temp_c"})    return df.sort_values(["gov", "ts"]).reset_index(drop=True)SCADA = load_scada()print(f"rows              : {len(SCADA):,}")print(f"governorates      : {sorted(SCADA.gov.unique())}")print(f"period            : {SCADA.ts.min()}  ->  {SCADA.ts.max()}")print(f"temperature (degC): {SCADA.temp_c.min():.1f} .. {SCADA.temp_c.max():.1f}"      f"  ({SCADA.temp_c.nunique():,} distinct values)")print(f"demand (MW)       : {SCADA.demand_mw.min():.1f} .. {SCADA.demand_mw.max():.1f}")display(SCADA.groupby("gov").agg(n=("demand_mw","size"),                                 mu=("demand_mw","mean"),                                 sigma=("demand_mw","std"),                                 t_min=("temp_c","min"),                                 t_max=("temp_c","max")).round(2))

# =====================================================================
# الخلية 3 — تدريب BiLSTM لكل محافظة (حتمي)
# =====================================================================
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
tf.config.experimental.enable_op_determinism()      # <-- الجديد
from tensorflow import keras
from numpy.lib.stride_tricks import sliding_window_view as swv
tf.keras.utils.set_random_seed(SEED)
print("GPU:", tf.config.list_physical_devices('GPU'), flush=True)

def make_sequences(demand_n, temp_n, lookback=LOOKBACK):
    X = np.stack([np.stack([demand_n[i-lookback:i], temp_n[i-lookback:i]], -1)
                  for i in range(lookback, len(demand_n))]).astype("float32")
    y = demand_n[lookback:].astype("float32")
    return X, y

def metrics(y, yh):
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(RMSE_MW=float(np.sqrt(np.mean((y - yh) ** 2))),
                MAE_MW=float(np.mean(np.abs(y - yh))),
                R2=float(1 - ss_res / ss_tot),
                MAPE_pct=float(np.mean(np.abs((y - yh) / np.maximum(y, 1e-6))) * 100))

def train_one(gdf, gov, arch="BiLSTM"):
    d = gdf.demand_mw.values.astype("float32")
    t = gdf.temp_c.values.astype("float32")
    n_train = len(d) - TEST_DAYS * 24
    dm, ds = d[:n_train].mean(), d[:n_train].std()
    tm, ts_ = t[:n_train].mean(), t[:n_train].std()
    X, y = make_sequences((d - dm) / ds, (t - tm) / ts_)
    split = n_train - LOOKBACK
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    core = {"BiLSTM": layers.Bidirectional(layers.LSTM(64)),
            "LSTM":   layers.LSTM(64),
            "GRU":    layers.GRU(64)}[arch]
    model = keras.Sequential([layers.Input((LOOKBACK, 2)), core,
                              layers.Dense(32, activation="relu"), layers.Dense(1)])
    model.compile(keras.optimizers.Adam(1e-3), loss="mse")
    hist = model.fit(Xtr, ytr, validation_split=0.1, epochs=25, batch_size=1024,
                     callbacks=[keras.callbacks.EarlyStopping(patience=4,
                                restore_best_weights=True)], verbose=0)
    yh = model.predict(Xte, verbose=0).flatten() * ds + dm
    yt = yte * ds + dm
    m = metrics(yt, yh); m.update(Governorate=gov, epochs=len(hist.history["loss"]))
    fc = pd.DataFrame({"Governorate": gov,
                       "Timestamp": gdf.ts.values[n_train:][:len(yt)],
                       "Demand_Actual": yt, "Demand_Forecast_BiLSTM": yh,
                       "temperature_2m": gdf.temp_c.values[n_train:][:len(yt)]})
    return m, fc

@stage("تدريب BiLSTM لكل محافظة", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)

# =====================================================================
# الخلية 3 — تدريب BiLSTM لكل محافظة (مع إظهار التقدم)
# =====================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy.lib.stride_tricks import sliding_window_view as swv
tf.keras.utils.set_random_seed(SEED)
print("GPU:", tf.config.list_physical_devices('GPU'), flush=True)

def make_sequences(demand_n, temp_n, lookback=LOOKBACK):
    X = np.stack([swv(demand_n, lookback)[:-1], swv(temp_n, lookback)[:-1]], -1)
    return X.astype("float32"), demand_n[lookback:].astype("float32")

def metrics(y, yh):
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(RMSE_MW=float(np.sqrt(np.mean((y - yh) ** 2))),
                MAE_MW=float(np.mean(np.abs(y - yh))),
                R2=float(1 - ss_res / ss_tot),
                MAPE_pct=float(np.mean(np.abs((y - yh) / np.maximum(y, 1e-6))) * 100))

class Tick(keras.callbacks.Callback):
    """يطبع كل 5 حقب حتى لا يبدو التدريب متوقفاً"""
    def __init__(self, gov): self.gov = gov
    def on_epoch_end(self, ep, logs=None):
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"      {self.gov} حقبة {ep+1:2d}  loss={logs['loss']:.5f} "
                  f"val={logs['val_loss']:.5f}", flush=True)

def train_one(gdf, gov, arch="BiLSTM"):
    d = gdf.demand_mw.values.astype("float32")
    t = gdf.temp_c.values.astype("float32")
    n_train = len(d) - TEST_DAYS * 24
    dm, ds = d[:n_train].mean(), d[:n_train].std()
    tm, ts_ = t[:n_train].mean(), t[:n_train].std()
    X, y = make_sequences((d - dm) / ds, (t - tm) / ts_)
    split = n_train - LOOKBACK
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    print(f"   {gov}: تدريب {len(Xtr):,} / اختبار {len(Xte):,}", flush=True)

    core = {"BiLSTM": layers.Bidirectional(layers.LSTM(64)),
            "LSTM":   layers.LSTM(64),
            "GRU":    layers.GRU(64)}[arch]
    model = keras.Sequential([layers.Input((LOOKBACK, 2)), core,
                              layers.Dense(32, activation="relu"), layers.Dense(1)])
    model.compile(keras.optimizers.Adam(1e-3), loss="mse")
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_split=0.1, epochs=25, batch_size=1024,
                     callbacks=[keras.callbacks.EarlyStopping(patience=4,
                                restore_best_weights=True), Tick(gov)], verbose=0)
    yh = model.predict(Xte, verbose=0).flatten() * ds + dm
    yt = yte * ds + dm
    m = metrics(yt, yh)
    m.update(Governorate=gov, epochs=len(hist.history["loss"]))
    print(f"   {gov}: انتهى في {time.time()-t0:.0f}ث  R2={m['R2']:.4f}  "
          f"RMSE={m['RMSE_MW']:.2f}", flush=True)
    fc = pd.DataFrame({"Governorate": gov,
                       "Timestamp": gdf.ts.values[n_train:][:len(yt)],
                       "Demand_Actual": yt, "Demand_Forecast_BiLSTM": yh,
                       "temperature_2m": gdf.temp_c.values[n_train:][:len(yt)]})
    return m, fc

@stage("تدريب BiLSTM لكل محافظة", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)
        mets.append(m); fcs.append(fc)
    pd.concat(fcs, ignore_index=True).to_csv(f"{OUT}/bilstm_forecast_hourly.csv", index=False)
    return pd.DataFrame(mets)[["Governorate","RMSE_MW","MAE_MW","R2","MAPE_pct","epochs"]]

BILSTM = run_bilstm()
FORECAST = pd.read_csv(f"{OUT}/bilstm_forecast_hourly.csv")
display(BILSTM.round(4))
print(f"\nالجدول 3 -> R² {BILSTM.R2.min():.3f}-{BILSTM.R2.max():.3f} | "
      f"RMSE {BILSTM.RMSE_MW.min():.2f}-{BILSTM.RMSE_MW.max():.2f} MW | "
      f"صفوف التنبؤ {len(FORECAST):,}", flush=True)

# =====================================================================
# الخلية 3 (كاملة) — المسار + الحتمية + التدريب
# =====================================================================
import os, time
OUT = "/content/drive/MyDrive/segan_results"
os.makedirs(OUT, exist_ok=True)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from tensorflow import keras
from tensorflow.keras import layers
from numpy.lib.stride_tricks import sliding_window_view as swv
tf.keras.utils.set_random_seed(SEED)

# احذف ناتج التشغيل غير الحتمي السابق
for f in ("bilstm_metrics.csv", "bilstm_forecast_hourly.csv"):
    p = f"{OUT}/{f}"
    if os.path.exists(p): os.remove(p); print("حُذف", p, flush=True)

print("OUT  =", OUT, flush=True)
print("GPU  =", tf.config.list_physical_devices('GPU'), flush=True)
print("حتمي =", os.environ.get("TF_DETERMINISTIC_OPS"), flush=True)
print("-"*55, flush=True)

def stage(name, outfile):
    def deco(fn):
        def wrapped(*a, **kw):
            path = f"{OUT}/{outfile}"
            if os.path.exists(path) and not FORCE:
                print(f"[تخطٍ] {name} -- {path} موجود", flush=True); return pd.read_csv(path)
            print(f"[تشغيل] {name}  ->  {path}", flush=True)
            t0 = time.time(); df = fn(*a, **kw)
            if df is None: raise RuntimeError(f"{name}: لم تُرجع جدولاً")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[تم] {name} في {time.time()-t0:,.0f}ث -> {path}", flush=True)
            return df
        return wrapped
    return deco

def make_sequences(demand_n, temp_n, lookback=LOOKBACK):
    X = np.stack([swv(demand_n, lookback)[:-1], swv(temp_n, lookback)[:-1]], -1)
    return X.astype("float32"), demand_n[lookback:].astype("float32")

def metrics(y, yh):
    ss_res = np.sum((y - yh) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(RMSE_MW=float(np.sqrt(np.mean((y - yh) ** 2))),
                MAE_MW=float(np.mean(np.abs(y - yh))),
                R2=float(1 - ss_res / ss_tot),
                MAPE_pct=float(np.mean(np.abs((y - yh) / np.maximum(y, 1e-6))) * 100))

class Tick(keras.callbacks.Callback):
    def __init__(self, gov): self.gov = gov
    def on_epoch_end(self, ep, logs=None):
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"      {self.gov} حقبة {ep+1:2d}  loss={logs['loss']:.5f} "
                  f"val={logs['val_loss']:.5f}", flush=True)

def train_one(gdf, gov, arch="BiLSTM"):
    d = gdf.demand_mw.values.astype("float32"); t = gdf.temp_c.values.astype("float32")
    n_train = len(d) - TEST_DAYS * 24
    dm, ds = d[:n_train].mean(), d[:n_train].std()
    tm, ts_ = t[:n_train].mean(), t[:n_train].std()
    X, y = make_sequences((d - dm) / ds, (t - tm) / ts_)
    split = n_train - LOOKBACK
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    print(f"   {gov}: تدريب {len(Xtr):,} / اختبار {len(Xte):,}", flush=True)
    core = {"BiLSTM": layers.Bidirectional(layers.LSTM(64)),
            "LSTM": layers.LSTM(64), "GRU": layers.GRU(64)}[arch]
    model = keras.Sequential([layers.Input((LOOKBACK, 2)), core,
                              layers.Dense(32, activation="relu"), layers.Dense(1)])
    model.compile(keras.optimizers.Adam(1e-3), loss="mse")
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_split=0.1, epochs=25, batch_size=1024,
                     callbacks=[keras.callbacks.EarlyStopping(patience=4,
                                restore_best_weights=True), Tick(gov)], verbose=0)
    yh = model.predict(Xte, verbose=0).flatten() * ds + dm
    yt = yte * ds + dm
    m = metrics(yt, yh); m.update(Governorate=gov, epochs=len(hist.history["loss"]))
    print(f"   {gov}: انتهى في {time.time()-t0:.0f}ث  R2={m['R2']:.4f}  "
          f"RMSE={m['RMSE_MW']:.2f}", flush=True)
    fc = pd.DataFrame({"Governorate": gov, "Timestamp": gdf.ts.values[n_train:][:len(yt)],
                       "Demand_Actual": yt, "Demand_Forecast_BiLSTM": yh,
                       "temperature_2m": gdf.temp_c.values[n_train:][:len(yt)]})
    return m, fc

@stage("تدريب BiLSTM لكل محافظة", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)
        mets.append(m); fcs.append(fc)
    pd.concat(fcs, ignore_index=True).to_csv(f"{OUT}/bilstm_forecast_hourly.csv", index=False)
    return pd.DataFrame(mets)[["Governorate","RMSE_MW","MAE_MW","R2","MAPE_pct","epochs"]]

BILSTM = run_bilstm()
FORECAST = pd.read_csv(f"{OUT}/bilstm_forecast_hourly.csv")
display(BILSTM.round(4))
print(f"\nالجدول 3 -> R² {BILSTM.R2.min():.3f}-{BILSTM.R2.max():.3f} | "
      f"RMSE {BILSTM.RMSE_MW.min():.2f}-{BILSTM.RMSE_MW.max():.2f} MW | "
      f"صفوف التنبؤ {len(FORECAST):,}", flush=True)

# توثيق البيئة لحزمة إعادة الإنتاج
import platform, sys, json
env = {
    "python": sys.version.split()[0],
    "tensorflow": tf.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "gpu": [d.name for d in tf.config.list_physical_devices('GPU')],
    "gpu_details": [tf.config.experimental.get_device_details(d)
                    for d in tf.config.list_physical_devices('GPU')],
    "platform": platform.platform(),
    "seed": SEED,
    "deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
}
try:
    import pulp; env["pulp"] = pulp.__version__
except ImportError: pass
json.dump(env, open(f"{OUT}/environment.json", "w"), indent=2, default=str)
print(json.dumps(env, indent=2, default=str))

# =====================================================================
# الخلية 4 — خط الأساس الوطني المجمّع + الارتباط بين المحافظات
# =====================================================================
import itertools

@stage("خط الأساس المجمّع + الارتباط", "forecast_baselines.csv")
def run_pooled():
    # --- نموذج واحد مُدرَّب على كل المحافظات مجتمعة
    parts, idx = [], []
    for g in CFG["governorates"]:
        gdf = SCADA[SCADA.gov == g].reset_index(drop=True)
        d = gdf.demand_mw.values.astype("float32"); t = gdf.temp_c.values.astype("float32")
        n_train = len(d) - TEST_DAYS * 24
        dm, ds = d[:n_train].mean(), d[:n_train].std()
        tm, ts_ = t[:n_train].mean(), t[:n_train].std()
        X, y = make_sequences((d - dm) / ds, (t - tm) / ts_)
        parts.append((X, y, n_train - LOOKBACK, dm, ds)); idx.append(g)

    Xtr = np.concatenate([p[0][:p[2]] for p in parts])
    ytr = np.concatenate([p[1][:p[2]] for p in parts])
    print(f"   المجمّع: تدريب {len(Xtr):,} تسلسلاً من {len(idx)} محافظات", flush=True)

    model = keras.Sequential([layers.Input((LOOKBACK, 2)),
                              layers.Bidirectional(layers.LSTM(64)),
                              layers.Dense(32, activation="relu"), layers.Dense(1)])
    model.compile(keras.optimizers.Adam(1e-3), loss="mse")
    t0 = time.time()
    model.fit(Xtr, ytr, validation_split=0.1, epochs=25, batch_size=1024,
              callbacks=[keras.callbacks.EarlyStopping(patience=4,
                         restore_best_weights=True)], verbose=0)
    print(f"   المجمّع: انتهى في {time.time()-t0:.0f}ث", flush=True)

    rows, yt_all, yh_all = [], [], []
    for g, (X, y, split, dm, ds) in zip(idx, parts):
        yh = model.predict(X[split:], verbose=0).flatten() * ds + dm
        yt = y[split:] * ds + dm
        yt_all.append(yt); yh_all.append(yh)
        rows.append(dict(scope=g, model="pooled_BiLSTM", **metrics(yt, yh)))
        print(f"      {g:7} R2={rows[-1]['R2']:.4f}  RMSE={rows[-1]['RMSE_MW']:.2f}", flush=True)
    rows.append(dict(scope="ALL", model="pooled_BiLSTM",
                     **metrics(np.concatenate(yt_all), np.concatenate(yh_all))))
    print(f"   المجمّع على كل ساعات الاختبار: R2={rows[-1]['R2']:.4f}", flush=True)

    # --- متوسط الارتباط الثنائي للطلب بين المحافظات
    wide = SCADA.pivot_table(index="ts", columns="gov", values="demand_mw")
    cm = wide.corr()
    pairs = list(itertools.combinations(CFG["governorates"], 2))
    pair_r = [cm.loc[a, b] for a, b in pairs]
    print("\n   الارتباط الثنائي:", flush=True)
    for (a, b), r_ in zip(pairs, pair_r):
        print(f"      {a:7} - {b:7} r={r_:+.4f}", flush=True)
    print(f"   المتوسط = {np.mean(pair_r):.4f}", flush=True)
    rows.append(dict(scope="inter_governorate", model="pearson_r_mean",
                     RMSE_MW=np.nan, MAE_MW=np.nan, R2=float(np.mean(pair_r)),
                     MAPE_pct=np.nan))
    return pd.DataFrame(rows)

POOLED = run_pooled()
display(POOLED.round(4))

pooled_all = POOLED[POOLED.scope == "ALL"].R2.iloc[0]
inter_r    = POOLED[POOLED.scope == "inter_governorate"].R2.iloc[0]
print(f"\n--- للمقارنة مع ما تدّعيه الورقة ---", flush=True)
print(f"خط الأساس المجمّع : R² = {pooled_all:.4f}   (الورقة تقول 0.768)", flush=True)
print(f"الارتباط المتوسط  : r  = {inter_r:.4f}   (الورقة تقول 0.409)", flush=True)
print(f"الإقليمي يتفوق؟   : {BILSTM.R2.min():.4f} > {pooled_all:.4f} = "
      f"{BILSTM.R2.min() > pooled_all}", flush=True)

# --- توثيق البيئة لحزمة إعادة الإنتاج
import platform, sys, json
env = {"python": sys.version.split()[0], "tensorflow": tf.__version__,
       "numpy": np.__version__, "pandas": pd.__version__,
       "gpu": [tf.config.experimental.get_device_details(d)
               for d in tf.config.list_physical_devices('GPU')],
       "platform": platform.platform(), "seed": SEED,
       "deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS")}
try:
    import pulp; env["pulp"] = pulp.__version__
except ImportError: pass
json.dump(env, open(f"{OUT}/environment.json", "w"), indent=2, default=str)
print("\nenvironment.json:", json.dumps(env, indent=2, default=str), flush=True)

# =====================================================================
# الخلية 5 — مقارنة المعماريات: BiLSTM / LSTM / GRU / XGBoost
# =====================================================================
import xgboost as xgb
print("xgboost", xgb.__version__, flush=True)

BENCH_EPOCHS = 30
BENCH_BATCH  = 256
N_BOOT       = 2000

@stage("مقارنة المعماريات", "architecture_benchmark.csv")
def run_benchmark():
    rows, errs = [], {}
    rng_b = np.random.default_rng(SEED)
    for g in CFG["governorates"]:
        gdf = SCADA[SCADA.gov == g].reset_index(drop=True)
        d = gdf.demand_mw.values.astype("float32"); t = gdf.temp_c.values.astype("float32")
        n_train = len(d) - TEST_DAYS * 24
        dm, ds = d[:n_train].mean(), d[:n_train].std()
        tm, ts_ = t[:n_train].mean(), t[:n_train].std()
        X, y = make_sequences((d - dm) / ds, (t - tm) / ts_)
        split = n_train - LOOKBACK
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
        yt = yte * ds + dm

        for arch in ["BiLSTM", "LSTM", "GRU", "XGBoost"]:
            tf.keras.utils.set_random_seed(SEED)          # نفس البداية لكل معمارية
            t0 = time.time()
            if arch == "XGBoost":
                m = xgb.XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist")
                m.fit(Xtr.reshape(len(Xtr), -1), ytr)
                yh = m.predict(Xte.reshape(len(Xte), -1)) * ds + dm
            else:
                core = {"BiLSTM": layers.Bidirectional(layers.LSTM(64)),
                        "LSTM":   layers.LSTM(64),
                        "GRU":    layers.GRU(64)}[arch]
                mdl = keras.Sequential([layers.Input((LOOKBACK, 2)), core, layers.Dense(1)])
                mdl.compile(keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse")
                mdl.fit(Xtr, ytr, epochs=BENCH_EPOCHS, batch_size=BENCH_BATCH, verbose=0)
                yh = mdl.predict(Xte, verbose=0).flatten() * ds + dm

            e = np.abs(yt - yh); errs[(g, arch)] = e
            idx = rng_b.integers(0, len(yt), (N_BOOT, len(yt)))
            bs = np.sqrt(np.mean((yt[idx] - yh[idx]) ** 2, axis=1))
            rows.append(dict(Governorate=g, Model=arch, **metrics(yt, yh),
                             RMSE_ci95_low=float(np.percentile(bs, 2.5)),
                             RMSE_ci95_high=float(np.percentile(bs, 97.5))))
            print(f"   {g:7} {arch:8} RMSE={rows[-1]['RMSE_MW']:6.2f} "
                  f"[{rows[-1]['RMSE_ci95_low']:5.1f}, {rows[-1]['RMSE_ci95_high']:5.1f}]  "
                  f"R2={rows[-1]['R2']:.4f}  ({time.time()-t0:.0f}ث)", flush=True)

    # اختبار ويلكوكسون على الأخطاء المطلقة المزدوجة: BiLSTM مقابل كل بديل
    wrows = []
    for g in CFG["governorates"]:
        for arch in ["LSTM", "GRU", "XGBoost"]:
            _, p = stats.wilcoxon(errs[(g, "BiLSTM")], errs[(g, arch)])
            wrows.append(dict(Governorate=g, pair=f"BiLSTM vs {arch}", wilcoxon_p=float(p),
                              mae_BiLSTM=float(errs[(g, "BiLSTM")].mean()),
                              mae_other=float(errs[(g, arch)].mean())))
    pd.DataFrame(wrows).to_csv(f"{OUT}/architecture_wilcoxon.csv", index=False)
    return pd.DataFrame(rows)

BENCH = run_benchmark()
display(BENCH.round(4))

print("\n--- أدنى RMSE لكل محافظة ---", flush=True)
best = BENCH.loc[BENCH.groupby("Governorate").RMSE_MW.idxmin()][["Governorate","Model","RMSE_MW"]]
display(best.round(2))
print(f"\nمدى R² عبر كل الأزواج : {BENCH.R2.min():.3f} - {BENCH.R2.max():.3f}", flush=True)
print(f"مدى MAPE              : {BENCH.MAPE_pct.min():.2f}% - {BENCH.MAPE_pct.max():.2f}%", flush=True)
gap = (BENCH.groupby("Governorate")
            .apply(lambda x: x[x.Model=="BiLSTM"].RMSE_MW.iloc[0] - x.RMSE_MW.min()))
print(f"أكبر فجوة BiLSTM عن الأفضل : {gap.max():.2f} MW (في {gap.idxmax()})", flush=True)
WIL = pd.read_csv(f"{OUT}/architecture_wilcoxon.csv")
display(WIL.round(6))

# =====================================================================
# الخلية 6 — P_max، الأنظمة الحرارية، النوافذ الممثِّلة، السيناريوهات
# =====================================================================
PIPE = {}

def prepare_pipeline():
    if PIPE: return PIPE
    s = SCADA.copy()
    s["avail"] = s["Supply_Hours"] / 24.0
    g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")
    PIPE["mu"], PIPE["sigma"] = g["mu"].to_dict(), g["sigma"].to_dict()
    PIPE["govs"] = CFG["governorates"]
    PIPE["pmax"] = {f: PIPE["mu"][f] + K_SIGMA * PIPE["sigma"][f] for f in PIPE["govs"]}
    PIPE["arr"] = {f: s[s.gov == f].reset_index(drop=True) for f in PIPE["govs"]}
    PIPE["rel_dev"] = {}
    for f in PIPE["govs"]:
        d = FORECAST[FORECAST.Governorate == f]
        PIPE["rel_dev"][f] = ((d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values) /
                              np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6))
    CFG["thermal_regimes"] = [round(float(x), 1) for x in
                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]
    PIPE["window"] = {}
    for rt in CFG["thermal_regimes"]:
        PIPE["window"][rt] = {}
        for f in PIPE["govs"]:
            gdf = PIPE["arr"][f]
            roll = gdf["temp_c"].rolling(T_HORIZON).mean().values
            cand = np.where(~np.isnan(roll))[0]
            best = cand[np.argmin(np.abs(roll[cand] - rt))]
            sl = gdf.iloc[best - T_HORIZON + 1: best + 1]
            PIPE["window"][rt][f] = dict(demand=sl["demand_mw"].values,
                                         temp=sl["temp_c"].values,
                                         avail=sl["avail"].values,
                                         start=str(sl["ts"].iloc[0]))
    PIPE["scen_cache"] = {}
    return PIPE

def scenarios(rt, n_scen):
    P = prepare_pipeline()
    key = (round(rt, 1), n_scen)
    if key in P["scen_cache"]: return P["scen_cache"][key]
    out = {}
    for gi, f in enumerate(P["govs"]):
        rng = np.random.default_rng([SEED, int(round(rt * 10)), gi, n_scen])
        base = P["window"][rt][f]["demand"]
        rd = P["rel_dev"][f]
        out[f] = np.stack([base * (1 + rng.choice(rd, size=T_HORIZON, replace=True))
                           for _ in range(n_scen)])
    P["scen_cache"][key] = out
    return out
