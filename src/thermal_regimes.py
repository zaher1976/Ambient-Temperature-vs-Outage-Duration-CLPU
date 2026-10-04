import json

NB = "/content/drive/MyDrive/Colab Notebooks/CLPU Mitigation via Temporal Coordination.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):

    if cell["cell_type"] != "code":
        continue

    txt = "".join(cell["source"])

    if (
        "def prepare_pipeline" in txt or
        "def scenarios" in txt or
        "def build_milp" in txt or
        "def solve_with_gap" in txt or
        "def report" in txt
    ):
        print("\n" + "=" * 80)
        print("CELL", i)
        print("=" * 80)
        print(txt)

import os, json

ROOT = "/content/drive/MyDrive/Colab Notebooks"

targets = [
    "prepare_pipeline",
    "def scenarios(",
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

P = prepare_pipeline()
print("الأنظمة الحرارية (مئينات السجل):", CFG["thermal_regimes"], flush=True)
print(f"\nP_max(f) = mu + k*sigma،  k = {K_SIGMA}", flush=True)
for f in P["govs"]:
    print(f"  {f:7} mu={P['mu'][f]:8.2f}  sigma={P['sigma'][f]:7.2f}  "
          f"P_max={P['pmax'][f]:8.2f} MW", flush=True)

print("\nتوزيع بواقي التنبؤ (منها تُولَّد السيناريوهات):", flush=True)
for f in P["govs"]:
    rd = P["rel_dev"][f]
    print(f"  {f:7} n={len(rd):4}  متوسط={rd.mean():+.4f}  "
          f"انحراف={rd.std():.4f}  مدى=[{rd.min():+.3f}, {rd.max():+.3f}]", flush=True)

print("\nالنوافذ الممثِّلة (24 ساعة لكل محافظة لكل نظام):", flush=True)
for rt in CFG["thermal_regimes"]:
    means = {f: round(float(P['window'][rt][f]['temp'].mean()), 1) for f in P["govs"]}
    print(f"  الهدف {rt:>5}°م -> متوسطات فعلية {means}", flush=True)

print("\nتواريخ النوافذ (نظام 46.4°م مثالاً):", flush=True)
rt_hot = CFG["thermal_regimes"][-1]
for f in P["govs"]:
    w = P["window"][rt_hot][f]
    print(f"  {f:7} يبدأ {w['start']}  طلب {w['demand'].mean():7.1f} MW  "
          f"إتاحة {w['avail'].mean():.3f}", flush=True)

sc = scenarios(CFG["thermal_regimes"][0], N_SCENARIOS)
print(f"\nاختبار توليد السيناريوهات: شكل مصفوفة أنبار = {sc['Anbar'].shape} "
      f"(متوقع ({N_SCENARIOS}, {T_HORIZON}))", flush=True)

# =====================================================================
# الخلية 6 (معدّلة) — نافذة واحدة متزامنة لكل نظام حراري
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
    PIPE["rel_dev"] = {f: ((d := FORECAST[FORECAST.Governorate == f]).Demand_Actual.values -
                           d.Demand_Forecast_BiLSTM.values) /
                          np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6)
                       for f in PIPE["govs"]}

    # --- شبكة زمنية مشتركة لكل المحافظات
    wide_t = s.pivot_table(index="ts", columns="gov", values="temp_c")
    wide_t = wide_t.dropna(how="any").sort_index()          # ساعات متاحة للجميع
    PIPE["shared_ts"] = wide_t.index
    print(f"   طوابع زمنية مشتركة بين المحافظات الخمس: {len(wide_t):,} "
          f"من {s.ts.nunique():,}", flush=True)

    national = wide_t.mean(axis=1)                          # متوسط الحرارة الوطني
    roll = national.rolling(T_HORIZON).mean()

    CFG["thermal_regimes"] = [round(float(x), 1) for x in
                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]
    PIPE["window"] = {}
    for rt in CFG["thermal_regimes"]:
        end = roll.sub(rt).abs().idxmin()                   # نهاية أقرب نافذة وطنية
        pos = wide_t.index.get_loc(end)
        ts_win = wide_t.index[pos - T_HORIZON + 1: pos + 1]  # نفس الساعات للجميع
        PIPE["window"][rt] = {}
        for f in PIPE["govs"]:
            gdf = PIPE["arr"][f].set_index("ts").loc[ts_win]
            PIPE["window"][rt][f] = dict(demand=gdf["demand_mw"].values,
                                         temp=gdf["temp_c"].values,
                                         avail=gdf["avail"].values)
        PIPE["window"][rt]["_ts"] = ts_win
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

P = prepare_pipeline()
print("\nالأنظمة الحرارية:", CFG["thermal_regimes"], flush=True)
print(f"\nP_max(f) = mu + k*sigma،  k = {K_SIGMA}", flush=True)
for f in P["govs"]:
    print(f"  {f:7} P_max={P['pmax'][f]:8.2f} MW", flush=True)

print("\nالنوافذ المتزامنة (نفس الساعات لكل المحافظات):", flush=True)
for rt in CFG["thermal_regimes"]:
    ts = P["window"][rt]["_ts"]
    nat = np.mean([P["window"][rt][f]["temp"].mean() for f in P["govs"]])
    per = {f: round(float(P['window'][rt][f]['temp'].mean()), 1) for f in P["govs"]}
    print(f"  {rt:>5}°م  {ts[0]} -> {ts[-1]}  وطني={nat:5.1f}  {per}", flush=True)

sc = scenarios(CFG["thermal_regimes"][0], N_SCENARIOS)
print(f"\nشكل مصفوفة السيناريوهات (أنبار) = {sc['Anbar'].shape}", flush=True)

# =====================================================================
# BOOT CELL — everything needed after a runtime restart, in one paste.
# Run this, then CELL B (cell_B_resumable.py). Nothing is retrained and no
# completed stage is recomputed: every cached result is read back from Drive.
# =====================================================================


# ---------- 1. configuration ----------


# =====================================================================
# 1. Configuration
# =====================================================================
import os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

OUT = "results"
os.makedirs(OUT, exist_ok=True)
FORCE = False          # True = recompute every stage, ignoring cached CSVs

CFG = dict(
    scada_csv       = "NATIONAL_MASTER_MATRIX_CLEAN_v2.csv",
    hachmann_csv    = "hachmann_digitized.csv",
    governorates    = ["Anbar", "Babil", "Kirkuk", "Najaf", "Wasit"],
    thermal_regimes = [10.5, 17.5, 25.5, 32.8, 40.5, 46.4],   # recomputed in stage 6
    regime_pcts     = [10, 30, 50, 70, 90, 99],
    configs         = ["S1", "S2", "S3", "S4", "S5"],
)

# --- MILP parameters (Eq. 1-18 of the manuscript) --------------------
T_HORIZON    = 24          # planning horizon, hours
N_SCENARIOS  = 20          # |Omega|
REPORT_N     = 20          # scenarios used for MaxRamp / disparity / JFI
K_SIGMA      = 1.0         # P_max(f) = mu + k*sigma                      (Eq. 1)
C_CURT       = 1.2         # curtailment cost                             (Eq. 2)
C_RAMP       = 0.4         # ramping cost                                 (Eq. 2)
C_DISC       = 0.2         # discomfort price                             (Eq. 2, 18)
EPS_FAIR     = 0.05        # fairness tolerance                           (Eq. 16-17)
TEMP_BINS    = [-np.inf, 30, 35, 40, 45, np.inf]   # temperature bin edges, degC
LAMBDA_BY_BIN= [0.30, 0.45, 0.60, 0.70, 0.80]      # lambda_b -> 1.30x .. 1.80x

# --- solver ----------------------------------------------------------
TIME_LIMIT   = 300         # seconds per instance
GAP_REL      = 0.03        # 3% relative MIP gap

# --- forecasting -----------------------------------------------------
LOOKBACK     = 24
TEST_DAYS    = 30
SEED         = 42

LAMBDA_RANGES = {
    "conservative": [0.10, 0.20, 0.30, 0.40, 0.50],   # 1.10x - 1.50x
    "baseline":     LAMBDA_BY_BIN,                    # 1.30x - 1.80x
    "aggressive":   [0.50, 0.60, 0.75, 0.90, 1.00],   # 1.50x - 2.00x
}
SCENARIO_GRID = [10, 20, 30, 50]
EPS_GRID      = [0.03, 0.05, 0.10]
CONV_REGIMES  = "extremes"     # convergence / eps studies run at the two extreme regimes

rng_global = np.random.default_rng(SEED)

def stage(name, outfile):
    """Decorator: skip the stage if its output already exists (unless FORCE).
    The path is resolved at call time, so changing OUT takes effect immediately."""
    def deco(fn):
        def wrapped(*a, **kw):
            path = f"{OUT}/{outfile}"
            if os.path.exists(path) and not FORCE:
                print(f"[skip] {name} -- {path} already exists", flush=True)
                return pd.read_csv(path)
            print(f"[run ] {name}  ->  {path}", flush=True)
            t0 = time.time()
            df = fn(*a, **kw)
            if df is None:
                raise RuntimeError(f"{name}: stage returned no table")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[done] {name} in {time.time()-t0:,.0f}s -> {path}", flush=True)
            return df
        return wrapped
    return deco

print("configuration loaded | N_SCENARIOS =", N_SCENARIOS,
      "| time limit =", TIME_LIMIT, "s | gap =", GAP_REL)



# ---------- 1b. Drive path (cell 1 resets OUT, so this must follow it) ----------
OUT = "/content/drive/MyDrive/segan_results"
os.makedirs(OUT, exist_ok=True)
print("OUT =", OUT, "|", len(os.listdir(OUT)), "cached files")

# The dataset lives on Drive. Cell 1 carries a bare filename, which only resolves
# while a copy happens to sit in /content; that copy does not survive a restart.
CFG["scada_csv"] = "/content/drive/MyDrive/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv"
assert os.path.exists(CFG["scada_csv"]), (
    "dataset not found at " + CFG["scada_csv"] + " — check the path or remount Drive")
print("dataset  =", CFG["scada_csv"])


# ---------- 2. data ----------


# =====================================================================
# 2. Data
# =====================================================================
def load_scada():
    raw = pd.read_csv(CFG["scada_csv"])
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    counts = raw.Governorate.value_counts()
    problems = []
    if len(raw) != 206040:
        problems.append(f"expected 206,040 rows, got {len(raw):,}")
    if len(counts) != 5:
        problems.append(f"expected 5 governorates, got {len(counts)}")
    if len(counts) and counts.min() != 41208:
        problems.append(f"expected 41,208 rows per governorate, min is {counts.min():,}")
    dups = raw.duplicated(subset=["Governorate", "Timestamp"]).sum()
    if dups:
        problems.append(f"{dups} duplicate (Governorate, Timestamp) rows")
    if problems:
        raise ValueError("SCADA file failed integrity checks:\n  - " +
                         "\n  - ".join(problems) +
                         f"\nCounts: {counts.to_dict()}")
    df = raw.rename(columns={"Governorate": "gov", "Timestamp": "ts",
                             "Demand_Value": "demand_mw", "temperature_2m": "temp_c"})
    return df.sort_values(["gov", "ts"]).reset_index(drop=True)

SCADA = load_scada()
print(f"rows              : {len(SCADA):,}")
print(f"governorates      : {sorted(SCADA.gov.unique())}")
print(f"period            : {SCADA.ts.min()}  ->  {SCADA.ts.max()}")
print(f"temperature (degC): {SCADA.temp_c.min():.1f} .. {SCADA.temp_c.max():.1f}"
      f"  ({SCADA.temp_c.nunique():,} distinct values)")
print(f"demand (MW)       : {SCADA.demand_mw.min():.1f} .. {SCADA.demand_mw.max():.1f}")
display(SCADA.groupby("gov").agg(n=("demand_mw","size"),
                                 mu=("demand_mw","mean"),
                                 sigma=("demand_mw","std"),
                                 t_min=("temp_c","min"),
                                 t_max=("temp_c","max")).round(2))


# ---------- 3. BiLSTM (cached: reads bilstm_metrics.csv, no retraining) ----------


# =====================================================================
# 3. BiLSTM forecasting
# =====================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)

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

@stage("BiLSTM per governorate", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)
        print(f"  {g:7} RMSE={m['RMSE_MW']:6.2f}  MAE={m['MAE_MW']:6.2f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE_pct']:.2f}%  (epochs={m['epochs']})")
        mets.append(m); fcs.append(fc)
    pd.concat(fcs, ignore_index=True).to_csv(f"{OUT}/bilstm_forecast_hourly.csv", index=False)
    cols = ["Governorate", "RMSE_MW", "MAE_MW", "R2", "MAPE_pct", "epochs"]
    return pd.DataFrame(mets)[cols]

BILSTM = run_bilstm()
FORECAST = pd.read_csv(f"{OUT}/bilstm_forecast_hourly.csv")
display(BILSTM.round(4))
print(f"\nTable 3 -> R2 {BILSTM.R2.min():.3f}-{BILSTM.R2.max():.3f} | "
      f"RMSE {BILSTM.RMSE_MW.min():.2f}-{BILSTM.RMSE_MW.max():.2f} MW")


# ---------- 6. pipeline, synchronous national windows ----------


PIPE = {}

def prepare_pipeline():
    if PIPE:
        return PIPE
    s = SCADA.copy()
    s["avail"] = s["Supply_Hours"] / 24.0
    g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")
    PIPE["mu"], PIPE["sigma"] = g["mu"].to_dict(), g["sigma"].to_dict()
    PIPE["govs"] = CFG["governorates"]
    PIPE["pmax"] = {f: PIPE["mu"][f] + K_SIGMA * PIPE["sigma"][f] for f in PIPE["govs"]}
    PIPE["arr"] = {f: s[s.gov == f].sort_values("ts").reset_index(drop=True)
                   for f in PIPE["govs"]}

    PIPE["rel_dev"] = {}
    for f in PIPE["govs"]:
        d = FORECAST[FORECAST.Governorate == f]
        PIPE["rel_dev"][f] = ((d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values) /
                              np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6))

    # thermal regimes recomputed from the record, not assumed
    CFG["thermal_regimes"] = [round(float(x), 1) for x in
                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]

    # ---- one 24-hour window per regime, the same hours for all governorates ----
    wide_t = (s.pivot_table(index="ts", columns="gov", values="temp_c")
                .dropna(how="any").sort_index())[PIPE["govs"]]
    national = wide_t.mean(axis=1)
    roll = national.rolling(T_HORIZON).mean()
    by_ts = {f: PIPE["arr"][f].set_index("ts") for f in PIPE["govs"]}

    PIPE["window"], PIPE["window_span"] = {}, {}
    for rt in CFG["thermal_regimes"]:
        pos = int(np.nanargmin(np.abs(roll.values - rt)))
        ts_win = wide_t.index[pos - T_HORIZON + 1: pos + 1]
        PIPE["window"][rt] = {}
        for f in PIPE["govs"]:
            sl = by_ts[f].reindex(ts_win)
            assert not sl[["demand_mw", "temp_c", "avail"]].isna().any().any(), \
                f"gap in the record for {f} at regime {rt}"
            PIPE["window"][rt][f] = dict(demand=sl["demand_mw"].values,
                                         temp=sl["temp_c"].values,
                                         avail=sl["avail"].values)
        PIPE["window_span"][rt] = (str(ts_win[0])[:16], str(ts_win[-1])[:16],
                                   float(national.reindex(ts_win).mean()))

    PIPE["scen_cache"] = {}
    PIPE["shared_timestamps"] = len(wide_t)
    return PIPE


def scenarios(rt, n_scen):
    P = prepare_pipeline()
    key = (round(rt, 1), n_scen)
    if key in P["scen_cache"]:
        return P["scen_cache"][key]
    out = {}
    for gi, f in enumerate(P["govs"]):
        rng = np.random.default_rng([SEED, int(round(rt * 10)), gi, n_scen])
        base = P["window"][rt][f]["demand"]
        rd = P["rel_dev"][f]
        out[f] = np.stack([base * (1 + rng.choice(rd, size=T_HORIZON, replace=True))
                           for _ in range(n_scen)])
    P["scen_cache"][key] = out
    return out


P = prepare_pipeline()
print("thermal regimes:", CFG["thermal_regimes"])
print("shared hourly timestamps across all five governorates:", f"{P['shared_timestamps']:,}")
print("\nP_max(f) = mu + k*sigma, k =", K_SIGMA)
for f in P["govs"]:
    print(f"  {f:7} mu={P['mu'][f]:8.2f}  sigma={P['sigma'][f]:7.2f}  P_max={P['pmax'][f]:8.2f} MW")

# ---- state check: the restored pipeline must reproduce the Table 5 windows ----
EXPECTED = {10.5: ("2020-01-08 11:00", 10.5), 17.5: ("2020-11-16 01:00", 17.5),
            25.6: ("2024-04-08 09:00", 25.6), 32.8: ("2024-09-16 16:00", 32.8),
            40.5: ("2022-06-19 17:00", 40.5), 46.4: ("2020-07-29 12:00", 43.0)}
print("\n--- window check against the run that produced Table 5 ---")
bad = []
for rt in CFG["thermal_regimes"]:
    start, end, nat = P["window_span"][rt]
    want_start, want_nat = EXPECTED.get(rt, (None, None))
    ok = (start == want_start) and (abs(nat - want_nat) < 0.05)
    print(f"  {rt:>5} C  {start} -> {end}  national mean {nat:5.1f} C   "
          f"{'OK' if ok else 'MISMATCH, expected ' + str(want_start)}")
    if not ok:
        bad.append(rt)
assert P["shared_timestamps"] == 41208, \
    f"shared timestamps {P['shared_timestamps']:,}, expected 41,208 — the input data differs"
assert not bad, (f"windows differ at {bad}: this session would not be comparable with "
                 f"Table 5. Do not run cells 11 or 12 until this matches.")
print("\nstate restored and consistent with Table 5 — safe to continue")


# ---------- 7. MILP definitions ----------


# =====================================================================
# 7. MILP (Eq. 1-18)
# =====================================================================
import pulp

def gamma_thermal(temp, lam=None):
    lam = lam or LAMBDA_BY_BIN
    for i in range(len(lam)):
        if TEMP_BINS[i] <= temp < TEMP_BINS[i+1]:
            return lam[i]
    return lam[-1]

def gamma_duration(dt_out=0):
    """Fixed reconnection-surge benchmark: duration-style kernel evaluated at dt_out = 0,
    i.e. a thermally invariant surge of 0.30 (1.30x)."""
    return 0.3 + 0.5 * min(dt_out, 10) / 10.0

def gamma_s5(temp, dt_out=0, lam=None):
    """Equal-weight blend of the duration-style and thermal kernels."""
    return 0.5 * gamma_duration(dt_out) + 0.5 * gamma_thermal(temp, lam)

def build_milp(regime_temp, n_scenarios=N_SCENARIOS, config="S4",
               eps_fair=EPS_FAIR, lam=None, k=K_SIGMA,
               cost_scale=(1.0, 1.0, 1.0)):
    P = prepare_pipeline()
    F = P["govs"]
    Cc, Cr, Cd = C_CURT*cost_scale[0], C_RAMP*cost_scale[1], C_DISC*cost_scale[2]
    pmax = {f: P["mu"][f] + k * P["sigma"][f] for f in F}
    scen = scenarios(regime_temp, n_scenarios)
    win  = P["window"][regime_temp]
    P_gen = {f: win[f]["avail"] * pmax[f] for f in F}
    temps = {f: win[f]["temp"] for f in F}

    N, T = n_scenarios, T_HORIZON
    prob = pulp.LpProblem(f"LoadMgmt_{config}_{regime_temp:.0f}", pulp.LpMinimize)
    u, v, Pc, S, Phi, R = {}, {}, {}, {}, {}, {}
    for f in F:
        for s in range(N):
            for t in range(T):
                u[f,t,s]   = pulp.LpVariable(f"u_{f}_{t}_{s}", cat="Binary")
                v[f,t,s]   = pulp.LpVariable(f"v_{f}_{t}_{s}", lowBound=0, upBound=1)
                Pc[f,t,s]  = pulp.LpVariable(f"curt_{f}_{t}_{s}", lowBound=0)
                S[f,t,s]   = pulp.LpVariable(f"S_{f}_{t}_{s}", lowBound=0)
                Phi[f,t,s] = pulp.LpVariable(f"phi_{f}_{t}_{s}", lowBound=0)
    for t in range(T):
        for s in range(N):
            R[t,s] = pulp.LpVariable(f"R_{t}_{s}", lowBound=0)

    scen_cost = {s: [] for s in range(N)}                       # Eq. (2)
    for s in range(N):
        for t in range(T):
            scen_cost[s] += [Cc * pulp.lpSum(Pc[f,t,s] for f in F),
                             Cr * R[t,s],
                             Cd * pulp.lpSum(Phi[f,t,s] for f in F)]
    prob += (1.0/N) * pulp.lpSum(x for s in range(N) for x in scen_cost[s])

    eff = {}
    for f in F:
        for s in range(N):
            for t in range(T):
                base = float(scen[f][s, t])
                prev = 0 if t == 0 else u[f,t-1,s]
                prob += (v[f,t,s] >= u[f,t,s] - prev), f"clpu_v1_{f}_{t}_{s}"   # Eq. (7)
                prob += (v[f,t,s] <= u[f,t,s]),        f"clpu_v2_{f}_{t}_{s}"   # Eq. (8)
                if t > 0:
                    prob += (v[f,t,s] <= 1 - u[f,t-1,s]), f"clpu_v3_{f}_{t}_{s}"  # Eq. (9)
                gamma = {"S1": 0.0, "S2": 0.0,
                         "S3": gamma_duration(0),
                         "S4": gamma_thermal(temps[f][t], lam),
                         "S5": gamma_s5(temps[f][t], 0, lam)}[config]            # Eq. (10)
                surge = gamma * base
                e = u[f,t,s] * base + surge * v[f,t,s]                            # Eq. (11)
                eff[f,t,s] = e
                prob += (Pc[f,t,s] <= e),             f"clpu_c1_{f}_{t}_{s}"      # Eq. (4)
                prob += (Pc[f,t,s] <= base + surge),  f"clpu_c2_{f}_{t}_{s}"      # Eq. (5)
                prob += (e - Pc[f,t,s] <= pmax[f]),   f"clpu_c3_{f}_{t}_{s}"      # Eq. (6)
                served = e - Pc[f,t,s]
                if t == 0:                                                        # Eq. (14-15)
                    prob += (S[f,t,s] == served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Cd * (1 - u[f,t,s]) * base), f"disc_{f}_{t}_{s}"
                else:
                    prob += (S[f,t,s] == S[f,t-1,s] + served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Phi[f,t-1,s] + Cd * (1 - u[f,t,s]) * base),\
                            f"disc_{f}_{t}_{s}"                                   # Eq. (18)

    for s in range(N):                                                            # Eq. (3)
        for t in range(T):
            prob += (pulp.lpSum(eff[f,t,s] - Pc[f,t,s] for f in F)
                     <= sum(P_gen[f][t] for f in F)), f"bal_{t}_{s}"
    for s in range(N):                                                            # Eq. (12-13)
        for t in range(1, T):
            lhs = pulp.lpSum(eff[f,t,s] - eff[f,t-1,s] for f in F)
            prob += (R[t,s] >= lhs),  f"ramp_p_{t}_{s}"
            prob += (R[t,s] >= -lhs), f"ramp_n_{t}_{s}"
    if config in ("S2", "S4", "S5"):                                              # Eq. (16-17)
        for t in range(T):
            for s in range(N):
                for f in F:
                    for j in F:
                        if f != j:
                            prob += (S[f,t,s] - S[j,t,s] <= eps_fair), f"fair_{f}_{j}_{t}_{s}"

    savg = {}
    for f in F:
        savg[f] = pulp.LpVariable(f"Savg_{f}", lowBound=0)
        prob += (savg[f] == (1.0/N) * pulp.lpSum(S[f, T-1, s] for s in range(N))), f"savg_{f}"

    h = dict(vars=dict(u=u, v=v, S=S, R=R, Pcurt=Pc, Phi=Phi, savg=savg),
             scenario_costs={s: pulp.lpSum(scen_cost[s]) for s in range(N)},
             groups={g: [c for n_, c in prob.constraints.items() if n_.startswith(g)]
                     for g in ("ramp_", "fair_", "clpu_")})
    return prob, h

def solve(prob, time_limit=TIME_LIMIT, gap=GAP_REL):
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap))
    return dict(status=pulp.LpStatus[prob.status],
                objective=pulp.value(prob.objective),
                solve_time_s=round(time.time() - t0, 2),
                n_variables=prob.numVariables(),
                n_constraints=prob.numConstraints())

def report(prob, h, n_scen, report_n=None):
    """Metrics over the first `report_n` scenarios (default: all of them)."""
    F = prepare_pipeline()["govs"]
    m = min(report_n or n_scen, n_scen)      # never index past the solved scenarios
    S, R = h["vars"]["S"], h["vars"]["R"]
    max_ramp = max(pulp.value(R[t,s]) for t in range(T_HORIZON) for s in range(m))
    disp, jfi = [], []
    for s in range(m):
        x = np.array([pulp.value(S[f, T_HORIZON-1, s]) for f in F], dtype=float)
        disp.append(x.max() - x.min())
        denom = len(x) * (x**2).sum()
        jfi.append((x.sum()**2) / denom if denom > 0 else np.nan)   # all-zero service -> undefined
    return dict(Z=pulp.value(prob.objective), MaxRamp=float(max_ramp),
                Disparity_Mean=float(np.mean(disp)), Real_JFI=float(np.nanmean(jfi)))

print("MILP builder ready")


# ---------- solver wrapper that records the MIP gap ----------


import re

def solve_with_gap(prob, time_limit=TIME_LIMIT, gap=GAP_REL, tag="run"):
    """solve(), plus the lower bound and relative gap parsed from the CBC log."""
    log = f"{OUT}/cbc_{tag}.log"
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap, logPath=log))
    elapsed = round(time.time() - t0, 2)
    z = pulp.value(prob.objective)
    lb, rel = None, None
    try:
        txt = open(log, errors="ignore").read()
        m = re.search(r"^Lower bound:\s+([-\d.eE+]+)", txt, re.M)
        if m:
            lb = float(m.group(1))
            rel = (z - lb) / abs(z) if z not in (None, 0) else None
        if lb is None and re.search(r"Result - Optimal solution found", txt):
            lb, rel = z, 0.0          # proven optimal: the bound equals the objective
    except FileNotFoundError:
        pass
    return dict(status=pulp.LpStatus[prob.status], objective=z, solve_time_s=elapsed,
                lower_bound=lb, gap_rel=rel,
                hit_cap=elapsed >= time_limit - 1,
                n_variables=prob.numVariables(), n_constraints=prob.numConstraints())



print("\n" + "=" * 60)
print("boot complete — build_milp, solve, solve_with_gap and report are defined")
print("=" * 60)

# =====================================================================
# ONE CELL: boot + the contested-instance check, resumable.
# Paste and run. Nothing is retrained; cached stages are read back from Drive.
#
# Budget: 900 s per instance, three times the 300 s used for Table 5. The first
# long run showed why a larger budget buys little here — six times the budget
# moved the 17.5 C S2 incumbent by 0.24% and left a 33% gap — so the question
# this answers is whether the S4-vs-S2 differences survive more computation,
# not whether the gap closes. All six instances use the same budget, so the
# comparison is internally consistent; the earlier 1800 s run is kept on Drive
# as separate evidence and is not mixed into this set.
# =====================================================================

# 1. Configuration
# =====================================================================
import os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

OUT = "results"
os.makedirs(OUT, exist_ok=True)
FORCE = False          # True = recompute every stage, ignoring cached CSVs

CFG = dict(
    scada_csv       = "NATIONAL_MASTER_MATRIX_CLEAN_v2.csv",
    hachmann_csv    = "hachmann_digitized.csv",
    governorates    = ["Anbar", "Babil", "Kirkuk", "Najaf", "Wasit"],
    thermal_regimes = [10.5, 17.5, 25.5, 32.8, 40.5, 46.4],   # recomputed in stage 6
    regime_pcts     = [10, 30, 50, 70, 90, 99],
    configs         = ["S1", "S2", "S3", "S4", "S5"],
)

# --- MILP parameters (Eq. 1-18 of the manuscript) --------------------
T_HORIZON    = 24          # planning horizon, hours
N_SCENARIOS  = 20          # |Omega|
REPORT_N     = 20          # scenarios used for MaxRamp / disparity / JFI
K_SIGMA      = 1.0         # P_max(f) = mu + k*sigma                      (Eq. 1)
C_CURT       = 1.2         # curtailment cost                             (Eq. 2)
C_RAMP       = 0.4         # ramping cost                                 (Eq. 2)
C_DISC       = 0.2         # discomfort price                             (Eq. 2, 18)
EPS_FAIR     = 0.05        # fairness tolerance                           (Eq. 16-17)
TEMP_BINS    = [-np.inf, 30, 35, 40, 45, np.inf]   # temperature bin edges, degC
LAMBDA_BY_BIN= [0.30, 0.45, 0.60, 0.70, 0.80]      # lambda_b -> 1.30x .. 1.80x

# --- solver ----------------------------------------------------------
TIME_LIMIT   = 300         # seconds per instance
GAP_REL      = 0.03        # 3% relative MIP gap

# --- forecasting -----------------------------------------------------
LOOKBACK     = 24
TEST_DAYS    = 30
SEED         = 42

LAMBDA_RANGES = {
    "conservative": [0.10, 0.20, 0.30, 0.40, 0.50],   # 1.10x - 1.50x
    "baseline":     LAMBDA_BY_BIN,                    # 1.30x - 1.80x
    "aggressive":   [0.50, 0.60, 0.75, 0.90, 1.00],   # 1.50x - 2.00x
}
SCENARIO_GRID = [10, 20, 30, 50]
EPS_GRID      = [0.03, 0.05, 0.10]
CONV_REGIMES  = "extremes"     # convergence / eps studies run at the two extreme regimes

rng_global = np.random.default_rng(SEED)

def stage(name, outfile):
    """Decorator: skip the stage if its output already exists (unless FORCE).
    The path is resolved at call time, so changing OUT takes effect immediately."""
    def deco(fn):
        def wrapped(*a, **kw):
            path = f"{OUT}/{outfile}"
            if os.path.exists(path) and not FORCE:
                print(f"[skip] {name} -- {path} already exists", flush=True)
                return pd.read_csv(path)
            print(f"[run ] {name}  ->  {path}", flush=True)
            t0 = time.time()
            df = fn(*a, **kw)
            if df is None:
                raise RuntimeError(f"{name}: stage returned no table")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[done] {name} in {time.time()-t0:,.0f}s -> {path}", flush=True)
            return df
        return wrapped
    return deco

print("configuration loaded | N_SCENARIOS =", N_SCENARIOS,
      "| time limit =", TIME_LIMIT, "s | gap =", GAP_REL)



# ---------- 1b. Drive path (cell 1 resets OUT, so this must follow it) ----------
OUT = "/content/drive/MyDrive/segan_results"
os.makedirs(OUT, exist_ok=True)
print("OUT =", OUT, "|", len(os.listdir(OUT)), "cached files")

# The dataset lives on Drive. Cell 1 carries a bare filename, which only resolves
# while a copy happens to sit in /content; that copy does not survive a restart.
CFG["scada_csv"] = "/content/drive/MyDrive/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv"
assert os.path.exists(CFG["scada_csv"]), (
    "dataset not found at " + CFG["scada_csv"] + " — check the path or remount Drive")
print("dataset  =", CFG["scada_csv"])


# ---------- 2. data ----------


# =====================================================================
# 2. Data
# =====================================================================
def load_scada():
    raw = pd.read_csv(CFG["scada_csv"])
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    counts = raw.Governorate.value_counts()
    problems = []
    if len(raw) != 206040:
        problems.append(f"expected 206,040 rows, got {len(raw):,}")
    if len(counts) != 5:
        problems.append(f"expected 5 governorates, got {len(counts)}")
    if len(counts) and counts.min() != 41208:
        problems.append(f"expected 41,208 rows per governorate, min is {counts.min():,}")
    dups = raw.duplicated(subset=["Governorate", "Timestamp"]).sum()
    if dups:
        problems.append(f"{dups} duplicate (Governorate, Timestamp) rows")
    if problems:
        raise ValueError("SCADA file failed integrity checks:\n  - " +
                         "\n  - ".join(problems) +
                         f"\nCounts: {counts.to_dict()}")
    df = raw.rename(columns={"Governorate": "gov", "Timestamp": "ts",
                             "Demand_Value": "demand_mw", "temperature_2m": "temp_c"})
    return df.sort_values(["gov", "ts"]).reset_index(drop=True)

SCADA = load_scada()
print(f"rows              : {len(SCADA):,}")
print(f"governorates      : {sorted(SCADA.gov.unique())}")
print(f"period            : {SCADA.ts.min()}  ->  {SCADA.ts.max()}")
print(f"temperature (degC): {SCADA.temp_c.min():.1f} .. {SCADA.temp_c.max():.1f}"
      f"  ({SCADA.temp_c.nunique():,} distinct values)")
print(f"demand (MW)       : {SCADA.demand_mw.min():.1f} .. {SCADA.demand_mw.max():.1f}")
display(SCADA.groupby("gov").agg(n=("demand_mw","size"),
                                 mu=("demand_mw","mean"),
                                 sigma=("demand_mw","std"),
                                 t_min=("temp_c","min"),
                                 t_max=("temp_c","max")).round(2))


# ---------- 3. BiLSTM (cached: reads bilstm_metrics.csv, no retraining) ----------


# =====================================================================
# 3. BiLSTM forecasting
# =====================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)

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

@stage("BiLSTM per governorate", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)
        print(f"  {g:7} RMSE={m['RMSE_MW']:6.2f}  MAE={m['MAE_MW']:6.2f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE_pct']:.2f}%  (epochs={m['epochs']})")
        mets.append(m); fcs.append(fc)
    pd.concat(fcs, ignore_index=True).to_csv(f"{OUT}/bilstm_forecast_hourly.csv", index=False)
    cols = ["Governorate", "RMSE_MW", "MAE_MW", "R2", "MAPE_pct", "epochs"]
    return pd.DataFrame(mets)[cols]

BILSTM = run_bilstm()
FORECAST = pd.read_csv(f"{OUT}/bilstm_forecast_hourly.csv")
display(BILSTM.round(4))
print(f"\nTable 3 -> R2 {BILSTM.R2.min():.3f}-{BILSTM.R2.max():.3f} | "
      f"RMSE {BILSTM.RMSE_MW.min():.2f}-{BILSTM.RMSE_MW.max():.2f} MW")


# ---------- 6. pipeline, synchronous national windows ----------


PIPE = {}

def prepare_pipeline():
    if PIPE:
        return PIPE
    s = SCADA.copy()
    s["avail"] = s["Supply_Hours"] / 24.0
    g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")
    PIPE["mu"], PIPE["sigma"] = g["mu"].to_dict(), g["sigma"].to_dict()
    PIPE["govs"] = CFG["governorates"]
    PIPE["pmax"] = {f: PIPE["mu"][f] + K_SIGMA * PIPE["sigma"][f] for f in PIPE["govs"]}
    PIPE["arr"] = {f: s[s.gov == f].sort_values("ts").reset_index(drop=True)
                   for f in PIPE["govs"]}

    PIPE["rel_dev"] = {}
    for f in PIPE["govs"]:
        d = FORECAST[FORECAST.Governorate == f]
        PIPE["rel_dev"][f] = ((d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values) /
                              np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6))

    # thermal regimes recomputed from the record, not assumed
    CFG["thermal_regimes"] = [round(float(x), 1) for x in
                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]

    # ---- one 24-hour window per regime, the same hours for all governorates ----
    wide_t = (s.pivot_table(index="ts", columns="gov", values="temp_c")
                .dropna(how="any").sort_index())[PIPE["govs"]]
    national = wide_t.mean(axis=1)
    roll = national.rolling(T_HORIZON).mean()
    by_ts = {f: PIPE["arr"][f].set_index("ts") for f in PIPE["govs"]}

    PIPE["window"], PIPE["window_span"] = {}, {}
    for rt in CFG["thermal_regimes"]:
        pos = int(np.nanargmin(np.abs(roll.values - rt)))
        ts_win = wide_t.index[pos - T_HORIZON + 1: pos + 1]
        PIPE["window"][rt] = {}
        for f in PIPE["govs"]:
            sl = by_ts[f].reindex(ts_win)
            assert not sl[["demand_mw", "temp_c", "avail"]].isna().any().any(), \
                f"gap in the record for {f} at regime {rt}"
            PIPE["window"][rt][f] = dict(demand=sl["demand_mw"].values,
                                         temp=sl["temp_c"].values,
                                         avail=sl["avail"].values)
        PIPE["window_span"][rt] = (str(ts_win[0])[:16], str(ts_win[-1])[:16],
                                   float(national.reindex(ts_win).mean()))

    PIPE["scen_cache"] = {}
    PIPE["shared_timestamps"] = len(wide_t)
    return PIPE


def scenarios(rt, n_scen):
    P = prepare_pipeline()
    key = (round(rt, 1), n_scen)
    if key in P["scen_cache"]:
        return P["scen_cache"][key]
    out = {}
    for gi, f in enumerate(P["govs"]):
        rng = np.random.default_rng([SEED, int(round(rt * 10)), gi, n_scen])
        base = P["window"][rt][f]["demand"]
        rd = P["rel_dev"][f]
        out[f] = np.stack([base * (1 + rng.choice(rd, size=T_HORIZON, replace=True))
                           for _ in range(n_scen)])
    P["scen_cache"][key] = out
    return out


P = prepare_pipeline()
print("thermal regimes:", CFG["thermal_regimes"])
print("shared hourly timestamps across all five governorates:", f"{P['shared_timestamps']:,}")
print("\nP_max(f) = mu + k*sigma, k =", K_SIGMA)
for f in P["govs"]:
    print(f"  {f:7} mu={P['mu'][f]:8.2f}  sigma={P['sigma'][f]:7.2f}  P_max={P['pmax'][f]:8.2f} MW")

# ---- state check: the restored pipeline must reproduce the Table 5 windows ----
EXPECTED = {10.5: ("2020-01-08 11:00", 10.5), 17.5: ("2020-11-16 01:00", 17.5),
            25.6: ("2024-04-08 09:00", 25.6), 32.8: ("2024-09-16 16:00", 32.8),
            40.5: ("2022-06-19 17:00", 40.5), 46.4: ("2020-07-29 12:00", 43.0)}
print("\n--- window check against the run that produced Table 5 ---")
bad = []
for rt in CFG["thermal_regimes"]:
    start, end, nat = P["window_span"][rt]
    want_start, want_nat = EXPECTED.get(rt, (None, None))
    ok = (start == want_start) and (abs(nat - want_nat) < 0.05)
    print(f"  {rt:>5} C  {start} -> {end}  national mean {nat:5.1f} C   "
          f"{'OK' if ok else 'MISMATCH, expected ' + str(want_start)}")
    if not ok:
        bad.append(rt)
assert P["shared_timestamps"] == 41208, \
    f"shared timestamps {P['shared_timestamps']:,}, expected 41,208 — the input data differs"
assert not bad, (f"windows differ at {bad}: this session would not be comparable with "
                 f"Table 5. Do not run cells 11 or 12 until this matches.")
print("\nstate restored and consistent with Table 5 — safe to continue")


# ---------- 7. MILP definitions ----------


# =====================================================================
# 7. MILP (Eq. 1-18)
# =====================================================================
import pulp

def gamma_thermal(temp, lam=None):
    lam = lam or LAMBDA_BY_BIN
    for i in range(len(lam)):
        if TEMP_BINS[i] <= temp < TEMP_BINS[i+1]:
            return lam[i]
    return lam[-1]

def gamma_duration(dt_out=0):
    """Fixed reconnection-surge benchmark: duration-style kernel evaluated at dt_out = 0,
    i.e. a thermally invariant surge of 0.30 (1.30x)."""
    return 0.3 + 0.5 * min(dt_out, 10) / 10.0

def gamma_s5(temp, dt_out=0, lam=None):
    """Equal-weight blend of the duration-style and thermal kernels."""
    return 0.5 * gamma_duration(dt_out) + 0.5 * gamma_thermal(temp, lam)

def build_milp(regime_temp, n_scenarios=N_SCENARIOS, config="S4",
               eps_fair=EPS_FAIR, lam=None, k=K_SIGMA,
               cost_scale=(1.0, 1.0, 1.0)):
    P = prepare_pipeline()
    F = P["govs"]
    Cc, Cr, Cd = C_CURT*cost_scale[0], C_RAMP*cost_scale[1], C_DISC*cost_scale[2]
    pmax = {f: P["mu"][f] + k * P["sigma"][f] for f in F}
    scen = scenarios(regime_temp, n_scenarios)
    win  = P["window"][regime_temp]
    P_gen = {f: win[f]["avail"] * pmax[f] for f in F}
    temps = {f: win[f]["temp"] for f in F}

    N, T = n_scenarios, T_HORIZON
    prob = pulp.LpProblem(f"LoadMgmt_{config}_{regime_temp:.0f}", pulp.LpMinimize)
    u, v, Pc, S, Phi, R = {}, {}, {}, {}, {}, {}
    for f in F:
        for s in range(N):
            for t in range(T):
                u[f,t,s]   = pulp.LpVariable(f"u_{f}_{t}_{s}", cat="Binary")
                v[f,t,s]   = pulp.LpVariable(f"v_{f}_{t}_{s}", lowBound=0, upBound=1)
                Pc[f,t,s]  = pulp.LpVariable(f"curt_{f}_{t}_{s}", lowBound=0)
                S[f,t,s]   = pulp.LpVariable(f"S_{f}_{t}_{s}", lowBound=0)
                Phi[f,t,s] = pulp.LpVariable(f"phi_{f}_{t}_{s}", lowBound=0)
    for t in range(T):
        for s in range(N):
            R[t,s] = pulp.LpVariable(f"R_{t}_{s}", lowBound=0)

    scen_cost = {s: [] for s in range(N)}                       # Eq. (2)
    for s in range(N):
        for t in range(T):
            scen_cost[s] += [Cc * pulp.lpSum(Pc[f,t,s] for f in F),
                             Cr * R[t,s],
                             Cd * pulp.lpSum(Phi[f,t,s] for f in F)]
    prob += (1.0/N) * pulp.lpSum(x for s in range(N) for x in scen_cost[s])

    eff = {}
    for f in F:
        for s in range(N):
            for t in range(T):
                base = float(scen[f][s, t])
                prev = 0 if t == 0 else u[f,t-1,s]
                prob += (v[f,t,s] >= u[f,t,s] - prev), f"clpu_v1_{f}_{t}_{s}"   # Eq. (7)
                prob += (v[f,t,s] <= u[f,t,s]),        f"clpu_v2_{f}_{t}_{s}"   # Eq. (8)
                if t > 0:
                    prob += (v[f,t,s] <= 1 - u[f,t-1,s]), f"clpu_v3_{f}_{t}_{s}"  # Eq. (9)
                gamma = {"S1": 0.0, "S2": 0.0,
                         "S3": gamma_duration(0),
                         "S4": gamma_thermal(temps[f][t], lam),
                         "S5": gamma_s5(temps[f][t], 0, lam)}[config]            # Eq. (10)
                surge = gamma * base
                e = u[f,t,s] * base + surge * v[f,t,s]                            # Eq. (11)
                eff[f,t,s] = e
                prob += (Pc[f,t,s] <= e),             f"clpu_c1_{f}_{t}_{s}"      # Eq. (4)
                prob += (Pc[f,t,s] <= base + surge),  f"clpu_c2_{f}_{t}_{s}"      # Eq. (5)
                prob += (e - Pc[f,t,s] <= pmax[f]),   f"clpu_c3_{f}_{t}_{s}"      # Eq. (6)
                served = e - Pc[f,t,s]
                if t == 0:                                                        # Eq. (14-15)
                    prob += (S[f,t,s] == served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Cd * (1 - u[f,t,s]) * base), f"disc_{f}_{t}_{s}"
                else:
                    prob += (S[f,t,s] == S[f,t-1,s] + served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Phi[f,t-1,s] + Cd * (1 - u[f,t,s]) * base),\
                            f"disc_{f}_{t}_{s}"                                   # Eq. (18)

    for s in range(N):                                                            # Eq. (3)
        for t in range(T):
            prob += (pulp.lpSum(eff[f,t,s] - Pc[f,t,s] for f in F)
                     <= sum(P_gen[f][t] for f in F)), f"bal_{t}_{s}"
    for s in range(N):                                                            # Eq. (12-13)
        for t in range(1, T):
            lhs = pulp.lpSum(eff[f,t,s] - eff[f,t-1,s] for f in F)
            prob += (R[t,s] >= lhs),  f"ramp_p_{t}_{s}"
            prob += (R[t,s] >= -lhs), f"ramp_n_{t}_{s}"
    if config in ("S2", "S4", "S5"):                                              # Eq. (16-17)
        for t in range(T):
            for s in range(N):
                for f in F:
                    for j in F:
                        if f != j:
                            prob += (S[f,t,s] - S[j,t,s] <= eps_fair), f"fair_{f}_{j}_{t}_{s}"

    savg = {}
    for f in F:
        savg[f] = pulp.LpVariable(f"Savg_{f}", lowBound=0)
        prob += (savg[f] == (1.0/N) * pulp.lpSum(S[f, T-1, s] for s in range(N))), f"savg_{f}"

    h = dict(vars=dict(u=u, v=v, S=S, R=R, Pcurt=Pc, Phi=Phi, savg=savg),
             scenario_costs={s: pulp.lpSum(scen_cost[s]) for s in range(N)},
             groups={g: [c for n_, c in prob.constraints.items() if n_.startswith(g)]
                     for g in ("ramp_", "fair_", "clpu_")})
    return prob, h

def solve(prob, time_limit=TIME_LIMIT, gap=GAP_REL):
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap))
    return dict(status=pulp.LpStatus[prob.status],
                objective=pulp.value(prob.objective),
                solve_time_s=round(time.time() - t0, 2),
                n_variables=prob.numVariables(),
                n_constraints=prob.numConstraints())

def report(prob, h, n_scen, report_n=None):
    """Metrics over the first `report_n` scenarios (default: all of them)."""
    F = prepare_pipeline()["govs"]
    m = min(report_n or n_scen, n_scen)      # never index past the solved scenarios
    S, R = h["vars"]["S"], h["vars"]["R"]
    max_ramp = max(pulp.value(R[t,s]) for t in range(T_HORIZON) for s in range(m))
    disp, jfi = [], []
    for s in range(m):
        x = np.array([pulp.value(S[f, T_HORIZON-1, s]) for f in F], dtype=float)
        disp.append(x.max() - x.min())
        denom = len(x) * (x**2).sum()
        jfi.append((x.sum()**2) / denom if denom > 0 else np.nan)   # all-zero service -> undefined
    return dict(Z=pulp.value(prob.objective), MaxRamp=float(max_ramp),
                Disparity_Mean=float(np.mean(disp)), Real_JFI=float(np.nanmean(jfi)))

print("MILP builder ready")


# ---------- solver wrapper that records the MIP gap ----------


import re

def solve_with_gap(prob, time_limit=TIME_LIMIT, gap=GAP_REL, tag="run"):
    """solve(), plus the lower bound and relative gap parsed from the CBC log."""
    log = f"{OUT}/cbc_{tag}.log"
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap, logPath=log))
    elapsed = round(time.time() - t0, 2)
    z = pulp.value(prob.objective)
    lb, rel = None, None
    try:
        txt = open(log, errors="ignore").read()
        m = re.search(r"^Lower bound:\s+([-\d.eE+]+)", txt, re.M)
        if m:
            lb = float(m.group(1))
            rel = (z - lb) / abs(z) if z not in (None, 0) else None
        if lb is None and re.search(r"Result - Optimal solution found", txt):
            lb, rel = z, 0.0          # proven optimal: the bound equals the objective
    except FileNotFoundError:
        pass
    return dict(status=pulp.LpStatus[prob.status], objective=z, solve_time_s=elapsed,
                lower_bound=lb, gap_rel=rel,
                hit_cap=elapsed >= time_limit - 1,
                n_variables=prob.numVariables(), n_constraints=prob.numConstraints())



print("\n" + "=" * 60)
print("boot complete — build_milp, solve, solve_with_gap and report are defined")
print("=" * 60)


# =====================================================================
# PART 2 — the six contested instances at a uniform 900 s budget
# =====================================================================
LONG_LIMIT = 900       # seconds per instance; raise to 1800 to repeat the slower protocol
LONG_GAP   = 0.01      # target gap; not expected to be reached
PARTIAL    = f"{OUT}/table5_longrun_{LONG_LIMIT}_partial.csv"
FINAL      = f"{OUT}/table5_longrun_{LONG_LIMIT}.csv"

CONTESTED = [(CFG["thermal_regimes"][i], cfg) for i in (1, 2, 4) for cfg in ("S2", "S4")]

def _resume():
    if not os.path.exists(PARTIAL):
        return set(), []
    df = pd.read_csv(PARTIAL)
    return {(round(float(r.regime), 1), r.config) for r in df.itertuples()}, df.to_dict("records")

def run_contested():
    main = pd.read_csv(f"{OUT}/table5_main.csv")
    done, rows = _resume()
    todo = [k for k in CONTESTED if (round(k[0], 1), k[1]) not in done]
    print(f"budget {LONG_LIMIT} s | {len(done)} done, {len(todo)} to solve "
          f"(~{len(todo)*LONG_LIMIT/60:.0f} min)", flush=True)
    if done:
        print("  already on Drive: " + ", ".join(f"{r:g} {c}" for r, c in sorted(done)),
              flush=True)

    for i, (rt, cfg) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] solving {rt:g} {cfg} ...", flush=True)
        prob, h = build_milp(rt, N_SCENARIOS, config=cfg)
        info = solve_with_gap(prob, time_limit=LONG_LIMIT, gap=LONG_GAP,
                              tag=f"long{LONG_LIMIT}_{rt}_{cfg}")
        r = report(prob, h, N_SCENARIOS)
        ref = float(main[(np.isclose(main.regime, rt)) & (main.config == cfg)].iloc[0].Z)
        rows.append(dict(regime=rt, config=cfg, budget_s=LONG_LIMIT, **info, **r,
                         Z_table5=ref, improvement_pct=100*(r["Z"]/ref - 1)))
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)      # saved before the next solve
        g = info["gap_rel"]
        gtxt = "n/a" if g is None else "%.2f%%" % (100*g)
        print(f"        {rt:>5} {cfg}  Z={r['Z']:>12,.1f}  (Table 5: {ref:>12,.1f}, "
              f"{100*(r['Z']/ref-1):+6.2f}%)  gap={gtxt:>7}  "
              f"{info['solve_time_s']:>6.1f}s  {'CAP' if info['hit_cap'] else 'converged'}"
              f"   [saved]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(FINAL, index=False)
    return out

LONG = run_contested()
display(LONG.round(4))

print("\n--- S4 vs S2: Table 5 against the longer budget ---")
zl = {(round(r.regime, 1), r.config): r.Z for r in LONG.itertuples()}
main = pd.read_csv(f"{OUT}/table5_main.csv")
zt = {(round(r.regime, 1), r.config): r.Z for r in main.itertuples()}
for rt in sorted({k[0] for k in zl}):
    if (rt, "S4") in zl and (rt, "S2") in zl:
        new = 100*(zl[(rt, "S4")]/zl[(rt, "S2")] - 1)
        old = 100*(zt[(rt, "S4")]/zt[(rt, "S2")] - 1)
        print(f"  {rt:>5} C: {old:+6.2f}%  ->  {new:+6.2f}%   "
              f"(shift {new-old:+.2f} points, {'same sign' if new*old > 0 else 'SIGN REVERSED'})")

print("\n--- how far each incumbent moved, and where it stopped ---")
for r in LONG.itertuples():
    g = "n/a" if pd.isna(r.gap_rel) else f"{100*r.gap_rel:.2f}%"
    print(f"  {r.regime:>5} {r.config}  {r.improvement_pct:+6.2f}% vs Table 5   "
          f"gap {g:>7}   {'reached the cap' if r.hit_cap else 'converged'}")

# =====================================================================
# ONE CELL: boot + the contested-instance check, resumable.
# Paste and run. Nothing is retrained; cached stages are read back from Drive.
#
# Budget: 900 s per instance, three times the 300 s used for Table 5. The first
# long run showed why a larger budget buys little here — six times the budget
# moved the 17.5 C S2 incumbent by 0.24% and left a 33% gap — so the question
# this answers is whether the S4-vs-S2 differences survive more computation,
# not whether the gap closes. All six instances use the same budget, so the
# comparison is internally consistent; the earlier 1800 s run is kept on Drive
# as separate evidence and is not mixed into this set.
# =====================================================================

# ---------- 0. mount Drive (must precede anything that touches OUT) ----------
import os, shutil

def _mount_drive(mp="/content/drive"):
    if os.path.ismount(mp):
        print("Drive already mounted")
        return
    # A failed earlier run can leave empty stub directories at the mountpoint,
    # which drive.mount refuses to mount over. Remove them ONLY if the tree holds
    # no files at all, so real data is never deleted by this.
    if os.path.isdir(mp):
        if any(files for _, _, files in os.walk(mp)):
            raise RuntimeError(
                mp + " is not mounted but contains files — inspect it before rerunning; "
                "this script will not delete them")
        shutil.rmtree(mp)
        print("removed empty stub directories at", mp)
    from google.colab import drive
    drive.mount(mp)

_mount_drive()
assert os.path.ismount("/content/drive"), "Drive did not mount"

# 1. Configuration
# =====================================================================
import os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

OUT = "results"
os.makedirs(OUT, exist_ok=True)
FORCE = False          # True = recompute every stage, ignoring cached CSVs

CFG = dict(
    scada_csv       = "NATIONAL_MASTER_MATRIX_CLEAN_v2.csv",
    hachmann_csv    = "hachmann_digitized.csv",
    governorates    = ["Anbar", "Babil", "Kirkuk", "Najaf", "Wasit"],
    thermal_regimes = [10.5, 17.5, 25.5, 32.8, 40.5, 46.4],   # recomputed in stage 6
    regime_pcts     = [10, 30, 50, 70, 90, 99],
    configs         = ["S1", "S2", "S3", "S4", "S5"],
)

# --- MILP parameters (Eq. 1-18 of the manuscript) --------------------
T_HORIZON    = 24          # planning horizon, hours
N_SCENARIOS  = 20          # |Omega|
REPORT_N     = 20          # scenarios used for MaxRamp / disparity / JFI
K_SIGMA      = 1.0         # P_max(f) = mu + k*sigma                      (Eq. 1)
C_CURT       = 1.2         # curtailment cost                             (Eq. 2)
C_RAMP       = 0.4         # ramping cost                                 (Eq. 2)
C_DISC       = 0.2         # discomfort price                             (Eq. 2, 18)
EPS_FAIR     = 0.05        # fairness tolerance                           (Eq. 16-17)
TEMP_BINS    = [-np.inf, 30, 35, 40, 45, np.inf]   # temperature bin edges, degC
LAMBDA_BY_BIN= [0.30, 0.45, 0.60, 0.70, 0.80]      # lambda_b -> 1.30x .. 1.80x

# --- solver ----------------------------------------------------------
TIME_LIMIT   = 300         # seconds per instance
GAP_REL      = 0.03        # 3% relative MIP gap

# --- forecasting -----------------------------------------------------
LOOKBACK     = 24
TEST_DAYS    = 30
SEED         = 42

LAMBDA_RANGES = {
    "conservative": [0.10, 0.20, 0.30, 0.40, 0.50],   # 1.10x - 1.50x
    "baseline":     LAMBDA_BY_BIN,                    # 1.30x - 1.80x
    "aggressive":   [0.50, 0.60, 0.75, 0.90, 1.00],   # 1.50x - 2.00x
}
SCENARIO_GRID = [10, 20, 30, 50]
EPS_GRID      = [0.03, 0.05, 0.10]
CONV_REGIMES  = "extremes"     # convergence / eps studies run at the two extreme regimes

rng_global = np.random.default_rng(SEED)

def stage(name, outfile):
    """Decorator: skip the stage if its output already exists (unless FORCE).
    The path is resolved at call time, so changing OUT takes effect immediately."""
    def deco(fn):
        def wrapped(*a, **kw):
            path = f"{OUT}/{outfile}"
            if os.path.exists(path) and not FORCE:
                print(f"[skip] {name} -- {path} already exists", flush=True)
                return pd.read_csv(path)
            print(f"[run ] {name}  ->  {path}", flush=True)
            t0 = time.time()
            df = fn(*a, **kw)
            if df is None:
                raise RuntimeError(f"{name}: stage returned no table")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[done] {name} in {time.time()-t0:,.0f}s -> {path}", flush=True)
            return df
        return wrapped
    return deco

print("configuration loaded | N_SCENARIOS =", N_SCENARIOS,
      "| time limit =", TIME_LIMIT, "s | gap =", GAP_REL)



# ---------- 1b. Drive path (cell 1 resets OUT, so this must follow it) ----------
OUT = "/content/drive/MyDrive/segan_results"
os.makedirs(OUT, exist_ok=True)
print("OUT =", OUT, "|", len(os.listdir(OUT)), "cached files")

# The dataset lives on Drive. Cell 1 carries a bare filename, which only resolves
# while a copy happens to sit in /content; that copy does not survive a restart.
CFG["scada_csv"] = "/content/drive/MyDrive/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv"
assert os.path.exists(CFG["scada_csv"]), (
    "dataset not found at " + CFG["scada_csv"] + " — check the path or remount Drive")
print("dataset  =", CFG["scada_csv"])


# ---------- 2. data ----------


# =====================================================================
# 2. Data
# =====================================================================
def load_scada():
    raw = pd.read_csv(CFG["scada_csv"])
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    counts = raw.Governorate.value_counts()
    problems = []
    if len(raw) != 206040:
        problems.append(f"expected 206,040 rows, got {len(raw):,}")
    if len(counts) != 5:
        problems.append(f"expected 5 governorates, got {len(counts)}")
    if len(counts) and counts.min() != 41208:
        problems.append(f"expected 41,208 rows per governorate, min is {counts.min():,}")
    dups = raw.duplicated(subset=["Governorate", "Timestamp"]).sum()
    if dups:
        problems.append(f"{dups} duplicate (Governorate, Timestamp) rows")
    if problems:
        raise ValueError("SCADA file failed integrity checks:\n  - " +
                         "\n  - ".join(problems) +
                         f"\nCounts: {counts.to_dict()}")
    df = raw.rename(columns={"Governorate": "gov", "Timestamp": "ts",
                             "Demand_Value": "demand_mw", "temperature_2m": "temp_c"})
    return df.sort_values(["gov", "ts"]).reset_index(drop=True)

SCADA = load_scada()
print(f"rows              : {len(SCADA):,}")
print(f"governorates      : {sorted(SCADA.gov.unique())}")
print(f"period            : {SCADA.ts.min()}  ->  {SCADA.ts.max()}")
print(f"temperature (degC): {SCADA.temp_c.min():.1f} .. {SCADA.temp_c.max():.1f}"
      f"  ({SCADA.temp_c.nunique():,} distinct values)")
print(f"demand (MW)       : {SCADA.demand_mw.min():.1f} .. {SCADA.demand_mw.max():.1f}")
display(SCADA.groupby("gov").agg(n=("demand_mw","size"),
                                 mu=("demand_mw","mean"),
                                 sigma=("demand_mw","std"),
                                 t_min=("temp_c","min"),
                                 t_max=("temp_c","max")).round(2))


# ---------- 3. BiLSTM (cached: reads bilstm_metrics.csv, no retraining) ----------


# =====================================================================
# 3. BiLSTM forecasting
# =====================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)

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

@stage("BiLSTM per governorate", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)
        print(f"  {g:7} RMSE={m['RMSE_MW']:6.2f}  MAE={m['MAE_MW']:6.2f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE_pct']:.2f}%  (epochs={m['epochs']})")
        mets.append(m); fcs.append(fc)
    pd.concat(fcs, ignore_index=True).to_csv(f"{OUT}/bilstm_forecast_hourly.csv", index=False)
    cols = ["Governorate", "RMSE_MW", "MAE_MW", "R2", "MAPE_pct", "epochs"]
    return pd.DataFrame(mets)[cols]

BILSTM = run_bilstm()
FORECAST = pd.read_csv(f"{OUT}/bilstm_forecast_hourly.csv")
display(BILSTM.round(4))
print(f"\nTable 3 -> R2 {BILSTM.R2.min():.3f}-{BILSTM.R2.max():.3f} | "
      f"RMSE {BILSTM.RMSE_MW.min():.2f}-{BILSTM.RMSE_MW.max():.2f} MW")


# ---------- 6. pipeline, synchronous national windows ----------


PIPE = {}

def prepare_pipeline():
    if PIPE:
        return PIPE
    s = SCADA.copy()
    s["avail"] = s["Supply_Hours"] / 24.0
    g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")
    PIPE["mu"], PIPE["sigma"] = g["mu"].to_dict(), g["sigma"].to_dict()
    PIPE["govs"] = CFG["governorates"]
    PIPE["pmax"] = {f: PIPE["mu"][f] + K_SIGMA * PIPE["sigma"][f] for f in PIPE["govs"]}
    PIPE["arr"] = {f: s[s.gov == f].sort_values("ts").reset_index(drop=True)
                   for f in PIPE["govs"]}

    PIPE["rel_dev"] = {}
    for f in PIPE["govs"]:
        d = FORECAST[FORECAST.Governorate == f]
        PIPE["rel_dev"][f] = ((d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values) /
                              np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6))

    # thermal regimes recomputed from the record, not assumed
    CFG["thermal_regimes"] = [round(float(x), 1) for x in
                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]

    # ---- one 24-hour window per regime, the same hours for all governorates ----
    wide_t = (s.pivot_table(index="ts", columns="gov", values="temp_c")
                .dropna(how="any").sort_index())[PIPE["govs"]]
    national = wide_t.mean(axis=1)
    roll = national.rolling(T_HORIZON).mean()
    by_ts = {f: PIPE["arr"][f].set_index("ts") for f in PIPE["govs"]}

    PIPE["window"], PIPE["window_span"] = {}, {}
    for rt in CFG["thermal_regimes"]:
        pos = int(np.nanargmin(np.abs(roll.values - rt)))
        ts_win = wide_t.index[pos - T_HORIZON + 1: pos + 1]
        PIPE["window"][rt] = {}
        for f in PIPE["govs"]:
            sl = by_ts[f].reindex(ts_win)
            assert not sl[["demand_mw", "temp_c", "avail"]].isna().any().any(), \
                f"gap in the record for {f} at regime {rt}"
            PIPE["window"][rt][f] = dict(demand=sl["demand_mw"].values,
                                         temp=sl["temp_c"].values,
                                         avail=sl["avail"].values)
        PIPE["window_span"][rt] = (str(ts_win[0])[:16], str(ts_win[-1])[:16],
                                   float(national.reindex(ts_win).mean()))

    PIPE["scen_cache"] = {}
    PIPE["shared_timestamps"] = len(wide_t)
    return PIPE


def scenarios(rt, n_scen):
    P = prepare_pipeline()
    key = (round(rt, 1), n_scen)
    if key in P["scen_cache"]:
        return P["scen_cache"][key]
    out = {}
    for gi, f in enumerate(P["govs"]):
        rng = np.random.default_rng([SEED, int(round(rt * 10)), gi, n_scen])
        base = P["window"][rt][f]["demand"]
        rd = P["rel_dev"][f]
        out[f] = np.stack([base * (1 + rng.choice(rd, size=T_HORIZON, replace=True))
                           for _ in range(n_scen)])
    P["scen_cache"][key] = out
    return out


P = prepare_pipeline()
print("thermal regimes:", CFG["thermal_regimes"])
print("shared hourly timestamps across all five governorates:", f"{P['shared_timestamps']:,}")
print("\nP_max(f) = mu + k*sigma, k =", K_SIGMA)
for f in P["govs"]:
    print(f"  {f:7} mu={P['mu'][f]:8.2f}  sigma={P['sigma'][f]:7.2f}  P_max={P['pmax'][f]:8.2f} MW")

# ---- state check: the restored pipeline must reproduce the Table 5 windows ----
EXPECTED = {10.5: ("2020-01-08 11:00", 10.5), 17.5: ("2020-11-16 01:00", 17.5),
            25.6: ("2024-04-08 09:00", 25.6), 32.8: ("2024-09-16 16:00", 32.8),
            40.5: ("2022-06-19 17:00", 40.5), 46.4: ("2020-07-29 12:00", 43.0)}
print("\n--- window check against the run that produced Table 5 ---")
bad = []
for rt in CFG["thermal_regimes"]:
    start, end, nat = P["window_span"][rt]
    want_start, want_nat = EXPECTED.get(rt, (None, None))
    ok = (start == want_start) and (abs(nat - want_nat) < 0.05)
    print(f"  {rt:>5} C  {start} -> {end}  national mean {nat:5.1f} C   "
          f"{'OK' if ok else 'MISMATCH, expected ' + str(want_start)}")
    if not ok:
        bad.append(rt)
assert P["shared_timestamps"] == 41208, \
    f"shared timestamps {P['shared_timestamps']:,}, expected 41,208 — the input data differs"
assert not bad, (f"windows differ at {bad}: this session would not be comparable with "
                 f"Table 5. Do not run cells 11 or 12 until this matches.")
print("\nstate restored and consistent with Table 5 — safe to continue")


# ---------- 7. MILP definitions ----------


# =====================================================================
# 7. MILP (Eq. 1-18)
# =====================================================================
import pulp

def gamma_thermal(temp, lam=None):
    lam = lam or LAMBDA_BY_BIN
    for i in range(len(lam)):
        if TEMP_BINS[i] <= temp < TEMP_BINS[i+1]:
            return lam[i]
    return lam[-1]

def gamma_duration(dt_out=0):
    """Fixed reconnection-surge benchmark: duration-style kernel evaluated at dt_out = 0,
    i.e. a thermally invariant surge of 0.30 (1.30x)."""
    return 0.3 + 0.5 * min(dt_out, 10) / 10.0

def gamma_s5(temp, dt_out=0, lam=None):
    """Equal-weight blend of the duration-style and thermal kernels."""
    return 0.5 * gamma_duration(dt_out) + 0.5 * gamma_thermal(temp, lam)

def build_milp(regime_temp, n_scenarios=N_SCENARIOS, config="S4",
               eps_fair=EPS_FAIR, lam=None, k=K_SIGMA,
               cost_scale=(1.0, 1.0, 1.0)):
    P = prepare_pipeline()
    F = P["govs"]
    Cc, Cr, Cd = C_CURT*cost_scale[0], C_RAMP*cost_scale[1], C_DISC*cost_scale[2]
    pmax = {f: P["mu"][f] + k * P["sigma"][f] for f in F}
    scen = scenarios(regime_temp, n_scenarios)
    win  = P["window"][regime_temp]
    P_gen = {f: win[f]["avail"] * pmax[f] for f in F}
    temps = {f: win[f]["temp"] for f in F}

    N, T = n_scenarios, T_HORIZON
    prob = pulp.LpProblem(f"LoadMgmt_{config}_{regime_temp:.0f}", pulp.LpMinimize)
    u, v, Pc, S, Phi, R = {}, {}, {}, {}, {}, {}
    for f in F:
        for s in range(N):
            for t in range(T):
                u[f,t,s]   = pulp.LpVariable(f"u_{f}_{t}_{s}", cat="Binary")
                v[f,t,s]   = pulp.LpVariable(f"v_{f}_{t}_{s}", lowBound=0, upBound=1)
                Pc[f,t,s]  = pulp.LpVariable(f"curt_{f}_{t}_{s}", lowBound=0)
                S[f,t,s]   = pulp.LpVariable(f"S_{f}_{t}_{s}", lowBound=0)
                Phi[f,t,s] = pulp.LpVariable(f"phi_{f}_{t}_{s}", lowBound=0)
    for t in range(T):
        for s in range(N):
            R[t,s] = pulp.LpVariable(f"R_{t}_{s}", lowBound=0)

    scen_cost = {s: [] for s in range(N)}                       # Eq. (2)
    for s in range(N):
        for t in range(T):
            scen_cost[s] += [Cc * pulp.lpSum(Pc[f,t,s] for f in F),
                             Cr * R[t,s],
                             Cd * pulp.lpSum(Phi[f,t,s] for f in F)]
    prob += (1.0/N) * pulp.lpSum(x for s in range(N) for x in scen_cost[s])

    eff = {}
    for f in F:
        for s in range(N):
            for t in range(T):
                base = float(scen[f][s, t])
                prev = 0 if t == 0 else u[f,t-1,s]
                prob += (v[f,t,s] >= u[f,t,s] - prev), f"clpu_v1_{f}_{t}_{s}"   # Eq. (7)
                prob += (v[f,t,s] <= u[f,t,s]),        f"clpu_v2_{f}_{t}_{s}"   # Eq. (8)
                if t > 0:
                    prob += (v[f,t,s] <= 1 - u[f,t-1,s]), f"clpu_v3_{f}_{t}_{s}"  # Eq. (9)
                gamma = {"S1": 0.0, "S2": 0.0,
                         "S3": gamma_duration(0),
                         "S4": gamma_thermal(temps[f][t], lam),
                         "S5": gamma_s5(temps[f][t], 0, lam)}[config]            # Eq. (10)
                surge = gamma * base
                e = u[f,t,s] * base + surge * v[f,t,s]                            # Eq. (11)
                eff[f,t,s] = e
                prob += (Pc[f,t,s] <= e),             f"clpu_c1_{f}_{t}_{s}"      # Eq. (4)
                prob += (Pc[f,t,s] <= base + surge),  f"clpu_c2_{f}_{t}_{s}"      # Eq. (5)
                prob += (e - Pc[f,t,s] <= pmax[f]),   f"clpu_c3_{f}_{t}_{s}"      # Eq. (6)
                served = e - Pc[f,t,s]
                if t == 0:                                                        # Eq. (14-15)
                    prob += (S[f,t,s] == served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Cd * (1 - u[f,t,s]) * base), f"disc_{f}_{t}_{s}"
                else:
                    prob += (S[f,t,s] == S[f,t-1,s] + served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Phi[f,t-1,s] + Cd * (1 - u[f,t,s]) * base),\
                            f"disc_{f}_{t}_{s}"                                   # Eq. (18)

    for s in range(N):                                                            # Eq. (3)
        for t in range(T):
            prob += (pulp.lpSum(eff[f,t,s] - Pc[f,t,s] for f in F)
                     <= sum(P_gen[f][t] for f in F)), f"bal_{t}_{s}"
    for s in range(N):                                                            # Eq. (12-13)
        for t in range(1, T):
            lhs = pulp.lpSum(eff[f,t,s] - eff[f,t-1,s] for f in F)
            prob += (R[t,s] >= lhs),  f"ramp_p_{t}_{s}"
            prob += (R[t,s] >= -lhs), f"ramp_n_{t}_{s}"
    if config in ("S2", "S4", "S5"):                                              # Eq. (16-17)
        for t in range(T):
            for s in range(N):
                for f in F:
                    for j in F:
                        if f != j:
                            prob += (S[f,t,s] - S[j,t,s] <= eps_fair), f"fair_{f}_{j}_{t}_{s}"

    savg = {}
    for f in F:
        savg[f] = pulp.LpVariable(f"Savg_{f}", lowBound=0)
        prob += (savg[f] == (1.0/N) * pulp.lpSum(S[f, T-1, s] for s in range(N))), f"savg_{f}"

    h = dict(vars=dict(u=u, v=v, S=S, R=R, Pcurt=Pc, Phi=Phi, savg=savg),
             scenario_costs={s: pulp.lpSum(scen_cost[s]) for s in range(N)},
             groups={g: [c for n_, c in prob.constraints.items() if n_.startswith(g)]
                     for g in ("ramp_", "fair_", "clpu_")})
    return prob, h

def solve(prob, time_limit=TIME_LIMIT, gap=GAP_REL):
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap))
    return dict(status=pulp.LpStatus[prob.status],
                objective=pulp.value(prob.objective),
                solve_time_s=round(time.time() - t0, 2),
                n_variables=prob.numVariables(),
                n_constraints=prob.numConstraints())

def report(prob, h, n_scen, report_n=None):
    """Metrics over the first `report_n` scenarios (default: all of them)."""
    F = prepare_pipeline()["govs"]
    m = min(report_n or n_scen, n_scen)      # never index past the solved scenarios
    S, R = h["vars"]["S"], h["vars"]["R"]
    max_ramp = max(pulp.value(R[t,s]) for t in range(T_HORIZON) for s in range(m))
    disp, jfi = [], []
    for s in range(m):
        x = np.array([pulp.value(S[f, T_HORIZON-1, s]) for f in F], dtype=float)
        disp.append(x.max() - x.min())
        denom = len(x) * (x**2).sum()
        jfi.append((x.sum()**2) / denom if denom > 0 else np.nan)   # all-zero service -> undefined
    return dict(Z=pulp.value(prob.objective), MaxRamp=float(max_ramp),
                Disparity_Mean=float(np.mean(disp)), Real_JFI=float(np.nanmean(jfi)))

print("MILP builder ready")


# ---------- solver wrapper that records the MIP gap ----------


import re

def solve_with_gap(prob, time_limit=TIME_LIMIT, gap=GAP_REL, tag="run"):
    """solve(), plus the lower bound and relative gap parsed from the CBC log."""
    log = f"{OUT}/cbc_{tag}.log"
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap, logPath=log))
    elapsed = round(time.time() - t0, 2)
    z = pulp.value(prob.objective)
    lb, rel = None, None
    try:
        txt = open(log, errors="ignore").read()
        m = re.search(r"^Lower bound:\s+([-\d.eE+]+)", txt, re.M)
        if m:
            lb = float(m.group(1))
            rel = (z - lb) / abs(z) if z not in (None, 0) else None
        if lb is None and re.search(r"Result - Optimal solution found", txt):
            lb, rel = z, 0.0          # proven optimal: the bound equals the objective
    except FileNotFoundError:
        pass
    return dict(status=pulp.LpStatus[prob.status], objective=z, solve_time_s=elapsed,
                lower_bound=lb, gap_rel=rel,
                hit_cap=elapsed >= time_limit - 1,
                n_variables=prob.numVariables(), n_constraints=prob.numConstraints())



print("\n" + "=" * 60)
print("boot complete — build_milp, solve, solve_with_gap and report are defined")
print("=" * 60)


# =====================================================================
# PART 2 — the six contested instances at a uniform 900 s budget
# =====================================================================
LONG_LIMIT = 900       # seconds per instance; raise to 1800 to repeat the slower protocol
LONG_GAP   = 0.01      # target gap; not expected to be reached
PARTIAL    = f"{OUT}/table5_longrun_{LONG_LIMIT}_partial.csv"
FINAL      = f"{OUT}/table5_longrun_{LONG_LIMIT}.csv"

CONTESTED = [(CFG["thermal_regimes"][i], cfg) for i in (1, 2, 4) for cfg in ("S2", "S4")]

def _resume():
    if not os.path.exists(PARTIAL):
        return set(), []
    df = pd.read_csv(PARTIAL)
    return {(round(float(r.regime), 1), r.config) for r in df.itertuples()}, df.to_dict("records")

def run_contested():
    main = pd.read_csv(f"{OUT}/table5_main.csv")
    done, rows = _resume()
    todo = [k for k in CONTESTED if (round(k[0], 1), k[1]) not in done]
    print(f"budget {LONG_LIMIT} s | {len(done)} done, {len(todo)} to solve "
          f"(~{len(todo)*LONG_LIMIT/60:.0f} min)", flush=True)
    if done:
        print("  already on Drive: " + ", ".join(f"{r:g} {c}" for r, c in sorted(done)),
              flush=True)

    for i, (rt, cfg) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] solving {rt:g} {cfg} ...", flush=True)
        prob, h = build_milp(rt, N_SCENARIOS, config=cfg)
        info = solve_with_gap(prob, time_limit=LONG_LIMIT, gap=LONG_GAP,
                              tag=f"long{LONG_LIMIT}_{rt}_{cfg}")
        r = report(prob, h, N_SCENARIOS)
        ref = float(main[(np.isclose(main.regime, rt)) & (main.config == cfg)].iloc[0].Z)
        rows.append(dict(regime=rt, config=cfg, budget_s=LONG_LIMIT, **info, **r,
                         Z_table5=ref, improvement_pct=100*(r["Z"]/ref - 1)))
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)      # saved before the next solve
        g = info["gap_rel"]
        gtxt = "n/a" if g is None else "%.2f%%" % (100*g)
        print(f"        {rt:>5} {cfg}  Z={r['Z']:>12,.1f}  (Table 5: {ref:>12,.1f}, "
              f"{100*(r['Z']/ref-1):+6.2f}%)  gap={gtxt:>7}  "
              f"{info['solve_time_s']:>6.1f}s  {'CAP' if info['hit_cap'] else 'converged'}"
              f"   [saved]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(FINAL, index=False)
    return out

LONG = run_contested()
display(LONG.round(4))

print("\n--- S4 vs S2: Table 5 against the longer budget ---")
zl = {(round(r.regime, 1), r.config): r.Z for r in LONG.itertuples()}
main = pd.read_csv(f"{OUT}/table5_main.csv")
zt = {(round(r.regime, 1), r.config): r.Z for r in main.itertuples()}
for rt in sorted({k[0] for k in zl}):
    if (rt, "S4") in zl and (rt, "S2") in zl:
        new = 100*(zl[(rt, "S4")]/zl[(rt, "S2")] - 1)
        old = 100*(zt[(rt, "S4")]/zt[(rt, "S2")] - 1)
        print(f"  {rt:>5} C: {old:+6.2f}%  ->  {new:+6.2f}%   "
              f"(shift {new-old:+.2f} points, {'same sign' if new*old > 0 else 'SIGN REVERSED'})")

print("\n--- how far each incumbent moved, and where it stopped ---")
for r in LONG.itertuples():
    g = "n/a" if pd.isna(r.gap_rel) else f"{100*r.gap_rel:.2f}%"
    print(f"  {r.regime:>5} {r.config}  {r.improvement_pct:+6.2f}% vs Table 5   "
          f"gap {g:>7}   {'reached the cap' if r.hit_cap else 'converged'}")

# =====================================================================
# ONE CELL: boot + the gap sweep over all 30 Table 5 instances, resumable.
# Paste and run. Rows already on Drive are skipped, so a restarted runtime
# costs at most the solve that was in flight.
# =====================================================================

# ---------- 0. mount Drive (must precede anything that touches OUT) ----------
import os, shutil

def _mount_drive(mp="/content/drive"):
    if os.path.ismount(mp):
        print("Drive already mounted")
        return
    # A failed earlier run can leave empty stub directories at the mountpoint,
    # which drive.mount refuses to mount over. Remove them ONLY if the tree holds
    # no files at all, so real data is never deleted by this.
    if os.path.isdir(mp):
        if any(files for _, _, files in os.walk(mp)):
            raise RuntimeError(
                mp + " is not mounted but contains files — inspect it before rerunning; "
                "this script will not delete them")
        shutil.rmtree(mp)
        print("removed empty stub directories at", mp)
    from google.colab import drive
    drive.mount(mp)

_mount_drive()
assert os.path.ismount("/content/drive"), "Drive did not mount"

# 1. Configuration
# =====================================================================
import os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

OUT = "results"
os.makedirs(OUT, exist_ok=True)
FORCE = False          # True = recompute every stage, ignoring cached CSVs

CFG = dict(
    scada_csv       = "NATIONAL_MASTER_MATRIX_CLEAN_v2.csv",
    hachmann_csv    = "hachmann_digitized.csv",
    governorates    = ["Anbar", "Babil", "Kirkuk", "Najaf", "Wasit"],
    thermal_regimes = [10.5, 17.5, 25.5, 32.8, 40.5, 46.4],   # recomputed in stage 6
    regime_pcts     = [10, 30, 50, 70, 90, 99],
    configs         = ["S1", "S2", "S3", "S4", "S5"],
)

# --- MILP parameters (Eq. 1-18 of the manuscript) --------------------
T_HORIZON    = 24          # planning horizon, hours
N_SCENARIOS  = 20          # |Omega|
REPORT_N     = 20          # scenarios used for MaxRamp / disparity / JFI
K_SIGMA      = 1.0         # P_max(f) = mu + k*sigma                      (Eq. 1)
C_CURT       = 1.2         # curtailment cost                             (Eq. 2)
C_RAMP       = 0.4         # ramping cost                                 (Eq. 2)
C_DISC       = 0.2         # discomfort price                             (Eq. 2, 18)
EPS_FAIR     = 0.05        # fairness tolerance                           (Eq. 16-17)
TEMP_BINS    = [-np.inf, 30, 35, 40, 45, np.inf]   # temperature bin edges, degC
LAMBDA_BY_BIN= [0.30, 0.45, 0.60, 0.70, 0.80]      # lambda_b -> 1.30x .. 1.80x

# --- solver ----------------------------------------------------------
TIME_LIMIT   = 300         # seconds per instance
GAP_REL      = 0.03        # 3% relative MIP gap

# --- forecasting -----------------------------------------------------
LOOKBACK     = 24
TEST_DAYS    = 30
SEED         = 42

LAMBDA_RANGES = {
    "conservative": [0.10, 0.20, 0.30, 0.40, 0.50],   # 1.10x - 1.50x
    "baseline":     LAMBDA_BY_BIN,                    # 1.30x - 1.80x
    "aggressive":   [0.50, 0.60, 0.75, 0.90, 1.00],   # 1.50x - 2.00x
}
SCENARIO_GRID = [10, 20, 30, 50]
EPS_GRID      = [0.03, 0.05, 0.10]
CONV_REGIMES  = "extremes"     # convergence / eps studies run at the two extreme regimes

rng_global = np.random.default_rng(SEED)

def stage(name, outfile):
    """Decorator: skip the stage if its output already exists (unless FORCE).
    The path is resolved at call time, so changing OUT takes effect immediately."""
    def deco(fn):
        def wrapped(*a, **kw):
            path = f"{OUT}/{outfile}"
            if os.path.exists(path) and not FORCE:
                print(f"[skip] {name} -- {path} already exists", flush=True)
                return pd.read_csv(path)
            print(f"[run ] {name}  ->  {path}", flush=True)
            t0 = time.time()
            df = fn(*a, **kw)
            if df is None:
                raise RuntimeError(f"{name}: stage returned no table")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[done] {name} in {time.time()-t0:,.0f}s -> {path}", flush=True)
            return df
        return wrapped
    return deco

print("configuration loaded | N_SCENARIOS =", N_SCENARIOS,
      "| time limit =", TIME_LIMIT, "s | gap =", GAP_REL)



# ---------- 1b. Drive path (cell 1 resets OUT, so this must follow it) ----------
OUT = "/content/drive/MyDrive/segan_results"
os.makedirs(OUT, exist_ok=True)
print("OUT =", OUT, "|", len(os.listdir(OUT)), "cached files")

# The dataset lives on Drive. Cell 1 carries a bare filename, which only resolves
# while a copy happens to sit in /content; that copy does not survive a restart.
CFG["scada_csv"] = "/content/drive/MyDrive/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv"
assert os.path.exists(CFG["scada_csv"]), (
    "dataset not found at " + CFG["scada_csv"] + " — check the path or remount Drive")
print("dataset  =", CFG["scada_csv"])


# ---------- 2. data ----------


# =====================================================================
# 2. Data
# =====================================================================
def load_scada():
    raw = pd.read_csv(CFG["scada_csv"])
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    counts = raw.Governorate.value_counts()
    problems = []
    if len(raw) != 206040:
        problems.append(f"expected 206,040 rows, got {len(raw):,}")
    if len(counts) != 5:
        problems.append(f"expected 5 governorates, got {len(counts)}")
    if len(counts) and counts.min() != 41208:
        problems.append(f"expected 41,208 rows per governorate, min is {counts.min():,}")
    dups = raw.duplicated(subset=["Governorate", "Timestamp"]).sum()
    if dups:
        problems.append(f"{dups} duplicate (Governorate, Timestamp) rows")
    if problems:
        raise ValueError("SCADA file failed integrity checks:\n  - " +
                         "\n  - ".join(problems) +
                         f"\nCounts: {counts.to_dict()}")
    df = raw.rename(columns={"Governorate": "gov", "Timestamp": "ts",
                             "Demand_Value": "demand_mw", "temperature_2m": "temp_c"})
    return df.sort_values(["gov", "ts"]).reset_index(drop=True)

SCADA = load_scada()
print(f"rows              : {len(SCADA):,}")
print(f"governorates      : {sorted(SCADA.gov.unique())}")
print(f"period            : {SCADA.ts.min()}  ->  {SCADA.ts.max()}")
print(f"temperature (degC): {SCADA.temp_c.min():.1f} .. {SCADA.temp_c.max():.1f}"
      f"  ({SCADA.temp_c.nunique():,} distinct values)")
print(f"demand (MW)       : {SCADA.demand_mw.min():.1f} .. {SCADA.demand_mw.max():.1f}")
display(SCADA.groupby("gov").agg(n=("demand_mw","size"),
                                 mu=("demand_mw","mean"),
                                 sigma=("demand_mw","std"),
                                 t_min=("temp_c","min"),
                                 t_max=("temp_c","max")).round(2))


# ---------- 3. BiLSTM (cached: reads bilstm_metrics.csv, no retraining) ----------


# =====================================================================
# 3. BiLSTM forecasting
# =====================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)

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

@stage("BiLSTM per governorate", "bilstm_metrics.csv")
def run_bilstm():
    mets, fcs = [], []
    for g in CFG["governorates"]:
        m, fc = train_one(SCADA[SCADA.gov == g].reset_index(drop=True), g)
        print(f"  {g:7} RMSE={m['RMSE_MW']:6.2f}  MAE={m['MAE_MW']:6.2f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE_pct']:.2f}%  (epochs={m['epochs']})")
        mets.append(m); fcs.append(fc)
    pd.concat(fcs, ignore_index=True).to_csv(f"{OUT}/bilstm_forecast_hourly.csv", index=False)
    cols = ["Governorate", "RMSE_MW", "MAE_MW", "R2", "MAPE_pct", "epochs"]
    return pd.DataFrame(mets)[cols]

BILSTM = run_bilstm()
FORECAST = pd.read_csv(f"{OUT}/bilstm_forecast_hourly.csv")
display(BILSTM.round(4))
print(f"\nTable 3 -> R2 {BILSTM.R2.min():.3f}-{BILSTM.R2.max():.3f} | "
      f"RMSE {BILSTM.RMSE_MW.min():.2f}-{BILSTM.RMSE_MW.max():.2f} MW")


# ---------- 6. pipeline, synchronous national windows ----------


PIPE = {}

def prepare_pipeline():
    if PIPE:
        return PIPE
    s = SCADA.copy()
    s["avail"] = s["Supply_Hours"] / 24.0
    g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")
    PIPE["mu"], PIPE["sigma"] = g["mu"].to_dict(), g["sigma"].to_dict()
    PIPE["govs"] = CFG["governorates"]
    PIPE["pmax"] = {f: PIPE["mu"][f] + K_SIGMA * PIPE["sigma"][f] for f in PIPE["govs"]}
    PIPE["arr"] = {f: s[s.gov == f].sort_values("ts").reset_index(drop=True)
                   for f in PIPE["govs"]}

    PIPE["rel_dev"] = {}
    for f in PIPE["govs"]:
        d = FORECAST[FORECAST.Governorate == f]
        PIPE["rel_dev"][f] = ((d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values) /
                              np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6))

    # thermal regimes recomputed from the record, not assumed
    CFG["thermal_regimes"] = [round(float(x), 1) for x in
                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]

    # ---- one 24-hour window per regime, the same hours for all governorates ----
    wide_t = (s.pivot_table(index="ts", columns="gov", values="temp_c")
                .dropna(how="any").sort_index())[PIPE["govs"]]
    national = wide_t.mean(axis=1)
    roll = national.rolling(T_HORIZON).mean()
    by_ts = {f: PIPE["arr"][f].set_index("ts") for f in PIPE["govs"]}

    PIPE["window"], PIPE["window_span"] = {}, {}
    for rt in CFG["thermal_regimes"]:
        pos = int(np.nanargmin(np.abs(roll.values - rt)))
        ts_win = wide_t.index[pos - T_HORIZON + 1: pos + 1]
        PIPE["window"][rt] = {}
        for f in PIPE["govs"]:
            sl = by_ts[f].reindex(ts_win)
            assert not sl[["demand_mw", "temp_c", "avail"]].isna().any().any(), \
                f"gap in the record for {f} at regime {rt}"
            PIPE["window"][rt][f] = dict(demand=sl["demand_mw"].values,
                                         temp=sl["temp_c"].values,
                                         avail=sl["avail"].values)
        PIPE["window_span"][rt] = (str(ts_win[0])[:16], str(ts_win[-1])[:16],
                                   float(national.reindex(ts_win).mean()))

    PIPE["scen_cache"] = {}
    PIPE["shared_timestamps"] = len(wide_t)
    return PIPE


def scenarios(rt, n_scen):
    P = prepare_pipeline()
    key = (round(rt, 1), n_scen)
    if key in P["scen_cache"]:
        return P["scen_cache"][key]
    out = {}
    for gi, f in enumerate(P["govs"]):
        rng = np.random.default_rng([SEED, int(round(rt * 10)), gi, n_scen])
        base = P["window"][rt][f]["demand"]
        rd = P["rel_dev"][f]
        out[f] = np.stack([base * (1 + rng.choice(rd, size=T_HORIZON, replace=True))
                           for _ in range(n_scen)])
    P["scen_cache"][key] = out
    return out


P = prepare_pipeline()
print("thermal regimes:", CFG["thermal_regimes"])
print("shared hourly timestamps across all five governorates:", f"{P['shared_timestamps']:,}")
print("\nP_max(f) = mu + k*sigma, k =", K_SIGMA)
for f in P["govs"]:
    print(f"  {f:7} mu={P['mu'][f]:8.2f}  sigma={P['sigma'][f]:7.2f}  P_max={P['pmax'][f]:8.2f} MW")

# ---- state check: the restored pipeline must reproduce the Table 5 windows ----
EXPECTED = {10.5: ("2020-01-08 11:00", 10.5), 17.5: ("2020-11-16 01:00", 17.5),
            25.6: ("2024-04-08 09:00", 25.6), 32.8: ("2024-09-16 16:00", 32.8),
            40.5: ("2022-06-19 17:00", 40.5), 46.4: ("2020-07-29 12:00", 43.0)}
print("\n--- window check against the run that produced Table 5 ---")
bad = []
for rt in CFG["thermal_regimes"]:
    start, end, nat = P["window_span"][rt]
    want_start, want_nat = EXPECTED.get(rt, (None, None))
    ok = (start == want_start) and (abs(nat - want_nat) < 0.05)
    print(f"  {rt:>5} C  {start} -> {end}  national mean {nat:5.1f} C   "
          f"{'OK' if ok else 'MISMATCH, expected ' + str(want_start)}")
    if not ok:
        bad.append(rt)
assert P["shared_timestamps"] == 41208, \
    f"shared timestamps {P['shared_timestamps']:,}, expected 41,208 — the input data differs"
assert not bad, (f"windows differ at {bad}: this session would not be comparable with "
                 f"Table 5. Do not run cells 11 or 12 until this matches.")
print("\nstate restored and consistent with Table 5 — safe to continue")


# ---------- 7. MILP definitions ----------


# =====================================================================
# 7. MILP (Eq. 1-18)
# =====================================================================
import pulp

def gamma_thermal(temp, lam=None):
    lam = lam or LAMBDA_BY_BIN
    for i in range(len(lam)):
        if TEMP_BINS[i] <= temp < TEMP_BINS[i+1]:
            return lam[i]
    return lam[-1]

def gamma_duration(dt_out=0):
    """Fixed reconnection-surge benchmark: duration-style kernel evaluated at dt_out = 0,
    i.e. a thermally invariant surge of 0.30 (1.30x)."""
    return 0.3 + 0.5 * min(dt_out, 10) / 10.0

def gamma_s5(temp, dt_out=0, lam=None):
    """Equal-weight blend of the duration-style and thermal kernels."""
    return 0.5 * gamma_duration(dt_out) + 0.5 * gamma_thermal(temp, lam)

def build_milp(regime_temp, n_scenarios=N_SCENARIOS, config="S4",
               eps_fair=EPS_FAIR, lam=None, k=K_SIGMA,
               cost_scale=(1.0, 1.0, 1.0)):
    P = prepare_pipeline()
    F = P["govs"]
    Cc, Cr, Cd = C_CURT*cost_scale[0], C_RAMP*cost_scale[1], C_DISC*cost_scale[2]
    pmax = {f: P["mu"][f] + k * P["sigma"][f] for f in F}
    scen = scenarios(regime_temp, n_scenarios)
    win  = P["window"][regime_temp]
    P_gen = {f: win[f]["avail"] * pmax[f] for f in F}
    temps = {f: win[f]["temp"] for f in F}

    N, T = n_scenarios, T_HORIZON
    prob = pulp.LpProblem(f"LoadMgmt_{config}_{regime_temp:.0f}", pulp.LpMinimize)
    u, v, Pc, S, Phi, R = {}, {}, {}, {}, {}, {}
    for f in F:
        for s in range(N):
            for t in range(T):
                u[f,t,s]   = pulp.LpVariable(f"u_{f}_{t}_{s}", cat="Binary")
                v[f,t,s]   = pulp.LpVariable(f"v_{f}_{t}_{s}", lowBound=0, upBound=1)
                Pc[f,t,s]  = pulp.LpVariable(f"curt_{f}_{t}_{s}", lowBound=0)
                S[f,t,s]   = pulp.LpVariable(f"S_{f}_{t}_{s}", lowBound=0)
                Phi[f,t,s] = pulp.LpVariable(f"phi_{f}_{t}_{s}", lowBound=0)
    for t in range(T):
        for s in range(N):
            R[t,s] = pulp.LpVariable(f"R_{t}_{s}", lowBound=0)

    scen_cost = {s: [] for s in range(N)}                       # Eq. (2)
    for s in range(N):
        for t in range(T):
            scen_cost[s] += [Cc * pulp.lpSum(Pc[f,t,s] for f in F),
                             Cr * R[t,s],
                             Cd * pulp.lpSum(Phi[f,t,s] for f in F)]
    prob += (1.0/N) * pulp.lpSum(x for s in range(N) for x in scen_cost[s])

    eff = {}
    for f in F:
        for s in range(N):
            for t in range(T):
                base = float(scen[f][s, t])
                prev = 0 if t == 0 else u[f,t-1,s]
                prob += (v[f,t,s] >= u[f,t,s] - prev), f"clpu_v1_{f}_{t}_{s}"   # Eq. (7)
                prob += (v[f,t,s] <= u[f,t,s]),        f"clpu_v2_{f}_{t}_{s}"   # Eq. (8)
                if t > 0:
                    prob += (v[f,t,s] <= 1 - u[f,t-1,s]), f"clpu_v3_{f}_{t}_{s}"  # Eq. (9)
                gamma = {"S1": 0.0, "S2": 0.0,
                         "S3": gamma_duration(0),
                         "S4": gamma_thermal(temps[f][t], lam),
                         "S5": gamma_s5(temps[f][t], 0, lam)}[config]            # Eq. (10)
                surge = gamma * base
                e = u[f,t,s] * base + surge * v[f,t,s]                            # Eq. (11)
                eff[f,t,s] = e
                prob += (Pc[f,t,s] <= e),             f"clpu_c1_{f}_{t}_{s}"      # Eq. (4)
                prob += (Pc[f,t,s] <= base + surge),  f"clpu_c2_{f}_{t}_{s}"      # Eq. (5)
                prob += (e - Pc[f,t,s] <= pmax[f]),   f"clpu_c3_{f}_{t}_{s}"      # Eq. (6)
                served = e - Pc[f,t,s]
                if t == 0:                                                        # Eq. (14-15)
                    prob += (S[f,t,s] == served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Cd * (1 - u[f,t,s]) * base), f"disc_{f}_{t}_{s}"
                else:
                    prob += (S[f,t,s] == S[f,t-1,s] + served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f,t,s] == Phi[f,t-1,s] + Cd * (1 - u[f,t,s]) * base),\
                            f"disc_{f}_{t}_{s}"                                   # Eq. (18)

    for s in range(N):                                                            # Eq. (3)
        for t in range(T):
            prob += (pulp.lpSum(eff[f,t,s] - Pc[f,t,s] for f in F)
                     <= sum(P_gen[f][t] for f in F)), f"bal_{t}_{s}"
    for s in range(N):                                                            # Eq. (12-13)
        for t in range(1, T):
            lhs = pulp.lpSum(eff[f,t,s] - eff[f,t-1,s] for f in F)
            prob += (R[t,s] >= lhs),  f"ramp_p_{t}_{s}"
            prob += (R[t,s] >= -lhs), f"ramp_n_{t}_{s}"
    if config in ("S2", "S4", "S5"):                                              # Eq. (16-17)
        for t in range(T):
            for s in range(N):
                for f in F:
                    for j in F:
                        if f != j:
                            prob += (S[f,t,s] - S[j,t,s] <= eps_fair), f"fair_{f}_{j}_{t}_{s}"

    savg = {}
    for f in F:
        savg[f] = pulp.LpVariable(f"Savg_{f}", lowBound=0)
        prob += (savg[f] == (1.0/N) * pulp.lpSum(S[f, T-1, s] for s in range(N))), f"savg_{f}"

    h = dict(vars=dict(u=u, v=v, S=S, R=R, Pcurt=Pc, Phi=Phi, savg=savg),
             scenario_costs={s: pulp.lpSum(scen_cost[s]) for s in range(N)},
             groups={g: [c for n_, c in prob.constraints.items() if n_.startswith(g)]
                     for g in ("ramp_", "fair_", "clpu_")})
    return prob, h

def solve(prob, time_limit=TIME_LIMIT, gap=GAP_REL):
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap))
    return dict(status=pulp.LpStatus[prob.status],
                objective=pulp.value(prob.objective),
                solve_time_s=round(time.time() - t0, 2),
                n_variables=prob.numVariables(),
                n_constraints=prob.numConstraints())

def report(prob, h, n_scen, report_n=None):
    """Metrics over the first `report_n` scenarios (default: all of them)."""
    F = prepare_pipeline()["govs"]
    m = min(report_n or n_scen, n_scen)      # never index past the solved scenarios
    S, R = h["vars"]["S"], h["vars"]["R"]
    max_ramp = max(pulp.value(R[t,s]) for t in range(T_HORIZON) for s in range(m))
    disp, jfi = [], []
    for s in range(m):
        x = np.array([pulp.value(S[f, T_HORIZON-1, s]) for f in F], dtype=float)
        disp.append(x.max() - x.min())
        denom = len(x) * (x**2).sum()
        jfi.append((x.sum()**2) / denom if denom > 0 else np.nan)   # all-zero service -> undefined
    return dict(Z=pulp.value(prob.objective), MaxRamp=float(max_ramp),
                Disparity_Mean=float(np.mean(disp)), Real_JFI=float(np.nanmean(jfi)))

print("MILP builder ready")


# ---------- solver wrapper that records the MIP gap ----------


import re

def solve_with_gap(prob, time_limit=TIME_LIMIT, gap=GAP_REL, tag="run"):
    """solve(), plus the lower bound and relative gap parsed from the CBC log."""
    log = f"{OUT}/cbc_{tag}.log"
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap, logPath=log))
    elapsed = round(time.time() - t0, 2)
    z = pulp.value(prob.objective)
    lb, rel = None, None
    try:
        txt = open(log, errors="ignore").read()
        m = re.search(r"^Lower bound:\s+([-\d.eE+]+)", txt, re.M)
        if m:
            lb = float(m.group(1))
            rel = (z - lb) / abs(z) if z not in (None, 0) else None
        if lb is None and re.search(r"Result - Optimal solution found", txt):
            lb, rel = z, 0.0          # proven optimal: the bound equals the objective
    except FileNotFoundError:
        pass
    return dict(status=pulp.LpStatus[prob.status], objective=z, solve_time_s=elapsed,
                lower_bound=lb, gap_rel=rel,
                hit_cap=elapsed >= time_limit - 1,
                n_variables=prob.numVariables(), n_constraints=prob.numConstraints())



print("\n" + "=" * 60)
print("boot complete — build_milp, solve, solve_with_gap and report are defined")
print("=" * 60)


# =====================================================================
# PART 2 — record the attained MIP gap for every row of Table 5
# Pure instrumentation: same instances, same 300 s budget, same 3% criterion.
# Objectives are expected to reproduce exactly; any that does not is flagged
# as DRIFT and listed separately. Nothing here replaces Table 5.
# =====================================================================
GAP_PARTIAL = f"{OUT}/table5_gaps_partial.csv"
GAP_FINAL   = f"{OUT}/table5_gaps.csv"
TOL         = 0.001        # objectives agreeing within 0.1% count as reproduced

def _resume_gaps():
    if not os.path.exists(GAP_PARTIAL):
        return set(), []
    df = pd.read_csv(GAP_PARTIAL)
    return {(round(float(r.regime), 1), r.config) for r in df.itertuples()}, df.to_dict("records")

def run_gap_sweep():
    main = pd.read_csv(f"{OUT}/table5_main.csv")
    todo_all = [(rt, cfg) for rt in CFG["thermal_regimes"] for cfg in CFG["configs"]]
    done, rows = _resume_gaps()
    todo = [k for k in todo_all if (round(k[0], 1), k[1]) not in done]
    print(f"{len(done)} done, {len(todo)} to solve (~{len(todo)*TIME_LIMIT/60:.0f} min)",
          flush=True)

    for i, (rt, cfg) in enumerate(todo, 1):
        prob, h = build_milp(rt, N_SCENARIOS, config=cfg)
        info = solve_with_gap(prob, time_limit=TIME_LIMIT, gap=GAP_REL, tag=f"gap_{rt}_{cfg}")
        ref = float(main[(np.isclose(main.regime, rt)) & (main.config == cfg)].iloc[0].Z)
        drift = 100 * (info["objective"] / ref - 1)
        rows.append(dict(regime=rt, config=cfg, **info, Z_table5=ref, drift_pct=drift,
                         reproduced=abs(drift) < 100 * TOL))
        pd.DataFrame(rows).to_csv(GAP_PARTIAL, index=False)
        g = info["gap_rel"]
        gtxt = "n/a" if g is None else "%.2f%%" % (100 * g)
        flag = "" if abs(drift) < 100 * TOL else f"   <<< DRIFT {drift:+.3f}%"
        print(f"  [{i:>2}/{len(todo)}] {rt:>5} {cfg}  Z={info['objective']:>12,.1f}  "
              f"gap={gtxt:>7}  LB={info['lower_bound'] or float('nan'):>12,.1f}  "
              f"{info['solve_time_s']:>6.1f}s{flag}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(GAP_FINAL, index=False)
    return out

GAPS = run_gap_sweep()
display(GAPS[["regime", "config", "objective", "Z_table5", "drift_pct",
              "lower_bound", "gap_rel", "hit_cap"]].round(4))

bad = GAPS[~GAPS.reproduced]
print(f"\nreproduced {len(GAPS) - len(bad)} of {len(GAPS)} objectives within {100*TOL:.1f}%")
if len(bad):
    print("NOT reproduced — report these, do not overwrite Table 5:")
    for r in bad.itertuples():
        print(f"  {r.regime:>5} {r.config}  {r.objective:,.1f} vs {r.Z_table5:,.1f} "
              f"({r.drift_pct:+.3f}%)")

print("\n--- attained gap by configuration ---")
for cfg in CFG["configs"]:
    g = GAPS[GAPS.config == cfg].gap_rel.dropna()
    if len(g):
        print(f"  {cfg}  n={len(g)}  median {100*g.median():5.2f}%  "
              f"range {100*g.min():5.2f}%-{100*g.max():5.2f}%")

print("\n--- attained gap by regime ---")
for rt in CFG["thermal_regimes"]:
    g = GAPS[np.isclose(GAPS.regime, rt)].gap_rel.dropna()
    if len(g):
        print(f"  {rt:>5} C  median {100*g.median():5.2f}%  "
              f"range {100*g.min():5.2f}%-{100*g.max():5.2f}%")

met = GAPS[(GAPS.gap_rel.notna()) & (GAPS.gap_rel <= GAP_REL)]
print(f"\ninstances that actually attained the {100*GAP_REL:.0f}% criterion: "
      f"{len(met)} of {len(GAPS)}")
print("  " + ", ".join(f"{r.regime:g} {r.config}" for r in met.itertuples()))

# =====================================================================# 6. Pipeline preparation# =====================================================================PIPE = {}def prepare_pipeline():    if PIPE: return PIPE    s = SCADA.copy()    s["avail"] = s["Supply_Hours"] / 24.0    g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")    PIPE["mu"], PIPE["sigma"] = g["mu"].to_dict(), g["sigma"].to_dict()    PIPE["govs"] = CFG["governorates"]    PIPE["pmax"] = {f: PIPE["mu"][f] + K_SIGMA * PIPE["sigma"][f] for f in PIPE["govs"]}    PIPE["arr"] = {f: s[s.gov == f].reset_index(drop=True) for f in PIPE["govs"]}    PIPE["rel_dev"] = {}    for f in PIPE["govs"]:        d = FORECAST[FORECAST.Governorate == f]        PIPE["rel_dev"][f] = ((d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values) /                              np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6))    # thermal regimes recomputed from the data, not assumed    CFG["thermal_regimes"] = [round(float(x), 1) for x in                              np.percentile(SCADA.temp_c.dropna(), CFG["regime_pcts"])]    PIPE["window"] = {}    for rt in CFG["thermal_regimes"]:        PIPE["window"][rt] = {}        for f in PIPE["govs"]:            gdf = PIPE["arr"][f]            roll = gdf["temp_c"].rolling(T_HORIZON).mean().values            cand = np.where(~np.isnan(roll))[0]            best = cand[np.argmin(np.abs(roll[cand] - rt))]            sl = gdf.iloc[best - T_HORIZON + 1: best + 1]            PIPE["window"][rt][f] = dict(demand=sl["demand_mw"].values,                                         temp=sl["temp_c"].values,                                         avail=sl["avail"].values)    PIPE["scen_cache"] = {}    return PIPEdef scenarios(rt, n_scen):    P = prepare_pipeline()    key = (round(rt, 1), n_scen)    if key in P["scen_cache"]: return P["scen_cache"][key]    out = {}    for gi, f in enumerate(P["govs"]):        rng = np.random.default_rng([SEED, int(round(rt * 10)), gi, n_scen])        base = P["window"][rt][f]["demand"]        rd = P["rel_dev"][f]        out[f] = np.stack([base * (1 + rng.choice(rd, size=T_HORIZON, replace=True))                           for _ in range(n_scen)])    P["scen_cache"][key] = out    return outP = prepare_pipeline()print("thermal regimes (percentiles of the record):", CFG["thermal_regimes"])print("\nP_max(f) = mu + k*sigma, k =", K_SIGMA)for f in P["govs"]:    print(f"  {f:7} mu={P['mu'][f]:8.2f}  sigma={P['sigma'][f]:7.2f}  P_max={P['pmax'][f]:8.2f} MW")for rt in CFG["thermal_regimes"]:    means = {f: round(float(P['window'][rt][f]['temp'].mean()), 1) for f in P["govs"]}    print(f"  target {rt:>5} degC -> window means {means}")

import json

NB = "/content/drive/MyDrive/Colab Notebooks/SEGAN_results_pipeline.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):

    if cell["cell_type"] != "code":
        continue

    txt = "".join(cell["source"])

    if any(k in txt for k in [
        "prepare_pipeline",
        "def scenarios(",
        "build_milp",
        "solve_with_gap",
        "report("
    ]):
        print("\n" + "="*80)
        print("CELL", i)
        print("="*80)
        print(txt)

import json
import os
import re

# Set target to the validated primary pipeline notebook
notebook_dir = "/content/drive/MyDrive/Colab Notebooks"
NOTEBOOK = os.path.join(notebook_dir, "SEGAN_results_pipeline.ipynb")

if not os.path.exists(NOTEBOOK):
    raise FileNotFoundError(f"Could not locate the primary notebook: {NOTEBOOK}")

print("Extracting modules from:", NOTEBOOK)
OUT = "/content/drive/MyDrive/SEGAN_CLPU_FAIRNESS/src"
os.makedirs(OUT, exist_ok=True)

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = []
for c in nb["cells"]:
    if c["cell_type"] == "code":
        cells.append("".join(c["source"]))

full_code = "\n\n".join(cells)

def save(name, text):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    print("created:", path, f"({os.path.getsize(path)} bytes)")

# 1. scenario_generation.py
m_scen = re.search(r"def scenarios\(.*?return out", full_code, flags=re.S)
if m_scen:
    save("scenario_generation.py", m_scen.group(0))

# 2. thermal_regimes.py
m_pipe = re.search(r"PIPE = \{\}.*?def scenarios\(", full_code, flags=re.S)
if m_pipe:
    txt = m_pipe.group(0).rsplit("def scenarios", 1)[0]
    save("thermal_regimes.py", txt)

# 3. milp_builder.py
blocks_milp = []
for fn in ["gamma_thermal", "gamma_duration", "gamma_s5", "build_milp"]:
    m = re.search(rf"def {fn}\(.*?(?=\ndef |\Z)", full_code, flags=re.S)
    if m:
        blocks_milp.append(m.group(0))
save("milp_builder.py", "\n\n".join(blocks_milp))

# 4. optimization_runner.py
blocks_opt = []
for fn in ["solve", "solve_with_gap", "report"]:
    m = re.search(rf"def {fn}\(.*?(?=\ndef |\Z)", full_code, flags=re.S)
    if m:
        blocks_opt.append(m.group(0))
save("optimization_runner.py", "\n\n".join(blocks_opt))

# 5. gap_analysis.py
m_gap = re.search(r"def run_gap_sweep\(.*?return out", full_code, flags=re.S)
if m_gap:
    save("gap_analysis.py", m_gap.group(0))
else:
    # Fallback search if name differs
    m_gap_fallback = re.search(r"def run_contested\(.*?return out", full_code, flags=re.S)
    if m_gap_fallback:
        save("gap_analysis.py", m_gap_fallback.group(0))

# 6. fairness_metrics.py
txt_fair = """
import numpy as np

def jain_fairness_index(x):
    x = np.asarray(x, dtype=float)
    denom = len(x) * np.sum(x**2)
    if denom <= 0:
        return np.nan
    return (np.sum(x)**2) / denom

def disparity(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(x) - np.min(x))
"""
save("fairness_metrics.py", txt_fair)

print("\nFinished extracting source modules.")

import json
import os

NOTEBOOK = "/content/drive/MyDrive/Colab Notebooks/SEGAN_results_pipeline.ipynb"

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

OUT = "/content/drive/MyDrive/SEGAN_CLPU_FAIRNESS/src"
os.makedirs(OUT, exist_ok=True)

targets = {
    "thermal_regimes.py": [
        "PIPE = {}",
        "def prepare_pipeline",
        "def scenarios"
    ],

    "milp_builder.py": [
        "def gamma_thermal",
        "def gamma_duration",
        "def gamma_s5",
        "def build_milp"
    ],

    "optimization_runner.py": [
        "def solve(",
        "def solve_with_gap",
        "def report("
    ],

    "gap_analysis.py": [
        "def run_gap_sweep",
        "def run_contested",
        "def run_contested_resumable"
    ]
}

for out_file, keys in targets.items():

    collected = []

    for cell in nb["cells"]:

        if cell["cell_type"] != "code":
            continue

        txt = "".join(cell["source"])

        if any(k in txt for k in keys):
            collected.append(txt)

    if collected:

        path = os.path.join(OUT, out_file)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(collected))

        print(
            out_file,
            os.path.getsize(path),
            "bytes"
        )