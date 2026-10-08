"""Per-governorate short-term demand forecasting, and the architecture benchmark.

TensorFlow is imported lazily, so the optimisation side of the package runs without
it. Training is seeded, but GPU kernels are not bit-for-bit reproducible across
hardware and library versions; results/environment.json records the environment the
reported figures came from.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C

ARCHITECTURES = ["BiLSTM", "LSTM", "GRU"]


def make_sequences(demand_n: np.ndarray, temp_n: np.ndarray, lookback: int = C.LOOKBACK):
    """Sliding windows of (demand, temperature) and the next-hour demand target."""
    X = np.stack(
        [
            np.stack([demand_n[i - lookback:i], temp_n[i - lookback:i]], -1)
            for i in range(lookback, len(demand_n))
        ]
    ).astype("float32")
    y = demand_n[lookback:].astype("float32")
    return X, y


def metrics(y: np.ndarray, yh: np.ndarray) -> dict:
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(
        RMSE_MW=float(np.sqrt(np.mean((y - yh) ** 2))),
        MAE_MW=float(np.mean(np.abs(y - yh))),
        R2=float(1 - ss_res / ss_tot),
        MAPE_pct=float(np.mean(np.abs((y - yh) / np.maximum(y, 1e-6))) * 100),
    )


def _build(arch: str):
    from tensorflow import keras
    from tensorflow.keras import layers

    core = {
        "BiLSTM": lambda: layers.Bidirectional(layers.LSTM(64)),
        "LSTM": lambda: layers.LSTM(64),
        "GRU": lambda: layers.GRU(64),
    }[arch]()
    model = keras.Sequential(
        [layers.Input((C.LOOKBACK, 2)), core,
         layers.Dense(32, activation="relu"), layers.Dense(1)]
    )
    model.compile(keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_one(gdf: pd.DataFrame, gov: str, arch: str = "BiLSTM"):
    """Train one model for one governorate on a chronological split.

    Returns (metrics dict, hourly forecast frame for the held-out period).
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(C.SEED)

    d = gdf.demand_mw.values.astype("float32")
    t = gdf.temp_c.values.astype("float32")
    n_train = len(d) - C.TEST_DAYS * 24
    dm, ds = d[:n_train].mean(), d[:n_train].std()
    tm, ts_ = t[:n_train].mean(), t[:n_train].std()
    X, y = make_sequences((d - dm) / ds, (t - tm) / ts_)
    split = n_train - C.LOOKBACK
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    model = _build(arch)
    hist = model.fit(
        Xtr, ytr, validation_split=0.1, epochs=25, batch_size=1024,
        callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=0,
    )
    yh = model.predict(Xte, verbose=0).flatten() * ds + dm
    yt = yte * ds + dm

    m = metrics(yt, yh)
    m.update(Governorate=gov, Architecture=arch, epochs=len(hist.history["loss"]))
    fc = pd.DataFrame(
        {
            "Governorate": gov,
            "Timestamp": gdf.ts.values[n_train:][: len(yt)],
            "Demand_Actual": yt,
            "Demand_Forecast_BiLSTM": yh,
            "temperature_2m": gdf.temp_c.values[n_train:][: len(yt)],
        }
    )
    return m, fc


def run(scada: pd.DataFrame, out_dir: str, force: bool = False, verbose: bool = True):
    """Train the production BiLSTM per governorate, caching to ``out_dir``.

    Returns (metrics frame, hourly forecast frame). If both cache files exist and
    ``force`` is false, they are read back instead of retraining — which is what
    makes the optimisation reproducible without a GPU.
    """
    mpath = os.path.join(out_dir, "bilstm_metrics.csv")
    fpath = os.path.join(out_dir, "bilstm_forecast_hourly.csv")
    if os.path.exists(mpath) and os.path.exists(fpath) and not force:
        if verbose:
            print(f"[cached] forecasts read from {mpath}")
        return pd.read_csv(mpath), pd.read_csv(fpath)

    mets, fcs = [], []
    for g in C.GOVERNORATES:
        m, fc = train_one(scada[scada.gov == g].reset_index(drop=True), g)
        if verbose:
            print(
                f"  {g:7} RMSE={m['RMSE_MW']:6.2f}  MAE={m['MAE_MW']:6.2f}  "
                f"R2={m['R2']:.4f}  MAPE={m['MAPE_pct']:.2f}%  (epochs={m['epochs']})",
                flush=True,
            )
        mets.append(m)
        fcs.append(fc)

    forecast = pd.concat(fcs, ignore_index=True)
    forecast.to_csv(fpath, index=False)
    cols = ["Governorate", "RMSE_MW", "MAE_MW", "R2", "MAPE_pct", "epochs"]
    bilstm = pd.DataFrame(mets)[cols]
    bilstm.to_csv(mpath, index=False)
    return bilstm, forecast


def benchmark(scada: pd.DataFrame, out_dir: str, force: bool = False, verbose: bool = True):
    """LSTM and GRU under the identical protocol, for the architecture comparison."""
    path = os.path.join(out_dir, "architecture_benchmark.csv")
    if os.path.exists(path) and not force:
        if verbose:
            print(f"[cached] architecture benchmark read from {path}")
        return pd.read_csv(path)
    rows = []
    for arch in ARCHITECTURES:
        for g in C.GOVERNORATES:
            m, _ = train_one(scada[scada.gov == g].reset_index(drop=True), g, arch=arch)
            if verbose:
                print(f"  {arch:7} {g:7} R2={m['R2']:.4f}  RMSE={m['RMSE_MW']:6.2f}", flush=True)
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
