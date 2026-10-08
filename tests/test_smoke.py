"""A self-contained check that the package builds and solves.

It uses a small synthetic record with the real schema, so it runs for anyone,
without the confidential SCADA data and without a GPU:

    python -m pytest tests/ -v        (or simply: python tests/test_smoke.py)

It does not reproduce the published numbers — only the real record does that.
What it establishes is that the formulation builds, the solver returns a feasible
schedule, the fairness constraint binds at the tolerance, and every reported
quantity can be read off the solution.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clpu import analysis, config as C, data, milp  # noqa: E402
from clpu.pipeline import Pipeline  # noqa: E402

HOURS = 24 * 60


def synthetic(tmp: str):
    """A record with the production schema: five governorates, hourly, 60 days."""
    rng = np.random.default_rng(7)
    ts = pd.date_range("2020-01-01", periods=HOURS, freq="h")
    frames = []
    for i, g in enumerate(C.GOVERNORATES):
        hour = ts.hour.values
        temp = (25 + 20 * np.sin(np.arange(HOURS) / (24 * 15))
                + 6 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 1.5, HOURS))
        dem = (900 + 60 * i + 180 * np.sin(hour / 24 * 2 * np.pi)
               + 8 * (temp - 25) + rng.normal(0, 40, HOURS))
        frames.append(pd.DataFrame({
            "Governorate": g, "Timestamp": ts, "Demand_Value": dem.clip(5),
            "temperature_2m": temp, "Supply_Hours": rng.integers(8, 20, HOURS),
        }))
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(tmp, "scada.csv")
    df.to_csv(path, index=False)

    fc = []
    for g in C.GOVERNORATES:
        d = df[df.Governorate == g].tail(24 * 30)
        act = d.Demand_Value.values
        fc.append(pd.DataFrame({
            "Governorate": g, "Timestamp": d.Timestamp.values,
            "Demand_Actual": act,
            "Demand_Forecast_BiLSTM": act * (1 + rng.normal(0, 0.03, len(act))),
            "temperature_2m": d.temperature_2m.values,
        }))
    return path, pd.concat(fc, ignore_index=True)


def build_pipeline(tmp="."):
    path, forecast = synthetic(tmp)
    scada = data.load(path, strict=False)
    return Pipeline(scada, forecast)


def test_pipeline_shapes():
    pipe = build_pipeline()
    assert len(pipe.regimes) == len(C.REGIME_PERCENTILES)
    assert set(pipe.pmax) == set(C.GOVERNORATES)
    for rt in pipe.regimes:
        for f in pipe.govs:
            assert len(pipe.window[rt][f]["demand"]) == C.T_HORIZON


def test_scenarios_are_deterministic():
    pipe = build_pipeline()
    a = pipe.scenarios(pipe.regimes[0], 4)
    pipe._scen_cache.clear()
    b = pipe.scenarios(pipe.regimes[0], 4)
    for f in pipe.govs:
        assert np.allclose(a[f], b[f]), "same seed must give the same scenarios"


def test_every_configuration_solves():
    pipe = build_pipeline()
    rt = pipe.regimes[3]
    for cfg in C.CONFIGS:
        prob, h = milp.build(pipe, rt, cfg, n_scenarios=3)
        info = milp.solve(prob, time_limit=30, gap=0.05)
        assert info["status"] in ("Optimal", "Not Solved"), f"{cfg}: {info['status']}"
        rep = analysis.report(prob, h)
        assert rep["Z"] is not None and rep["Z"] > 0
        assert rep["MaxRamp"] >= 0
        assert len(analysis.scenario_costs(h)) == 3
        st = analysis.schedule_statistics(h)
        assert st["recon_mean"] >= st["structural_floor"] - 1e-6, \
            "a feeder cannot reconnect fewer times than once at the start"
        assert 0 <= st["served_fh_mean"] <= st["max_feeder_hours"]


def test_fairness_constraint_binds():
    pipe = build_pipeline()
    rt = pipe.regimes[3]
    prob, h = milp.build(pipe, rt, "S2", n_scenarios=3)
    milp.solve(prob, time_limit=30, gap=0.05)
    rep = analysis.report(prob, h)
    assert rep["Disparity_Mean"] <= C.EPS_FAIR + 1e-6, \
        "the pairwise constraint must hold the disparity at or below eps_fair"
    assert rep["Real_JFI"] > 0.999, "the linear proxy should track Jain's index closely"


def test_unconstrained_is_cheaper_than_fair():
    pipe = build_pipeline()
    rt = pipe.regimes[3]
    z = {}
    for cfg in ("S1", "S2"):
        prob, h = milp.build(pipe, rt, cfg, n_scenarios=3)
        milp.solve(prob, time_limit=30, gap=0.05)
        z[cfg] = analysis.report(prob, h)["Z"]
    assert z["S2"] >= z["S1"], "adding a constraint cannot lower the optimum"


def test_discomfort_switch_changes_only_the_weighting():
    pipe = build_pipeline()
    rt = pipe.regimes[3]
    sizes = []
    for mode in ("double", "single"):
        prob, h = milp.build(pipe, rt, "S4", n_scenarios=3, disc_mode=mode)
        sizes.append((prob.numVariables(), prob.numConstraints()))
    assert sizes[0] == sizes[1], \
        "the two weightings must give structurally identical instances"


def test_certification_helper():
    assert analysis.certified(100.0, 120.0, 50.0, 90.0) == "A > B"
    assert analysis.certified(50.0, 70.0, 80.0, 95.0) == "A < B"
    assert analysis.certified(60.0, 120.0, 50.0, 110.0) == "overlapping"
    assert analysis.certified(None, 1.0, 2.0, 3.0) == "unknown"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        failures = 0
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    print(f"  PASS  {name}")
                except AssertionError as e:
                    failures += 1
                    print(f"  FAIL  {name}: {e}")
        print("\nall checks passed" if not failures else f"\n{failures} failure(s)")
        sys.exit(1 if failures else 0)
