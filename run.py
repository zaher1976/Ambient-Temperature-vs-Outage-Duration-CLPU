#!/usr/bin/env python3
"""Command-line entry point for the reproducibility package.

    python run.py check       --data PATH            # load the record, rebuild the windows
    python run.py forecast    --data PATH            # train the BiLSTM (needs TensorFlow)
    python run.py table5      --data PATH            # the 30 main instances
    python run.py sensitivity --data PATH            # kernel-coefficient sensitivity
    python run.py robustness  --data PATH            # the 30 instances, C_disc applied once
    python run.py all         --data PATH            # check, forecast, table5, robustness

Every sweep writes each instance to CSV as it finishes and skips what is already
recorded, so an interrupted run costs at most the instance in flight.

The raw SCADA record is not redistributed with this package; see data/README_DATA.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from clpu import analysis, config as C, data as data_mod, milp
from clpu.pipeline import Pipeline


# ─────────────────────────────────────────────── shared helpers

def _load(args):
    try:
        paths = C.Paths(data=args.data, out=args.out)
        scada = data_mod.load(paths.data, strict=not args.any_dataset)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"\n{exc}\n")
    print(data_mod.describe(scada))
    print()
    print(data_mod.summary(scada).to_string())
    return paths, scada


def _pipeline(paths, scada, args):
    from clpu import forecasting

    bilstm, forecast = forecasting.run(scada, paths.out, force=args.retrain)
    print(f"\nBiLSTM  R2 {bilstm.R2.min():.3f}-{bilstm.R2.max():.3f} | "
          f"RMSE {bilstm.RMSE_MW.min():.2f}-{bilstm.RMSE_MW.max():.2f} MW")
    pipe = Pipeline(scada, forecast, k_sigma=args.k_sigma)
    print()
    print(pipe.describe())
    print("\n--- window check against the run that produced the reported results ---")
    ok = pipe.verify_windows()
    print("state consistent with the reported results\n" if ok else
          "WARNING: the rebuilt windows differ from the reported run; "
          "results below are not comparable with the published tables\n")
    return pipe


def _sweep(pipe, paths, args, *, csv_name, instances, disc_mode, label):
    """Solve a list of (regime, config) pairs, resumable, recording everything."""
    csv_path = paths / csv_name
    scen_path = paths / csv_name.replace(".csv", "_scenario_costs.csv")
    rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
    scen_rows = pd.read_csv(scen_path).to_dict("records") if os.path.exists(scen_path) else []
    seen = {(round(float(r["regime"]), 1), r["config"]) for r in rows}
    todo = [k for k in instances if (round(k[0], 1), k[1]) not in seen]

    print(f"{label}: {len(seen)} of {len(instances)} recorded, {len(todo)} to solve "
          f"(~{len(todo) * args.time_limit / 60:.0f} min at the {args.time_limit}s cap)",
          flush=True)

    t0 = time.time()
    for i, (regime, cfg) in enumerate(todo, 1):
        prob, h = milp.build(pipe, regime, cfg, n_scenarios=args.n_scenarios,
                             eps_fair=args.eps_fair, disc_mode=disc_mode)
        info = milp.solve(prob, time_limit=args.time_limit, gap=args.gap,
                          log_path=paths / f"cbc_{cfg}_{regime:.0f}_{disc_mode}.log")
        rows.append(dict(
            regime=regime, config=cfg, disc_mode=disc_mode,
            **analysis.report(prob, h, C.REPORT_N),
            **analysis.schedule_statistics(h),
            **analysis.binding_shares(h),
            lower_bound=info["lower_bound"], gap_rel=info["gap_rel"],
            solve_time_s=info["solve_time_s"], hit_cap=info["hit_cap"],
            status=info["status"], n_variables=info["n_variables"],
            n_constraints=info["n_constraints"],
        ))
        scen_rows += [dict(regime=regime, config=cfg, scenario=s, cost=c)
                      for s, c in enumerate(analysis.scenario_costs(h))]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        pd.DataFrame(scen_rows).to_csv(scen_path, index=False)

        g = info["gap_rel"]
        eta = (time.time() - t0) / i * (len(todo) - i) / 60
        print(f"  [{i:>2}/{len(todo)}] {regime:>5} {cfg}  Z={rows[-1]['Z']:>11,.1f}  "
              f"ramp={rows[-1]['MaxRamp']:>7,.0f}  "
              f"gap={'n/a' if g is None else '%5.2f%%' % (100 * g)}  "
              f"{info['solve_time_s']:>5.0f}s   ETA {eta:,.0f} min", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nsaved -> {csv_path}\n        {scen_path}")
    return df


def _summarise(df, pipe):
    def z(regime, cfg, col="Z"):
        r = df[(df.regime.round(1) == round(regime, 1)) & (df.config == cfg)]
        return float(r.iloc[0][col]) if len(r) else float("nan")

    print("\n" + "=" * 72)
    print(f"{'regime':>7} | {'S3-S1':>9} {'S4-S2':>9} {'S5-S4':>9} | {'ramp S4/S2':>11} | "
          f"{'S4* > S2* ?':>12}")
    for rt in pipe.regimes:
        if not len(df[df.regime.round(1) == round(rt, 1)]):
            continue
        print(f"{rt:>7} | "
              f"{100 * (z(rt,'S3')/z(rt,'S1') - 1):>+8.2f}% "
              f"{100 * (z(rt,'S4')/z(rt,'S2') - 1):>+8.2f}% "
              f"{100 * (z(rt,'S5')/z(rt,'S4') - 1):>+8.2f}% | "
              f"{z(rt,'S4','MaxRamp')/z(rt,'S2','MaxRamp'):>10.2f}x | "
              f"{analysis.certified(z(rt,'S4','lower_bound'), z(rt,'S4'), z(rt,'S2','lower_bound'), z(rt,'S2')):>12}")

    g = df.gap_rel.dropna()
    if len(g):
        print(f"\nattained gap: median {100*g.median():.2f}%  "
              f"range {100*g.min():.2f}%-{100*g.max():.2f}%  "
              f"met the {100*C.GAP_REL:.0f}% criterion: {(g <= C.GAP_REL).sum()} of {len(g)}")
    fair = df[df.config.isin(["S2", "S4", "S5"])]
    if len(fair):
        print(f"fairness: disparity {fair.Disparity_Mean.min():.4f}-"
              f"{fair.Disparity_Mean.max():.4f}, ex-post JFI mean {fair.Real_JFI.mean():.6f}")
    print("=" * 72)


def _record_environment(paths):
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "seed": C.SEED}
    for mod in ("numpy", "pandas", "scipy", "pulp", "tensorflow", "xgboost", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    with open(paths / "environment_rerun.json", "w") as fh:
        json.dump(env, fh, indent=2)
    return env


# ─────────────────────────────────────────────── commands

def cmd_check(args):
    paths, scada = _load(args)
    _pipeline(paths, scada, args)
    print(json.dumps(_record_environment(paths), indent=2))


def cmd_forecast(args):
    from clpu import forecasting
    paths, scada = _load(args)
    bilstm, _ = forecasting.run(scada, paths.out, force=args.retrain)
    print(bilstm.round(4).to_string(index=False))
    if args.benchmark:
        print("\narchitecture benchmark (LSTM, GRU against BiLSTM):")
        print(forecasting.benchmark(scada, paths.out, force=args.retrain)
              .round(4).to_string(index=False))


def cmd_table5(args):
    paths, scada = _load(args)
    pipe = _pipeline(paths, scada, args)
    inst = [(rt, cfg) for rt in pipe.regimes for cfg in C.CONFIGS]
    df = _sweep(pipe, paths, args, csv_name="table5_main.csv", instances=inst,
                disc_mode="double", label="Table 5")
    _summarise(df, pipe)
    _record_environment(paths)


def cmd_robustness(args):
    paths, scada = _load(args)
    pipe = _pipeline(paths, scada, args)
    inst = [(rt, cfg) for rt in pipe.regimes for cfg in C.CONFIGS]
    df = _sweep(pipe, paths, args, csv_name="table5_single.csv", instances=inst,
                disc_mode="single", label="Table 5 with C_disc applied once")
    _summarise(df, pipe)


def cmd_sensitivity(args):
    paths, scada = _load(args)
    pipe = _pipeline(paths, scada, args)
    out, rows = paths / "sensitivity_lambda.csv", []
    if os.path.exists(out):
        rows = pd.read_csv(out).to_dict("records")
    seen = {(round(float(r["regime"]), 1), r["config"], r["range"]) for r in rows}
    todo = [(rt, cfg, name) for rt in pipe.regimes for cfg in ("S3", "S4")
            for name in C.LAMBDA_RANGES if (round(rt, 1), cfg, name) not in seen]
    print(f"kernel sensitivity: {len(todo)} of {len(pipe.regimes) * 2 * len(C.LAMBDA_RANGES)} "
          f"cells to solve", flush=True)
    for i, (rt, cfg, name) in enumerate(todo, 1):
        prob, h = milp.build(pipe, rt, cfg, n_scenarios=args.n_scenarios,
                             eps_fair=args.eps_fair, lambdas=C.LAMBDA_RANGES[name])
        info = milp.solve(prob, time_limit=args.time_limit, gap=args.gap,
                          log_path=paths / f"cbc_lam_{cfg}_{rt:.0f}_{name}.log")
        rows.append(dict(regime=rt, config=cfg, range=name,
                         **analysis.report(prob, h, C.REPORT_N),
                         lower_bound=info["lower_bound"], gap_rel=info["gap_rel"],
                         solve_time_s=info["solve_time_s"]))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  [{i:>2}/{len(todo)}] {rt:>5} {cfg} {name:<13} "
              f"Z={rows[-1]['Z']:>11,.1f}  {info['solve_time_s']:>5.0f}s", flush=True)
    print(f"\nsaved -> {out}")


def cmd_all(args):
    cmd_check(args)
    cmd_table5(args)
    cmd_robustness(args)


# ─────────────────────────────────────────────── argument parsing

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fairness-constrained, CLPU-aware load-shedding optimisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("command",
                   choices=["check", "forecast", "table5", "sensitivity",
                            "robustness", "all"])
    p.add_argument("--data", default="", help="path to the SCADA CSV "
                   "(or set CLPU_DATA)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--n-scenarios", type=int, default=C.N_SCENARIOS, dest="n_scenarios")
    p.add_argument("--time-limit", type=int, default=C.TIME_LIMIT, dest="time_limit")
    p.add_argument("--gap", type=float, default=C.GAP_REL)
    p.add_argument("--eps-fair", type=float, default=C.EPS_FAIR, dest="eps_fair")
    p.add_argument("--k-sigma", type=float, default=C.K_SIGMA, dest="k_sigma")
    p.add_argument("--retrain", action="store_true",
                   help="retrain the forecaster instead of using the cached output")
    p.add_argument("--benchmark", action="store_true",
                   help="also run the LSTM and GRU benchmark (forecast command)")
    p.add_argument("--any-dataset", action="store_true", dest="any_dataset",
                   help="skip the integrity checks and run on a different record; "
                        "the reported numbers will not be reproduced")
    args = p.parse_args(argv)

    if int(pulp_major()) >= 4:
        sys.exit("This package targets the PuLP 3.x modelling API. PuLP 4 rewrote "
                 "LpVariable and the models here will not build under it. "
                 "Install the pinned requirements: pip install -r requirements.txt")

    {"check": cmd_check, "forecast": cmd_forecast, "table5": cmd_table5,
     "sensitivity": cmd_sensitivity, "robustness": cmd_robustness,
     "all": cmd_all}[args.command](args)


def pulp_major() -> int:
    import pulp
    v = getattr(pulp, "__version__", "0")
    return int(v.split(".")[0]) if v and v[0].isdigit() else 0


if __name__ == "__main__":
    main()
