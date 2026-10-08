"""From the record to the optimisation inputs: thermal regimes, 24-hour windows,
capacity bounds and demand scenarios.

One synchronous national window per thermal regime: the same hours for every
governorate, because the power-balance constraint (Eq. 3) sums served load across
governorates within the hour, so a per-governorate window would make that sum
meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class Pipeline:
    """Everything the MILP needs, derived once from the record and cached."""

    def __init__(self, scada: pd.DataFrame, forecast: pd.DataFrame,
                 k_sigma: float = C.K_SIGMA):
        s = scada.copy()
        s["avail"] = s["Supply_Hours"] / 24.0

        g = s.groupby("gov")["demand_mw"].agg(mu="mean", sigma="std")
        self.govs = list(C.GOVERNORATES)
        self.mu = g["mu"].to_dict()
        self.sigma = g["sigma"].to_dict()
        #: P_max(f) = mu + k*sigma  (Eq. 1) — a statistically derived operational
        #: bound, not nameplate capacity.
        self.pmax = {f: self.mu[f] + k_sigma * self.sigma[f] for f in self.govs}

        # relative forecast residuals, resampled to build demand scenarios
        self.rel_dev = {}
        for f in self.govs:
            d = forecast[forecast.Governorate == f]
            self.rel_dev[f] = (
                (d.Demand_Actual.values - d.Demand_Forecast_BiLSTM.values)
                / np.maximum(d.Demand_Forecast_BiLSTM.values, 1e-6)
            )

        # thermal regimes are percentiles of the record, never assumed
        self.regimes = [
            round(float(x), 1)
            for x in np.percentile(scada.temp_c.dropna(), C.REGIME_PERCENTILES)
        ]

        wide_t = (
            s.pivot_table(index="ts", columns="gov", values="temp_c")
            .dropna(how="any")
            .sort_index()
        )[self.govs]
        self.shared_timestamps = len(wide_t)
        national = wide_t.mean(axis=1)
        roll = national.rolling(C.T_HORIZON).mean()
        by_ts = {
            f: s[s.gov == f].sort_values("ts").set_index("ts") for f in self.govs
        }

        self.window, self.window_span = {}, {}
        for rt in self.regimes:
            pos = int(np.nanargmin(np.abs(roll.values - rt)))
            ts_win = wide_t.index[pos - C.T_HORIZON + 1: pos + 1]
            self.window[rt] = {}
            for f in self.govs:
                sl = by_ts[f].reindex(ts_win)
                if sl[["demand_mw", "temp_c", "avail"]].isna().any().any():
                    raise ValueError(f"gap in the record for {f} at regime {rt}")
                self.window[rt][f] = dict(
                    demand=sl["demand_mw"].values,
                    temp=sl["temp_c"].values,
                    avail=sl["avail"].values,
                )
            self.window_span[rt] = (
                str(ts_win[0])[:16],
                str(ts_win[-1])[:16],
                float(national.reindex(ts_win).mean()),
            )

        self._scen_cache: dict = {}

    # ─────────────────────────────────────────── scenarios
    def scenarios(self, regime: float, n_scen: int) -> dict:
        """Bootstrapped demand scenarios for one regime.

        The seed depends on (SEED, regime, governorate, n_scen), so a given regime
        and scenario count always produce the same sample.
        """
        key = (round(regime, 1), n_scen)
        if key in self._scen_cache:
            return self._scen_cache[key]
        out = {}
        for gi, f in enumerate(self.govs):
            rng = np.random.default_rng([C.SEED, int(round(regime * 10)), gi, n_scen])
            base = self.window[regime][f]["demand"]
            rd = self.rel_dev[f]
            out[f] = np.stack(
                [
                    base * (1 + rng.choice(rd, size=C.T_HORIZON, replace=True))
                    for _ in range(n_scen)
                ]
            )
        self._scen_cache[key] = out
        return out

    # ─────────────────────────────────────────── self-checks
    def verify_windows(self, verbose: bool = True) -> bool:
        """Check the rebuilt windows against the run that produced the results.

        A mismatch means the pipeline is not the one behind the reported tables,
        and nothing computed afterwards is comparable with them.
        """
        ok = True
        for rt, (want_start, want_mean) in C.EXPECTED_WINDOWS.items():
            near = min(self.regimes, key=lambda r: abs(r - rt))
            if abs(near - rt) > 0.05 or near not in self.window_span:
                if verbose:
                    print(f"   {rt:>5} C  MISSING (regimes rebuilt as {self.regimes})")
                ok = False
                continue
            start, end, mean = self.window_span[near]
            good = start == want_start and abs(mean - want_mean) < 0.05
            ok &= good
            if verbose:
                print(
                    f"   {near:>5} C  {start} -> {end}  national mean {mean:5.1f} C   "
                    f"{'OK' if good else 'MISMATCH (expected %s, %.1f C)' % (want_start, want_mean)}"
                )
        return ok

    def describe(self) -> str:
        lines = [
            f"thermal regimes: {self.regimes}",
            f"shared hourly timestamps across all five governorates: {self.shared_timestamps:,}",
            "",
            "P_max(f) = mu + k*sigma",
        ]
        for f in self.govs:
            lines.append(
                f"  {f:7} mu={self.mu[f]:8.2f}  sigma={self.sigma[f]:7.2f}  "
                f"P_max={self.pmax[f]:8.2f} MW"
            )
        return "\n".join(lines)
