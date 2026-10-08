"""What is read off a solved schedule.

Everything here is a measurement of the returned solution, not a further
optimisation: the reported metrics, the reconnection counts behind Section 5.4,
the binding shares behind Section 5.10, and the per-scenario costs the paired
significance tests use.
"""
from __future__ import annotations

import numpy as np
import pulp

from . import config as C


def report(prob, handles, report_n: int | None = None) -> dict:
    """Objective, maximum ramping, mean disparity and ex-post Jain index."""
    F = handles["govs"]
    N = handles["n_scenarios"]
    m = min(report_n or N, N)
    S, R = handles["vars"]["S"], handles["vars"]["R"]

    max_ramp = max(pulp.value(R[t, s]) for t in range(C.T_HORIZON) for s in range(m))
    disp, jfi = [], []
    for s in range(m):
        x = np.array([pulp.value(S[f, C.T_HORIZON - 1, s]) for f in F], dtype=float)
        disp.append(x.max() - x.min())
        denom = len(x) * (x ** 2).sum()
        jfi.append((x.sum() ** 2) / denom if denom > 0 else np.nan)

    return dict(
        Z=pulp.value(prob.objective),
        MaxRamp=float(max_ramp),
        Disparity_Mean=float(np.mean(disp)),
        Real_JFI=float(np.nanmean(jfi)),
    )


def schedule_statistics(handles) -> dict:
    """Reconnection events and served feeder-hours, averaged over scenarios.

    The structural floor is one reconnection per feeder at the start of the
    horizon, |F| = 5, because every feeder enters de-energised (u_{f,-1,s} = 0).
    Served feeder-hours run from 0 to |F| * T = 120.
    """
    F = handles["govs"]
    N = handles["n_scenarios"]
    u, v = handles["vars"]["u"], handles["vars"]["v"]
    recon, served = [], []
    for s in range(N):
        recon.append(sum(pulp.value(v[f, t, s]) or 0.0
                         for f in F for t in range(C.T_HORIZON)))
        served.append(sum(pulp.value(u[f, t, s]) or 0.0
                          for f in F for t in range(C.T_HORIZON)))
    return dict(
        recon_mean=float(np.mean(recon)),
        recon_min=float(np.min(recon)),
        recon_max=float(np.max(recon)),
        served_fh_mean=float(np.mean(served)),
        structural_floor=float(len(F)),
        max_feeder_hours=float(len(F) * C.T_HORIZON),
    )


def binding_shares(handles, tol: float = 1e-6) -> dict:
    """Percentage of each constraint group that is tight at the returned solution."""
    out = {}
    for group, cons in handles["groups"].items():
        if not cons:
            out[f"bind_{group.rstrip('_')}"] = float("nan")
            continue
        n_bind = 0
        for c in cons:
            slack = c.value()
            if slack is not None and abs(slack) < tol:
                n_bind += 1
        out[f"bind_{group.rstrip('_')}"] = 100.0 * n_bind / len(cons)
    return out


def scenario_costs(handles) -> list[float]:
    """The |Omega| per-scenario objective contributions, for the paired tests."""
    return [float(pulp.value(handles["scenario_costs"][s]))
            for s in range(handles["n_scenarios"])]


def certified(lb_a: float | None, z_a: float, lb_b: float | None, z_b: float) -> str:
    """Can the sign of A minus B be certified from the solver's own bounds?

    An incumbent is an upper bound on its optimum and the solver's bound a lower
    one, so A* > B* is established only when lower(A) exceeds incumbent(B).
    """
    if lb_a is None or lb_b is None:
        return "unknown"
    if lb_a > z_b:
        return "A > B"
    if z_a < lb_b:
        return "A < B"
    return "overlapping"
