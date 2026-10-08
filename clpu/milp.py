"""The fairness-constrained, CLPU-aware MILP of Equations (1)-(18).

The five configurations differ in exactly two switches and are identical in every
other respect:

=========  ==================================  ==================
Config     CLPU surge kernel                   Fairness constraint
=========  ==================================  ==================
S1         none                                no
S2         none                                yes
S3         fixed, thermally blind (0.30)       no
S4         thermal kernel, Eq. (10)            yes
S5         equal-weight blend of the two       yes
=========  ==================================  ==================
"""
from __future__ import annotations

import re
import time

import pulp

from . import config as C
from .pipeline import Pipeline

# ─────────────────────────────────────────────── surge kernels


def gamma_thermal(temp: float, lambdas=None) -> float:
    """Temperature-driven surge coefficient (Eq. 10), via the bin indicator."""
    lam = lambdas or C.LAMBDA_BY_BIN
    for i in range(len(lam)):
        if C.TEMP_BINS[i] <= temp < C.TEMP_BINS[i + 1]:
            return lam[i]
    return lam[-1]


def gamma_duration(dt_out: float = 0.0) -> float:
    """Duration-style benchmark kernel, evaluated at dt_out = 0 for S3.

    At dt_out = 0 this is a thermally invariant surge of 0.30 (a 1.30x multiplier),
    which is exactly the coefficient of the lowest temperature bin — the reason S5
    reduces to S4 identically for every hour below 30 degC (Section 5.5).
    """
    return 0.3 + 0.5 * min(dt_out, 10.0) / 10.0


def gamma_blend(temp: float, dt_out: float = 0.0, lambdas=None) -> float:
    """Equal-weight blend of the duration-style and thermal kernels (S5)."""
    return 0.5 * gamma_duration(dt_out) + 0.5 * gamma_thermal(temp, lambdas)


def surge_coefficient(config: str, temp: float, lambdas=None) -> float:
    if config in ("S1", "S2"):
        return 0.0
    if config == "S3":
        return gamma_duration(0.0)
    if config == "S4":
        return gamma_thermal(temp, lambdas)
    if config == "S5":
        return gamma_blend(temp, 0.0, lambdas)
    raise ValueError(f"unknown configuration {config!r}; expected one of {C.CONFIGS}")


# ─────────────────────────────────────────────── the model


def build(pipe: Pipeline, regime: float, config: str = "S4", *,
          n_scenarios: int = C.N_SCENARIOS, eps_fair: float = C.EPS_FAIR,
          lambdas=None, cost_scale=(1.0, 1.0, 1.0), disc_mode: str = "double"):
    """Build one MILP instance.

    ``disc_mode="double"`` reproduces the reported results, in which C_disc
    multiplies both the accumulation of Phi (Eq. 18) and its contribution to the
    objective (Eq. 2). ``"single"`` applies it once; Section 5.11 reports what
    changes. Everything else is identical between the two.

    Returns (problem, handles) where handles carries the variables, the per-scenario
    cost expressions and the constraint groups.
    """
    if disc_mode not in ("double", "single"):
        raise ValueError("disc_mode must be 'double' or 'single'")

    F = pipe.govs
    Cc = C.C_CURT * cost_scale[0]
    Cr = C.C_RAMP * cost_scale[1]
    Cd = C.C_DISC * cost_scale[2]
    inner = Cd if disc_mode == "double" else 1.0

    pmax = pipe.pmax
    scen = pipe.scenarios(regime, n_scenarios)
    win = pipe.window[regime]
    P_gen = {f: win[f]["avail"] * pmax[f] for f in F}
    temps = {f: win[f]["temp"] for f in F}

    N, T = n_scenarios, C.T_HORIZON
    prob = pulp.LpProblem(f"LoadMgmt_{config}_{regime:.0f}_{disc_mode}", pulp.LpMinimize)

    u, v, Pc, S, Phi, R = {}, {}, {}, {}, {}, {}
    for f in F:
        for s in range(N):
            for t in range(T):
                u[f, t, s] = pulp.LpVariable(f"u_{f}_{t}_{s}", cat="Binary")
                v[f, t, s] = pulp.LpVariable(f"v_{f}_{t}_{s}", lowBound=0, upBound=1)
                Pc[f, t, s] = pulp.LpVariable(f"curt_{f}_{t}_{s}", lowBound=0)
                S[f, t, s] = pulp.LpVariable(f"S_{f}_{t}_{s}", lowBound=0)
                Phi[f, t, s] = pulp.LpVariable(f"phi_{f}_{t}_{s}", lowBound=0)
    for t in range(T):
        for s in range(N):
            R[t, s] = pulp.LpVariable(f"R_{t}_{s}", lowBound=0)

    # objective (Eq. 2), kept per scenario so paired tests can be run afterwards
    scen_cost = {s: [] for s in range(N)}
    for s in range(N):
        for t in range(T):
            scen_cost[s] += [
                Cc * pulp.lpSum(Pc[f, t, s] for f in F),
                Cr * R[t, s],
                Cd * pulp.lpSum(Phi[f, t, s] for f in F),
            ]
    prob += (1.0 / N) * pulp.lpSum(x for s in range(N) for x in scen_cost[s])

    eff = {}
    for f in F:
        for s in range(N):
            for t in range(T):
                base = float(scen[f][s, t])
                prev = 0 if t == 0 else u[f, t - 1, s]          # u_{f,-1,s} = 0
                prob += (v[f, t, s] >= u[f, t, s] - prev), f"clpu_v1_{f}_{t}_{s}"   # Eq. 7
                prob += (v[f, t, s] <= u[f, t, s]), f"clpu_v2_{f}_{t}_{s}"          # Eq. 8
                if t > 0:
                    prob += (v[f, t, s] <= 1 - u[f, t - 1, s]), f"clpu_v3_{f}_{t}_{s}"  # Eq. 9

                gamma = surge_coefficient(config, temps[f][t], lambdas)             # Eq. 10
                surge = gamma * base
                # Eq. (11): u*v == v because v <= u is enforced, so this stays linear
                e = u[f, t, s] * base + surge * v[f, t, s]
                eff[f, t, s] = e

                prob += (Pc[f, t, s] <= e), f"clpu_c1_{f}_{t}_{s}"                  # Eq. 4
                prob += (Pc[f, t, s] <= base + surge), f"clpu_c2_{f}_{t}_{s}"       # Eq. 5
                prob += (e - Pc[f, t, s] <= pmax[f]), f"clpu_c3_{f}_{t}_{s}"        # Eq. 6

                served = e - Pc[f, t, s]
                if t == 0:                                                          # Eqs. 14-15
                    prob += (S[f, t, s] == served / pmax[f]), f"svc_{f}_{t}_{s}"
                    prob += (Phi[f, t, s] == inner * (1 - u[f, t, s]) * base), \
                            f"disc_{f}_{t}_{s}"
                else:
                    prob += (S[f, t, s] == S[f, t - 1, s] + served / pmax[f]), \
                            f"svc_{f}_{t}_{s}"
                    prob += (Phi[f, t, s] == Phi[f, t - 1, s]
                             + inner * (1 - u[f, t, s]) * base), f"disc_{f}_{t}_{s}"  # Eq. 18

    for s in range(N):                                                              # Eq. 3
        for t in range(T):
            prob += (
                pulp.lpSum(eff[f, t, s] - Pc[f, t, s] for f in F)
                <= sum(P_gen[f][t] for f in F)
            ), f"bal_{t}_{s}"

    for s in range(N):                                                              # Eqs. 12-13
        for t in range(1, T):
            lhs = pulp.lpSum(eff[f, t, s] - eff[f, t - 1, s] for f in F)
            prob += (R[t, s] >= lhs), f"ramp_p_{t}_{s}"
            prob += (R[t, s] >= -lhs), f"ramp_n_{t}_{s}"

    if config in ("S2", "S4", "S5"):                                                # Eqs. 16-17
        for t in range(T):
            for s in range(N):
                for f in F:
                    for j in F:
                        if f != j:
                            prob += (S[f, t, s] - S[j, t, s] <= eps_fair), \
                                    f"fair_{f}_{j}_{t}_{s}"

    savg = {}
    for f in F:
        savg[f] = pulp.LpVariable(f"Savg_{f}", lowBound=0)
        prob += (savg[f] == (1.0 / N) * pulp.lpSum(S[f, C.T_HORIZON - 1, s]
                                                   for s in range(N))), f"savg_{f}"

    handles = dict(
        vars=dict(u=u, v=v, S=S, R=R, Pcurt=Pc, Phi=Phi, savg=savg),
        scenario_costs={s: pulp.lpSum(scen_cost[s]) for s in range(N)},
        groups={
            g: [c for n_, c in prob.constraints.items() if n_.startswith(g)]
            for g in ("ramp_", "fair_", "clpu_")
        },
        n_scenarios=N,
        govs=F,
    )
    return prob, handles


# ─────────────────────────────────────────────── solving


def solve(prob, *, time_limit: int = C.TIME_LIMIT, gap: float = C.GAP_REL,
          log_path: str | None = None) -> dict:
    """Solve, and report the lower bound and attained gap read from the CBC log.

    The attained gap matters: under this formulation most instances stop at a gap
    far wider than the prescribed criterion, and Section 5.9 shows that this
    measures the weakness of the dual bound rather than the quality of the
    incumbent. Without the log the bound cannot be recovered afterwards.
    """
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap,
                                 logPath=log_path))
    elapsed = round(time.time() - t0, 2)
    z = pulp.value(prob.objective)

    lb, rel = None, None
    if log_path:
        try:
            txt = open(log_path, errors="ignore").read()
            m = re.search(r"^Lower bound:\s+([-\d.eE+]+)", txt, re.M)
            if m:
                lb = float(m.group(1))
                rel = (z - lb) / abs(z) if z not in (None, 0) else None
            if lb is None and re.search(r"Result - Optimal solution found", txt):
                lb, rel = z, 0.0
        except FileNotFoundError:
            pass

    return dict(
        status=pulp.LpStatus[prob.status],
        objective=z,
        solve_time_s=elapsed,
        lower_bound=lb,
        gap_rel=rel,
        hit_cap=elapsed >= time_limit - 1,
        n_variables=prob.numVariables(),
        n_constraints=prob.numConstraints(),
    )
