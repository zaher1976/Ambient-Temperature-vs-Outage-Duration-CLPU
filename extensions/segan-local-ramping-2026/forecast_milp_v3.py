"""Experimental here-and-now stochastic MILP with multi-period CLPU and uninterrupted restoration history.

All powers are MW, hourly steps assumed. CLPU kernel coefficients are
synthetic sensitivity inputs, NOT calibrated field measurements.
Requires PuLP 2.x/3.x (legacy API).
"""
import numpy as np
import pulp


def build_model(forecast, scenario_demand, available_supply, pmax, *,
                clpu_kernel, fairness_epsilon=None,
                curtailment_weight=1.0, switching_weight=0.0,
                ramp_weight=0.0, initial_on=None):
    fc = np.asarray(forecast, dtype=float)
    d = np.asarray(scenario_demand, dtype=float)
    supply = np.asarray(available_supply, dtype=float)
    cap = np.asarray(pmax, dtype=float)
    alpha = np.asarray(clpu_kernel, dtype=float)
    if fc.ndim != 2 or d.ndim != 3 or d.shape[1:] != fc.shape:
        raise ValueError('forecast [F,T] and scenario_demand [S,F,T] required')
    S,F,T = d.shape
    if supply.shape != (T,) or cap.shape != (F,):
        raise ValueError('supply [T] and pmax [F] required')
    if alpha.ndim == 1:
        alpha = np.tile(alpha, (F,1))
    if alpha.ndim != 2 or alpha.shape[0] != F or alpha.shape[1] < 1:
        raise ValueError('clpu_kernel must be [K] or [F,K]')
    K = alpha.shape[1]
    init = np.zeros(F, dtype=int) if initial_on is None else np.asarray(initial_on, dtype=int)
    if init.shape != (F,) or not np.isin(init,[0,1]).all():
        raise ValueError('initial_on must be binary [F]')
    if not np.isfinite(fc).all() or not np.isfinite(d).all() or not np.isfinite(supply).all() or not np.isfinite(cap).all() or not np.isfinite(alpha).all():
        raise ValueError('nonfinite inputs')
    if np.any(d < 0) or np.any(fc < 0) or np.any(supply < 0) or np.any(cap <= 0) or np.any(alpha < 0):
        raise ValueError('invalid negative values or nonpositive capacity')
    if fairness_epsilon is not None and fairness_epsilon < 0:
        raise ValueError('fairness_epsilon must be nonnegative')
    prob = pulp.LpProblem('ForecastDriven_MultiPeriodCLPU_v3', pulp.LpMinimize)
    u = {(f,t):pulp.LpVariable(f'u_{f}_{t}',cat='Binary') for f in range(F) for t in range(T)}
    v = {(f,t):pulp.LpVariable(f'v_{f}_{t}',cat='Binary') for f in range(F) for t in range(T)}
    served = {(s,f,t):pulp.LpVariable(f'served_{s}_{f}_{t}',lowBound=0) for s in range(S) for f in range(F) for t in range(T)}
    shed = {(s,f,t):pulp.LpVariable(f'shed_{s}_{f}_{t}',lowBound=0) for s in range(S) for f in range(F) for t in range(T)}
    ramp = {(s,t):pulp.LpVariable(f'ramp_{s}_{t}',lowBound=0) for s in range(S) for t in range(1,T)}
    for f in range(F):
        for t in range(T):
            prev = int(init[f]) if t == 0 else u[f,t-1]
            prob += v[f,t] >= u[f,t] - prev
            prob += v[f,t] <= u[f,t]
            prob += v[f,t] <= 1-prev
    zvars = {}
    for s in range(S):
        for t in range(T):
            for f in range(F):
                # CLPU is counted only while the feeder is energised. A
                # past restoration event contributes a known-coefficient term
                # multiplied by current u, linearised with binary AND z.
                extras = []
                for k in range(min(K,t+1)):
                    if alpha[f,k] == 0: continue
                    if k == 0:
                        z = v[f,t]
                    else:
                        z = pulp.LpVariable(f'z_{f}_{t}_{k}', cat='Binary') if s == 0 else zvars[(f,t,k)]
                        if s == 0:
                            zvars[(f,t,k)] = z
                            # z is active only if the restoration at t-k
                            # has remained continuously energised through t.
                            prob += z <= v[f,t-k]
                            for j in range(t-k+1, t+1):
                                prob += z <= u[f,j]
                            prob += z >= v[f,t-k] + pulp.lpSum(u[f,j] for j in range(t-k+1, t+1)) - k
                    extras.append(float(d[s,f,t])*float(alpha[f,k])*z)
                extra = pulp.lpSum(extras)
                requested = float(d[s,f,t])*u[f,t] + extra
                prob += served[s,f,t] <= requested
                prob += served[s,f,t] <= float(cap[f])
                # shed is all baseline demand not served plus CLPU not served
                prob += shed[s,f,t] == float(d[s,f,t]) + extra - served[s,f,t]
            prob += pulp.lpSum(served[s,f,t] for f in range(F)) <= float(supply[t])
            if t > 0:
                delta = pulp.lpSum(served[s,f,t]-served[s,f,t-1] for f in range(F))
                prob += ramp[s,t] >= delta
                prob += ramp[s,t] >= -delta
        if fairness_epsilon is not None:
            for f in range(F):
                for j in range(f+1,F):
                    lhs = pulp.lpSum(served[s,f,t]*(1/float(cap[f])) - served[s,j,t]*(1/float(cap[j])) for t in range(T))
                    prob += lhs <= fairness_epsilon
                    prob += -lhs <= fairness_epsilon
    prob += (curtailment_weight/S)*pulp.lpSum(shed.values()) + switching_weight*pulp.lpSum(v.values()) + (ramp_weight/S)*pulp.lpSum(ramp.values())
    return prob, dict(u=u,v=v,served=served,shed=shed,ramp=ramp,z=zvars,scenario_count=S,clpu_kernel=alpha)


def solve_and_check(prob, *, seconds=60, gap=0.03, msg=False):
    solver = pulp.PULP_CBC_CMD(msg=msg,timeLimit=seconds,gapRel=gap)
    prob.solve(solver)
    return {'status':pulp.LpStatus[prob.status], 'objective':pulp.value(prob.objective),
            'warning':'CBC status is not a certified numerical optimality-gap report; inspect solver logs.'}
