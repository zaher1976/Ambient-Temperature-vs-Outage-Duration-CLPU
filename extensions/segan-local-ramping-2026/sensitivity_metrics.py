"""Independent, solver-agnostic metrics for frozen SEGAN sensitivity.

No retraining and no new scenarios. Hourly samples imply sums of MW
across 1-h intervals can be stated as MWh for energy quantities.
"""
from __future__ import annotations
import numpy as np


def postprocess(demand, supply, capacity, alpha, initial_on, u, v, served, shed, fairness_eps=0.1):
    demand = np.asarray(demand, dtype=float)
    supply = np.asarray(supply, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    initial_on = np.asarray(initial_on, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    served = np.asarray(served, dtype=float)
    shed = np.asarray(shed, dtype=float)
    S, F, T = demand.shape
    assert served.shape == shed.shape == demand.shape
    assert u.shape == v.shape == (F, T)
    assert capacity.shape == (F,) and supply.shape == (T,)
    if alpha.ndim == 1:
        alpha = np.tile(alpha, (F, 1))
    assert alpha.shape[0] == F
    assert np.isfinite(served).all() and np.isfinite(shed).all()
    # Evaluate restoration events without trusting the model's auxiliary z values.
    v_expected = np.zeros((F, T), dtype=float)
    for f in range(F):
        for t in range(T):
            v_expected[f, t] = max(0.0, u[f, t] - (initial_on[f] if t == 0 else u[f, t - 1]))
    excess = np.zeros_like(demand)
    for f in range(F):
        for t in range(T):
            for k in range(min(alpha.shape[1], t + 1)):
                if alpha[f, k] == 0: continue
                # Temporal AND of current and intermediate energization states.
                active = v[f, t-k] * np.prod(u[f, t-k+1:t+1])
                excess[:, f, t] += demand[:, f, t] * alpha[f, k] * active
    net = served.sum(axis=1)
    normalised_cumulative = (served / capacity[None, :, None]).sum(axis=2)
    den = F * np.sum(normalised_cumulative**2, axis=1)
    jfi = np.divide(np.square(np.sum(normalised_cumulative, axis=1)), den,
                    out=np.full_like(den, np.nan), where=den > 0)
    fair = max((abs(normalised_cumulative[:,f] - normalised_cumulative[:,j])).max()
               for f in range(F) for j in range(f+1,F))
    ramp_f = np.abs(np.diff(served, axis=2))
    ramp_aggregate = np.abs(np.diff(net, axis=1))
    return {
        'local_ramp_mw_total_variation': float(ramp_f.sum() / S),
        'aggregate_ramp_mw_total_variation': float(ramp_aggregate.sum() / S),
        'unserved_mwh': float(shed.sum() / S),
        'clpu_mwh': float(excess.sum() / S),
        'served_mwh': float(served.sum() / S),
        'mean_jfi_posthoc': float(np.nanmean(jfi)) if not np.isnan(jfi).all() else None,
        'max_balance_residual_mw': float(np.max(np.abs(shed - (demand + excess - served)))),
        'max_supply_excess_mw': float(np.maximum(net - supply[None,:],0).max()),
        'max_feeder_cap_excess_mw': float(np.maximum(served - capacity[None,:,None],0).max()),
        'max_requested_excess_mw': float(np.maximum(served - (demand * u[None,:,:] + excess),0).max()),
        'max_fairness_violation': float(max(0, fair - fairness_eps)),
        'max_restoration_indicator_residual': float(np.max(abs(v - v_expected))),
        'mean_scenario_local_ramp_mw': float(ramp_f.sum(axis=(1,2)).mean()),
        'min_jfi_scenario_posthoc': float(np.nanmin(jfi)) if not np.isnan(jfi).all() else None,
    }


def max_model_residual(prob):
    values = []
    for c in prob.constraints.values():
        x = c.value()
        if x is None: continue
        if c.sense == -1: values.append(max(0, x))
        elif c.sense == 1: values.append(max(0, -x))
        else: values.append(abs(x))
    ints = [abs(v.varValue - round(v.varValue)) for v in prob.variables()
            if v.cat == 'Integer' and v.varValue is not None]
    return {'max_constraint_violation':max(values, default=None),
            'max_integer_violation':max(ints, default=None)}


def arrays_from_model(mapping, shape):
    import pulp  # deferred; unit-test metric code does not need PuLP
    S,F,T = shape
    def value(obj):
        x = pulp.value(obj)
        if x is None or not np.isfinite(x): raise ValueError('Unusable MILP incumbent')
        return float(x)
    u = np.array([[value(mapping['u'][f,t]) for t in range(T)] for f in range(F)])
    v = np.array([[value(mapping['v'][f,t]) for t in range(T)] for f in range(F)])
    served = np.array([[[value(mapping['served'][s,f,t]) for t in range(T)]
                       for f in range(F)] for s in range(S)])
    shed = np.array([[[value(mapping['shed'][s,f,t]) for t in range(T)]
                     for f in range(F)] for s in range(S)])
    return u,v,served,shed
