#!/usr/bin/env python3
"""Run actual archived v4 MILP against PUBLIC SYNTHETIC example inputs.

This demo is not a reproduction of the Iraq SCADA cases or published percentages.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pulp
from forecast_milp_v4 import build_model_v4
from sensitivity_metrics import arrays_from_model, postprocess, max_model_residual


def run(case='small',time_limit=45,gap=0.001,output_dir='synthetic_run_results'):
    data=json.loads((ROOT/'examples'/f'synthetic_{case}.json').read_text(encoding='utf-8'))
    if data.get('metadata',{}).get('synthetic') is not True:
        raise ValueError('Not an explicitly marked synthetic dataset')
    fc=np.array(data['forecast_mw'],dtype=float)
    demand=np.array(data['scenario_demand_mw'],dtype=float)
    supply=np.array(data['available_supply_mw'],dtype=float)
    caps=np.array(data['pmax_mw'],dtype=float)
    initial=np.array(data['initial_on'],dtype=int)
    alpha=np.array(data['clpu_kernel'],dtype=float)
    S,F,T=demand.shape
    assert fc.shape==(F,T) and supply.shape==(T,) and caps.shape==(F,)
    assert len(data['feeder_names'])==F and np.isfinite(demand).all()
    assert np.all(demand>=0) and np.all(supply>=0)
    eps=float(data['fairness_epsilon'])
    results={}
    # Use the same v4 objective, inputs and binary MILP variables for each weight.
    for name,local_weight in (('without_local_penalty',0.0),('with_local_penalty',0.01)):
        problem,v=build_model_v4(fc,demand,supply,caps,clpu_kernel=alpha,
                    fairness_epsilon=eps,curtailment_weight=1.0,
                    switching_weight=0.05,ramp_weight=0.001,
                    local_ramp_weight=local_weight,initial_on=initial)
        solver=pulp.PULP_CBC_CMD(msg=True,timeLimit=time_limit,gapRel=gap,threads=1,
                                options=['randomSeed 42'])
        if not solver.available():raise RuntimeError('CBC not available in PuLP installation')
        status_code=problem.solve(solver)
        status=pulp.LpStatus[status_code]
        if status!='Optimal':
            raise RuntimeError(f'{name}: solver ended as {status}; rerun with a larger --time-limit')
        u,restoration,served,shed=arrays_from_model(v,demand.shape)
        metric=postprocess(demand,supply,caps,alpha,initial,u,restoration,served,shed,
                           fairness_eps=eps)
        diag=max_model_residual(problem)
        # Validate directly from solved variables instead of untested assumptions.
        for key in ['max_balance_residual_mw','max_supply_excess_mw',
                    'max_feeder_cap_excess_mw','max_requested_excess_mw',
                    'max_fairness_violation','max_restoration_indicator_residual']:
            if metric[key]>1e-4:raise AssertionError((name,key,metric[key]))
        if diag['max_constraint_violation'] is not None and diag['max_constraint_violation']>1e-4:
            raise AssertionError((name,'constraint_violation',diag['max_constraint_violation']))
        metrics={'dataset_id':data['metadata']['dataset_id'], 'source':'SYNTHETIC_ONLY',
                 'solver_status_pulp':status,'solver_gap_target':gap,'solver_time_limit_s':time_limit,
                 'scenario_count':S,'feeders':F,'hours':T,
                 'lambda_local':local_weight,'objective':float(pulp.value(problem.objective)),
                 **metric,**diag}
        results[name]=metrics
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        with (out/f'{case}_{name}_schedule.csv').open('w',newline='',encoding='utf8') as fp:
            writer=csv.writer(fp)
            writer.writerow(['scenario','feeder','hour_1_based','served_MW','shed_MW',
                             'energized','restoration'])
            for s in range(S):
                for f in range(F):
                    for t in range(T):
                        writer.writerow([s+1,data['feeder_names'][f],t+1,
                                         round(float(served[s,f,t]),6),round(float(shed[s,f,t]),6),
                                         round(float(u[f,t]),6),round(float(restoration[f,t]),6)])
        print(f'{name}: local TV {metric["local_ramp_mw_total_variation"]:.3f} MW, '
              f'unserved {metric["unserved_mwh"]:.3f} MWh, status {status}')
    out=Path(output_dir)
    (out/f'{case}_comparison.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    print(f'PASS: synthetic {case} v4 MILP, reconnection, CLPU, physical checks, two weights')
    print(f'Results saved to {out.resolve()}')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['small','day24'],default='small')
    p.add_argument('--time-limit',type=int,default=45)
    p.add_argument('--gap',type=float,default=0.001)
    p.add_argument('--output-dir',default='synthetic_run_results')
    a=p.parse_args()
    run(a.case,a.time_limit,a.gap,a.output_dir)
