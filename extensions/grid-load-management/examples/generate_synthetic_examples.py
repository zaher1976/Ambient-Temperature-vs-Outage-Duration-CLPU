#!/usr/bin/env python3
"""Deterministic, fully synthetic feeder inputs for illustrating the Grid Load Management MILP.

The generator does NOT read, sample, fit, or derive data from SCADA.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

SEED = 314159

def generate(case: str) -> dict:
    if case not in {"small", "day24"}:
        raise ValueError("case must be 'small' or 'day24'")
    F, T, S = (3, 8, 2) if case == 'small' else (5, 24, 3)
    rnd = random.Random(SEED + (0 if case == 'small' else 24))
    caps = [24.0 + 3.0*f for f in range(F)]
    forecast=[]
    for f in range(F):
        row=[]
        for t in range(T):
            # Artificial daily pattern, feeder diversity, and deterministic jitter.
            shape=1.0 + 0.18*math.sin((2*math.pi*t/max(T,8))-0.8+0.17*f)
            val=(11.0+2.6*f)*shape + rnd.uniform(-0.45,0.45)
            row.append(round(max(4.0,val),3))
        forecast.append(row)
    scenario=[]
    for s in range(S):
        bias = (s-(S-1)/2)*0.09
        srows=[]
        for f in range(F):
            srows.append([round(max(1.0,v*(1.0+bias)+rnd.uniform(-0.2,0.2)),3)
                          for v in forecast[f]])
        scenario.append(srows)
    supply=[]
    for t in range(T):
        total=sum(forecast[f][t] for f in range(F))
        # Scarcity is intentionally synthetic and makes optimization meaningful.
        ratio=0.68 + 0.05*math.sin(t*2*math.pi/max(T,8)+0.4)
        supply.append(round(ratio*total,3))
    return {
      "metadata": {
         "synthetic": True,
         "dataset_id": "GRIDLOAD_SYNTHETIC_"+case.upper()+"_V1",
         "source": "DETERMINISTIC_FORMULAS_ONLY_NO_SCADA",
         "seed": SEED + (0 if case == 'small' else 24),
         "time_step_hours": 1,
         "notes": "Artificial power traces; NOT field-derived; DO NOT cite as empirical outcomes."
      },
      "feeder_names": [f"SYNTHETIC_FEEDER_{f+1}" for f in range(F)],
      "forecast_mw": forecast,
      "scenario_demand_mw": scenario,
      "available_supply_mw": supply,
      "pmax_mw": caps,
      "initial_on": [0]*F,
      "clpu_kernel": [0.30,0.15,0.05],
      "fairness_epsilon": 1.0,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default=str(Path(__file__).resolve().parent))
    a=ap.parse_args()
    dst=Path(a.output_dir);dst.mkdir(parents=True,exist_ok=True)
    for case in ('small','day24'):
        path=dst/f'synthetic_{case}.json'
        path.write_text(json.dumps(generate(case),indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        print(f'Generated {path}')

if __name__=='__main__':main()
