#!/usr/bin/env python3
"""Create auditable case-level CSV and tables; never silently fill missing results."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
MULTS=[0.0,0.1,0.25,0.5,1.0,2.0]
COLUMNS=['window_id','fold','analysis_group','lambda_multiplier','lambda_local',
 'cbc_termination','pulp_status','validation','cbc_reported_relative_gap',
 'elapsed_s','local_ramp_mw_total_variation','aggregate_ramp_mw_total_variation',
 'unserved_mwh','clpu_mwh','served_mwh','mean_jfi_posthoc',
 'max_constraint_violation','max_integer_violation', 'max_balance_residual_mw',
 'max_fairness_violation','max_supply_excess_mw']
METRICS=['local_ramp_mw_total_variation','aggregate_ramp_mw_total_variation','unserved_mwh','clpu_mwh','mean_jfi_posthoc']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-dir',type=Path,default=HERE/'CBC_FULL_MILP_RESULTS')
    a=ap.parse_args(); d=a.results_dir
    cases=[]
    for p in sorted((d/'cases').glob('*.json')):
        obj=json.loads(p.read_text(encoding='utf-8'))
        row={k:obj.get(k) for k in COLUMNS}
        row.update({k:obj.get('metrics',{}).get(k) for k in METRICS})
        row.update({k:obj.get('metrics',{}).get(k) for k in COLUMNS if k.startswith('max_')})
        row.update({k:obj.get('diagnostics',{}).get(k) for k in COLUMNS if k.startswith('max_')})
        row['usable_feasible']=(obj.get('validation')=='FEASIBLE_CHECKED' and
          obj.get('cbc_termination') in ('CBC_REPORTED_OPTIMAL','TIME_LIMIT'))
        row['cbc_optimal_reported']=(obj.get('validation')=='FEASIBLE_CHECKED' and
          obj.get('cbc_termination')=='CBC_REPORTED_OPTIMAL')
        row['file']=p.name
        cases.append(row)
    d.mkdir(exist_ok=True,parents=True)
    keys=COLUMNS+['usable_feasible','cbc_optimal_reported','file']
    with open(d/'case_metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(cases)
    groups=defaultdict(list)
    for r in cases:
        if isinstance(r['window_id'],int) and 1<=r['window_id']<=11:
            groups[round(float(r['lambda_multiplier']),8)].append(r)
    summary=[]
    for mult in MULTS:
        group=groups.get(mult,[])
        ids={r['window_id'] for r in group}
        feasible=[r for r in group if r['usable_feasible']]
        complete=(len(ids)==11 and len(feasible)==11)
        record={'lambda_multiplier':mult,'lambda_local':mult*.01,
                'primary_windows_present':len(ids),'primary_windows_feasible':len(feasible),
                'primary_windows_cbc_optimal_reported':sum(r['cbc_optimal_reported'] for r in feasible),
                'primary_windows_time_limit_feasible':sum(r['cbc_termination']=='TIME_LIMIT' for r in feasible),
                'complete_for_11_windows':complete}
        for key in METRICS:
            record['mean_'+key]=(round(float(np.mean([r[key] for r in feasible])),9)
              if complete and all(r.get(key) is not None for r in feasible) else '')
        summary.append(record)
    with open(d/'primary_11_windows_summary.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    n=len(cases);n_ok=sum(r['usable_feasible'] for r in cases)
    complete=[r for r in summary if r['complete_for_11_windows']]
    doc=['# Frozen grid load-management full-MILP lambda sensitivity audit',
         '',f'- Stored case results: {n} of 72 expected.',f'- Feasibility-checked cases: {n_ok}.',
         f'- Complete primary-window multiplier groups: {len(complete)} of 6.',
         '- W00 is excluded from primary-window averages.',
         '- All ramp metrics are recalculated from served-power time series, not auxiliary ramp variables.',
         '- Full optimisation is over switching variables; results with time limits are feasible incumbents, not global optima.',
         '- No missing or invalid cases are imputed. Incomplete groups have empty average metrics.',
         '', '## Primary-window completeness', '',
         '| Multiplier | Cases | Feasible | CBC optimal reported | Time-limit feasible | Publishable numeric mean? |',
         '|---:|---:|---:|---:|---:|---|']
    for r in summary:
        doc.append(f"| {r['lambda_multiplier']:g} | {r['primary_windows_present']}/11 | {r['primary_windows_feasible']}/11 | {r['primary_windows_cbc_optimal_reported']} | {r['primary_windows_time_limit_feasible']} | {'Yes, with gap/status caveats' if r['complete_for_11_windows'] else 'No'} |")
    (d/'REPORT.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    print('\n'.join(doc[:7]));print('Reports written to',d.resolve())
    if len(complete)==6:
        try:
            import matplotlib.pyplot as plt
            group=sorted(complete,key=lambda x:x['lambda_multiplier'])
            x=[r['lambda_multiplier'] for r in group]
            plt.figure(figsize=(7,4));plt.plot(x,[r['mean_local_ramp_mw_total_variation'] for r in group],marker='o')
            plt.xlabel('Multiplier of baseline lambda (0.01)');plt.ylabel('Local ramp total variation (MW)')
            plt.title('Full MILP sensitivity (11 primary windows)');plt.tight_layout();plt.savefig(d/'figure_local_ramping_FULL_MILP.png',dpi=300);plt.close()
            plt.figure(figsize=(7,4));plt.plot(x,[r['mean_unserved_mwh'] for r in group],marker='o')
            plt.xlabel('Multiplier of baseline lambda (0.01)');plt.ylabel('Unserved energy (MWh)')
            plt.tight_layout();plt.savefig(d/'figure_unserved_FULL_MILP.png',dpi=300);plt.close()
            plt.figure(figsize=(7,4));plt.plot(x,[r['mean_aggregate_ramp_mw_total_variation'] for r in group],marker='o')
            plt.xlabel('Multiplier of baseline lambda (0.01)');plt.ylabel('Aggregate ramp total variation (MW)')
            plt.tight_layout();plt.savefig(d/'figure_aggregate_ramping_FULL_MILP.png',dpi=300);plt.close()
        except ImportError:
            print('matplotlib not installed; numeric reports completed, figures skipped.')

if __name__=='__main__':main()
