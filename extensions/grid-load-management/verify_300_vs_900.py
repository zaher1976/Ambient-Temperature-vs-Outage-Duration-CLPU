"""Grid Load Management CBC forensic check: original 72 and six targeted 900-second reruns.

This script never solves or edits the MILP. It requires only numpy,pandas.
Use --frozen path to the exact frozen model input; checksum is verified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_FROZEN_SHA = 'c2700bc8361d3d5b5ed7cfd4eed075f13c5dcf7e3bf6ae1d2812944ff2c616de'
EXPECTED_MODEL_SHA = 'def239830643f661ed7c231fff7f7f332021933620cb74225bd2cc2ac18b230a'
TARGETS = [(7,0.0),(7,0.25),(7,0.5),(7,1.0),(7,2.0),(8,0.5)]
MULTS=[0.0,0.1,0.25,0.5,1.0,2.0]
TOL=1e-3


def case_id(window,mult):
    return f'W{window:02d}_m{mult:g}'.replace('.', 'p')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def close(a,b,atol=1e-3):
    if a is None or b is None:
        return a is b
    return bool(np.isclose(float(a),float(b),atol=atol,rtol=1e-7))


def recompute(case,z):
    w=int(case['window_id'])
    D=np.asarray(z[f'demand_{w:02d}'],dtype=float)
    Q=np.asarray(z[f'supply_{w:02d}'],dtype=float)
    P=np.asarray(z[f'pmax_{w:02d}'],dtype=float)
    V=case['decision_variables']
    u=np.asarray(V['u'],dtype=float)
    v=np.asarray(V['v'],dtype=float)
    served=np.asarray(V['served'],dtype=float)
    shed=np.asarray(V['shed'],dtype=float)
    S,F,T=D.shape
    if not (u.shape==v.shape==(F,T) and served.shape==shed.shape==D.shape):
        raise ValueError('Decision arrays / frozen input shapes mismatch')
    if not all(np.isfinite(x).all() for x in [u,v,served,shed]):
        raise ValueError('Nonfinite decision variable')
    clpu=np.zeros_like(D)
    for f in range(F):
        for t in range(T):
            for k,alpha in enumerate([.3,.15,.05]):
                if k>t:break
                a = v[f,t-k] * np.prod(u[f,t-k+1:t+1])
                clpu[:,f,t]+=D[:,f,t]*alpha*a
    expected_v=np.zeros_like(v)
    expected_v[:,0]=np.maximum(0,u[:,0])
    expected_v[:,1:]=np.maximum(0,u[:,1:]-u[:,:-1])
    net=served.sum(axis=1)
    norm=(served/P[None,:,None]).sum(axis=2)
    den=F*np.square(norm).sum(axis=1)
    jfi=np.divide(np.square(norm.sum(axis=1)),den,
                  out=np.full(S,np.nan),where=den>0)
    fair=max(np.max(np.abs(norm[:,f]-norm[:,j]))
             for f in range(F) for j in range(f+1,F))
    metric={
        'local_ramp_mw_total_variation':float(np.abs(np.diff(served,axis=2)).sum()/S),
        'aggregate_ramp_mw_total_variation':float(np.abs(np.diff(net,axis=1)).sum()/S),
        'unserved_mwh':float(shed.sum()/S),
        'clpu_mwh':float(clpu.sum()/S),
        'served_mwh':float(served.sum()/S),
        'mean_jfi_posthoc':float(np.nanmean(jfi)),
        'max_balance_residual_mw':float(np.max(np.abs(shed-(D+clpu-served)))),
        'max_supply_excess_mw':float(np.maximum(net-Q[None,:],0).max()),
        'max_feeder_cap_excess_mw':float(np.maximum(served-P[None,:,None],0).max()),
        'max_requested_excess_mw':float(np.maximum(served-(D*u[None,:,:]+clpu),0).max()),
        'max_fairness_violation':float(max(0,fair-.1)),
        'max_restoration_indicator_residual':float(np.max(np.abs(v-expected_v))),
        'max_negative_served':float(np.maximum(-served,0).max()),
        'max_negative_shed':float(np.maximum(-shed,0).max()),
        'max_binary_integrality':float(max(np.max(abs(u-np.round(u))),np.max(abs(v-np.round(v))))),
    }
    return metric


def audit_case(folder,w,m,limit,z):
    name=case_id(w,m)
    p=folder/'cases'/f'{name}.json'
    row={'case':name,'window':w,'multiplier':m,'limit_s':limit}
    if not p.exists():
        return dict(row,status='MISSING',termination=None,gap_percent=None,elapsed_s=None,local_ramp=None,aggregate_ramp=None,unserved_mwh=None,clpu_mwh=None,jfi=None,checks='JSON missing')
    try:
        j=json.loads(p.read_text(encoding='utf-8'))
        met=recompute(j,z)
        errors=[]
        if int(j.get('window_id',-1))!=w:errors.append('window_id')
        if not close(j.get('lambda_multiplier'),m,atol=1e-9):errors.append('multiplier')
        if not close(j.get('lambda_local'),m*.01,atol=1e-9):errors.append('lambda_local')
        if j.get('validation')!='FEASIBLE_CHECKED':errors.append('validation')
        if j.get('cbc_termination') not in ['CBC_REPORTED_OPTIMAL','TIME_LIMIT']:
            errors.append('cbc_termination')
        provenance=j.get('provenance',{})
        if provenance.get('frozen_sha256')!=EXPECTED_FROZEN_SHA:errors.append('frozen_hash')
        if provenance.get('model_sha256')!=EXPECTED_MODEL_SHA:errors.append('model_hash')
        if provenance.get('pulp_version')!='3.3.2':errors.append('pulp_version')
        solver=j.get('solver_settings',{})
        if solver.get('limit_seconds')!=limit:errors.append('time_limit')
        if not close(solver.get('relative_gap'),.001,atol=1e-12):errors.append('gap_setting')
        if solver.get('threads')!=1 or solver.get('random_seed')!=42:
            errors.append('solver_reproducibility')
        for key,val in met.items():
            if key.startswith('max_'):
                if val>TOL:errors.append(f'constraint:{key}={val:.5g}')
            elif not close(j.get('metrics',{}).get(key),val,atol=TOL):
                errors.append(f'metric:{key}')
        for key,val in j.get('diagnostics',{}).items():
            if key.startswith('max_') and val is not None and float(val)>TOL:
                errors.append(f'model_constraint:{key}')
        incumbent=j.get('cbc_reported_objective')
        lb=j.get('cbc_lower_bound')
        gap=j.get('cbc_reported_relative_gap')
        if all(v is not None for v in [incumbent,lb,gap]):
            calc_gap=abs(float(incumbent)-float(lb))/max(abs(float(incumbent)),1e-12)
            if not close(calc_gap,gap,atol=1e-6):errors.append('cbc_gap_mismatch')
        log=folder/'logs'/f'{name}.cbc.log'
        if not log.exists():
            errors.append('cbc_log_missing')
        else:
            t=log.read_text(encoding='utf-8',errors='replace').lower()
            if j.get('cbc_termination')=='CBC_REPORTED_OPTIMAL' and 'result - optimal solution found' not in t:
                errors.append('cbc_log_no_optimal_marker')
            if j.get('cbc_termination')=='TIME_LIMIT' and not any(x in t for x in ['stopped on time','stopped on time limit','exiting on maximum time']):
                errors.append('cbc_log_no_time_marker')
        if errors:
            status='FAIL_AUDIT'
        elif gap is not None and gap>.001+1e-8:
            status='FEASIBLE_GAP_ABOVE_TARGET'
        elif j.get('cbc_termination')=='TIME_LIMIT' and gap is None:
            status='FEASIBLE_GAP_UNKNOWN'
        else:
            status='VERIFIED_REPORTED_STATUS'
        return dict(row,status=status,
                    termination=j.get('cbc_termination'),
                    gap_percent=None if gap is None else 100*float(gap),
                    elapsed_s=j.get('elapsed_s'),
                    local_ramp=met['local_ramp_mw_total_variation'],
                    aggregate_ramp=met['aggregate_ramp_mw_total_variation'],
                    unserved_mwh=met['unserved_mwh'],
                    clpu_mwh=met['clpu_mwh'],
                    jfi=met['mean_jfi_posthoc'],
                    checks='; '.join(errors) if errors else 'All saved-data and log checks passed')
    except Exception as exc:
        return dict(row,status='FAIL_AUDIT',termination=None,gap_percent=None,elapsed_s=None,local_ramp=None,aggregate_ramp=None,unserved_mwh=None,clpu_mwh=None,jfi=None,checks=f'{type(exc).__name__}: {exc}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',type=Path,default=Path('/content/drive/MyDrive/GRID_LOAD_MILP_FULL_MILP_RESULTS'))
    ap.add_argument('--rerun',type=Path,default=Path('/content/drive/MyDrive/GRID_LOAD_MILP_RERUN_900_RESULTS'))
    ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('/content/drive/MyDrive/GRID_LOAD_MILP_VERIFICATION'))
    a=ap.parse_args()
    for p in [a.original,a.frozen]:
        if not p.exists():raise FileNotFoundError(f'Missing source: {p}')
    hash_=sha256(a.frozen)
    if hash_!=EXPECTED_FROZEN_SHA:
        raise RuntimeError(f'FROZEN SHA256 MISMATCH: {hash_}')
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.frozen,allow_pickle=False) as z:
        original=[audit_case(a.original,w,m,300,z) for w in range(12) for m in MULTS]
        new=[audit_case(a.rerun,w,m,900,z) for w,m in TARGETS]
    odf=pd.DataFrame(original)
    ndf=pd.DataFrame(new)
    odf.to_csv(a.output/'CBC_72_original_audit.csv',index=False)
    ndf.to_csv(a.output/'CBC_6_rerun_900_audit.csv',index=False)
    print('ORIGINAL (72 cases):')
    print(odf['status'].value_counts().to_string())
    print('Original CBC reported optimal:',sum(odf.termination=='CBC_REPORTED_OPTIMAL'))
    print('Original time limit cases:',sum(odf.termination=='TIME_LIMIT'))
    print('\n900-SECOND RERUN (six cases):')
    print(ndf[['case','status','termination','gap_percent','elapsed_s','checks']].to_string(index=False))
    old6=odf[odf['case'].isin(ndf['case'])].set_index('case')
    comp=ndf.set_index('case').join(old6[['gap_percent','local_ramp','aggregate_ramp','unserved_mwh']],rsuffix='_300',how='left')
    comp['gap_change_pp']=comp['gap_percent']-comp['gap_percent_300']
    comp['local_ramp_change']=comp['local_ramp']-comp['local_ramp_300']
    comp['unserved_change_mwh']=comp['unserved_mwh']-comp['unserved_mwh_300']
    comp.to_csv(a.output/'CBC_300_vs_900_comparison.csv')
    if (ndf.status=='MISSING').any():
        print('\n900-second rerun incomplete/missing: DO NOT claim six cases checked at 900s.')
    elif (ndf.status=='FAIL_AUDIT').any():
        print('\nSome rerun cases failed verification: investigate logs and JSON.')
    else:
        print('\nAll six rerun JSONs present and feasible; inspect gaps before claiming optimality.')
    print('Audit written to',a.output)
    return odf,ndf,comp

if __name__=='__main__':main()
