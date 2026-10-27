#!/usr/bin/env python3
"""CBC full-MILP sweep of the UNMODIFIED Grid Load Management v4 formulation.

All binary variables, including the restoration schedule, are reoptimised.
Every case runs against one of the 12 SHA256-pinned frozen windows.
This program performs no forecasting, training, or scenario generation.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, re, sys, time
from pathlib import Path
import numpy as np
from sensitivity_metrics import arrays_from_model, max_model_residual, postprocess

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'multiwindow_inputs_frozen.npz'
MANIFEST = HERE / 'multiwindow_inputs_manifest.json'
EXPERIMENT = HERE / 'experiment_manifest.json'
VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
PRIMARY_WEIGHT = 0.01
TOL = 1e-3


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def save_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def frozen_integrity(manifest, experiment):
    got = sha256(FROZEN)
    assert got == manifest['files_sha256']['frozen_npz'],f'Frozen input hash mismatch: {got}'
    for fn, key in [('multiwindow_scenario_protocol.json', 'scenario_protocol')]:
        assert sha256(HERE/fn) == manifest['files_sha256'][key], f'{fn}: hash mismatch'
    assert len(manifest['windows'])==12
    ref = experiment['settings']
    assert ref['v4_local_ramp_weight'] == PRIMARY_WEIGHT
    assert ref['fairness_epsilon']==0.1 and ref['clpu_kernel']==[.3,.15,.05]
    assert ref['switching_weight']==.1 and ref['aggregate_ramp_weight']==.01
    return got


def read_log(logpath):
    content = logpath.read_text(errors='replace') if logpath.exists() else ''
    lower = content.lower()
    if 'result - optimal solution found' in lower:
        classification = 'CBC_REPORTED_OPTIMAL'
    elif ('result - stopped on time' in lower or 'stopped on time limit' in lower
          or 'exiting on maximum time' in lower):
        classification = 'TIME_LIMIT'
    elif 'no feasible solution found' in lower or 'infeasible' in lower:
        classification = 'NO_FEASIBLE_SOLUTION_REPORTED'
    else:
        classification = 'REVIEW_SOLVER_LOG'
    def number(pattern):
        hit=re.search(pattern,content,re.I|re.M)
        return float(hit.group(1)) if hit else None
    # CBC reports these near its closing summary. Solver gap != guaranteed 0.
    bounds=re.findall(r'^Lower bound:\s*([-+\d.eE]+)',content,re.M)
    objectives=re.findall(r'^Objective value:\s*([-+\d.eE]+)',content,re.M)
    bound=float(bounds[-1]) if bounds else None
    incumbent=float(objectives[-1]) if objectives else None
    gap=abs(incumbent-bound)/max(abs(incumbent),1e-12) if incumbent is not None and bound is not None else None
    return classification, gap, incumbent, bound


def make_cbc_solver(pulp, **kwargs):
    """Resolve CBC for PuLP 3.3.2, preferring bundled binary, then OS CBC.

    Selection modifies ONLY the solver bridge, not MILP formulation or data.
    """
    for api in ('PULP_CBC_CMD', 'COIN_CMD'):
        cls = getattr(pulp, api, None)
        if cls is None:
            continue
        try:
            solver = cls(**kwargs)
            if solver.available():
                return solver, api
        except Exception:
            continue
    raise RuntimeError('CBC unavailable. Install coinor-cbc (apt-get install -y coinor-cbc).')


def assert_pulp_legacy_api(pulp):
    if pulp.__version__ != '3.3.2':
        raise RuntimeError(
            f'This frozen Grid Load Management runner requires PuLP 3.3.2; found {pulp.__version__}. '
            'Run %pip install "pulp==3.3.2" and restart the Colab session.'
        )
    if not hasattr(pulp, 'PULP_CBC_CMD') or not hasattr(pulp, 'LpVariable'):
        raise RuntimeError('Legacy PuLP modelling API unavailable; restart the session.')


def solve_case(z, wm, multiplier, args, expected_sha, model_sha):
    import pulp
    assert_pulp_legacy_api(pulp)
    from forecast_milp_v4 import build_model_v4
    wid=int(wm['window_id'])
    d=np.array(z[f'demand_{wid:02d}'], dtype=float)
    supply=np.array(z[f'supply_{wid:02d}'], dtype=float)
    cap=np.array(z[f'pmax_{wid:02d}'], dtype=float)
    S,F,T=d.shape
    assert (S,F,T)==tuple(wm['demand_shape'])
    assert S==wm['scenario_count'] and F==5 and T==24
    weight=round(PRIMARY_WEIGHT*multiplier,8)
    name=f'W{wid:02d}_m{multiplier:g}'.replace('.','p')
    path=args.output_dir/'cases'/f'{name}.json'
    logpath=args.output_dir/'logs'/f'{name}.cbc.log'
    if path.exists() and not args.force:
        old=json.loads(path.read_text(encoding='utf-8'))
        if (old.get('provenance',{}).get('frozen_sha256')==expected_sha
            and old.get('provenance',{}).get('model_sha256')==model_sha
            and old.get('solver_settings',{}).get('limit_seconds')==args.time_limit
            and old.get('solver_settings',{}).get('relative_gap')==args.gap):
            print(f'{name}: SKIPPED (previous result preserved)',flush=True)
            return
        raise RuntimeError(f'Existing result {path} has incompatible provenance. Use --force intentionally.')
    prob, vars = build_model_v4(d[0],d,supply,cap,
        clpu_kernel=np.array([.30,.15,.05]), fairness_epsilon=.1,
        curtailment_weight=1.0, switching_weight=.1,ramp_weight=.01,
        local_ramp_weight=weight,initial_on=np.zeros(F,dtype=int))
    solver, solver_api = make_cbc_solver(pulp, msg=True, timeLimit=args.time_limit,
         gapRel=args.gap, threads=1, options=['randomSeed 42'],
         warmStart=False, logPath=str(logpath))
    start=time.monotonic()
    error=None
    try:prob.solve(solver)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    duration=time.monotonic()-start
    classification,gap,cbco,lower=read_log(logpath)
    raw_status=pulp.LpStatus.get(prob.status,str(prob.status))
    metrics={};diag={};variables={};validation='UNVERIFIED'
    if error is None:
        try:
            u,v,served,shed=arrays_from_model(vars,d.shape)
            metrics=postprocess(d,supply,cap,np.array([.30,.15,.05]),np.zeros(F),u,v,served,shed)
            diag=max_model_residual(prob)
            failures={key:val for key,val in {**metrics, **diag}.items()
                 if key.startswith(('max_')) and val is not None and val > TOL}
            validation='FEASIBLE_CHECKED' if not failures else 'INVALID_CONSTRAINT_RESIDUAL'
            variables=dict(u=u.tolist(),v=v.tolist(),served=served.tolist(),shed=shed.tolist())
            # CBC *reported* optimality is distinct from mathematical proof of global opt.
        except Exception as exc:
            error=f'Failed to extract/validate incumbent: {exc}'
    result=dict(window_id=wid,fold=wm['fold'],origin=wm['origin'],
        analysis_group=wm['analysis_group'],scenario_count=S,
        lambda_multiplier=multiplier,lambda_local=weight,
        pulp_status=raw_status,cbc_termination=classification,
        cbc_reported_relative_gap=gap,cbc_reported_objective=cbco,
        cbc_lower_bound=lower,reported_objective=float(pulp.value(prob.objective))
            if prob.objective is not None and pulp.value(prob.objective) is not None else None,
        elapsed_s=round(duration,3),validation=validation,
        diagnostics=diag,metrics=metrics,error=error,
        solver_settings={'limit_seconds':args.time_limit,'relative_gap':args.gap,'threads':1,
                         'random_seed':42,'warm_start':False},
        provenance={'frozen_sha256':expected_sha,'model_sha256':model_sha,
                    'input_model':'forecast_milp_v4.build_model_v4',
                    'training':'UNCHANGED','window_selection':'UNCHANGED',
                    'scenario_construction':'UNCHANGED',
                    'solver':'CBC','solver_api':solver_api,'pulp_version':pulp.__version__, 'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat()},
        decision_variables=variables,cbc_log_relative_path=str(logpath.relative_to(args.output_dir)))
    save_json_atomic(path,result)
    print(f"{name}: {raw_status} / {classification} / {validation} / "
          f"gap={gap if gap is not None else 'unreported'}",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--windows',default='all',help='all or comma-separated window IDs e.g. 0,1,6')
    ap.add_argument('--multipliers',default='all',help='all or comma separated values from 0,0.1,0.25,0.5,1,2')
    ap.add_argument('--time-limit',type=int,default=300)
    ap.add_argument('--gap',type=float,default=.001)
    ap.add_argument('--output-dir',type=Path,default=HERE/'CBC_FULL_MILP_RESULTS')
    ap.add_argument('--force',action='store_true',help='Overwrite previous cases only when deliberate')
    ap.add_argument('--dry-run',action='store_true',help='Check frozen files and print plan; no solver required')
    a=ap.parse_args()
    if a.windows=='all': windows=list(range(12))
    else: windows=[int(x) for x in a.windows.split(',')]
    if a.multipliers=='all': mults=VALUES[:]
    else: mults=[float(x) for x in a.multipliers.split(',')]
    assert len(set(windows))==len(windows) and all(0<=n<12 for n in windows)
    assert len(set(mults))==len(mults) and all(any(abs(v-m)<1e-9 for m in VALUES) for v in mults)
    assert a.time_limit>0 and 0<a.gap<1
    manifest=json.loads(MANIFEST.read_text())
    experiment=json.loads(EXPERIMENT.read_text())
    frozen_hash=frozen_integrity(manifest,experiment)
    model_hash=sha256(HERE/'forecast_milp_v4.py')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for sub in ['cases','logs']:(a.output_dir/sub).mkdir(exist_ok=True)
    with np.load(FROZEN,allow_pickle=False) as z:
        assert len(z.files)==48
        for wm in manifest['windows']:
            w=int(wm['window_id'])
            assert z[f'demand_{w:02d}'].shape==tuple(wm['demand_shape'])
            assert z[f'supply_{w:02d}'].shape==(24,)
            assert z[f'pmax_{w:02d}'].shape==(5,)
        plan={'cases':len(windows)*len(mults),'windows':windows,'multipliers':mults,
              'frozen_sha256':frozen_hash,'model_sha256':model_hash}
        print('Validated frozen dataset:',json.dumps(plan,ensure_ascii=False),flush=True)
        if a.dry_run: return
        import pulp
        assert_pulp_legacy_api(pulp)
        make_cbc_solver(pulp, msg=False)
        for w in windows:
            for m in mults:
                solve_case(z,manifest['windows'][w],m,a,frozen_hash,model_hash)
    print('Case-level results stored:',a.output_dir.resolve(),flush=True)

if __name__=='__main__':main()
