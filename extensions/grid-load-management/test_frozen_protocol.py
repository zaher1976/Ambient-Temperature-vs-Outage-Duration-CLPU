"""Fast self-tests: original files unchanged and metric arithmetic correct.
Run before CBC: python test_frozen_protocol.py
"""
from __future__ import annotations
import json
import numpy as np
from sensitivity_metrics import postprocess
from run_full_cbc import HERE,FROZEN,MANIFEST,EXPERIMENT,frozen_integrity,read_log

def main():
    manifest=json.loads(MANIFEST.read_text())
    experiment=json.loads(EXPERIMENT.read_text())
    assert frozen_integrity(manifest,experiment)
    with np.load(FROZEN,allow_pickle=False) as z:
        assert z['demand_00'].shape==(4,5,24)
        assert all(z[f'demand_{w:02d}'].shape==(5,5,24) for w in range(1,12))
        assert all(z[f'supply_{w:02d}'].shape==(24,) for w in range(12))
    d=np.full((1,2,3),10.0)
    u=np.array([[1,1,0],[0,0,0]],float)
    v=np.array([[1,0,0],[0,0,0]],float)
    sr=np.array([[[5,6,0],[0,0,0]]],float)
    extra=np.array([[[3,1.5,0],[0,0,0]]],float)
    sh=d+extra-sr
    m=postprocess(d,np.full(3,30),np.full(2,20),np.array([.3,.15,.05]),
                  np.array([0,0]),u,v,sr,sh,fairness_eps=1.0)
    assert abs(m['local_ramp_mw_total_variation']-7)<1e-10
    assert abs(m['aggregate_ramp_mw_total_variation']-7)<1e-10
    assert abs(m['unserved_mwh']-53.5)<1e-10
    assert abs(m['clpu_mwh']-4.5)<1e-10
    assert m['max_balance_residual_mw']<1e-10
    assert m['max_fairness_violation']<1e-10
    print('PASS: frozen SHA256; 12 unmodified windows; scenario counts; CLPU/ramp/energy identities.')
if __name__=='__main__':main()
