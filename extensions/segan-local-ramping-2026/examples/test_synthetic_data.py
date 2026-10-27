#!/usr/bin/env python3
"""Offline, standard-library-only validation for the two PUBLIC synthetic cases."""
import importlib.util
import json
import math
from pathlib import Path

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('syn_gen',P/'generate_synthetic_examples.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
    for case,expected in [('small',(3,8,2)),('day24',(5,24,3))]:
        source=(P/f'synthetic_{case}.json').read_text(encoding='utf8')
        dataset=json.loads(source)
        expected_serialized=json.dumps(module.generate(case),indent=2,ensure_ascii=True)+'\n'
        assert source==expected_serialized,case+' sample generation is not deterministic'
        assert dataset['metadata']['synthetic'] is True
        assert dataset['metadata']['source']=='DETERMINISTIC_FORMULAS_ONLY_NO_SCADA'
        F,T,S=expected
        assert len(dataset['forecast_mw'])==F
        assert all(len(x)==T for x in dataset['forecast_mw'])
        assert len(dataset['scenario_demand_mw'])==S
        assert all(len(x)==F for x in dataset['scenario_demand_mw'])
        assert all(len(x)==T for z in dataset['scenario_demand_mw'] for x in z)
        assert len(dataset['available_supply_mw'])==T
        assert len(dataset['pmax_mw'])==F
        assert dataset['initial_on']==[0]*F
        assert all(x>0 for x in dataset['available_supply_mw'])
        assert all(x>0 for x in dataset['pmax_mw'])
        assert all(math.isfinite(x) for z in dataset['scenario_demand_mw'] for y in z for x in y)
        assert abs(sum(dataset['clpu_kernel'])-0.5)<1e-9
    print('PASS: synthetic input formats, dimensions, determinism, nonnegativity')

if __name__=='__main__':main()
