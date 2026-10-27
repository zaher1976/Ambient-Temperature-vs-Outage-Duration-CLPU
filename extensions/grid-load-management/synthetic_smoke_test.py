"""Check first-hour reconnection on synthetic (not SCADA-derived) data."""
import numpy as np
import pulp
from forecast_milp_v4 import build_model_v4

def main():
    assert pulp.__version__ == '3.3.2', pulp.__version__
    forecast = np.full((2,3), 10.0)
    demand = np.full((1,2,3), 10.0)
    supply = np.full(3, 100.0)
    cap = np.full(2, 30.0)
    prob, var = build_model_v4(forecast,demand,supply,cap,
        clpu_kernel=np.array([0.30,0.15,0.05]),fairness_epsilon=None,
        switching_weight=0.1,ramp_weight=0.01,
        local_ramp_weight=0.01,initial_on=np.zeros(2,dtype=int))
    solver = pulp.PULP_CBC_CMD(msg=False,threads=1,gapRel=0.001,options=['randomSeed 42'])
    if not solver.available():
        raise RuntimeError('CBC solver is required for synthetic smoke test')
    prob.solve(solver)
    assert pulp.LpStatus[prob.status]=='Optimal', pulp.LpStatus[prob.status]
    for f in range(2):
        assert abs(pulp.value(var['u'][f,0])-1) < 1e-6, 'Feeder not connected in first hour'
        assert abs(pulp.value(var['v'][f,0])-1) < 1e-6, 't=0 restoration not represented'
        expected=10*(1+0.30)
        actual=pulp.value(var['served'][0,f,0])
        assert abs(actual-expected)<1e-5, (f,actual,expected)
    print('PASS: initial reconnection, CLPU contribution and CBC MILP')

if __name__=='__main__':
    main()
