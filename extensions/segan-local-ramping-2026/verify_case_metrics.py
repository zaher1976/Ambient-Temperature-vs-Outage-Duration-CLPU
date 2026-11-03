"""Recalculate four metrics from authorised SEGAN case JSON solutions.

Usage: python verify_case_metrics.py /path/to/cases
No confidential case files are bundled or uploaded by this script.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def verify(path):
    obj = json.loads(Path(path).read_text())
    v = obj["decision_variables"]
    served = np.asarray(v["served"], dtype=float)
    shed = np.asarray(v["shed"], dtype=float)
    u = np.asarray(v["u"], dtype=float)
    restore = np.asarray(v["v"], dtype=float)
    n = served.shape[0]
    recomputed = {
        "local_ramp_mw_total_variation": float(np.abs(np.diff(served, axis=2)).sum() / n),
        "aggregate_ramp_mw_total_variation": float(np.abs(np.diff(served.sum(axis=1), axis=1)).sum() / n),
        "served_mwh": float(served.sum() / n),
        "unserved_mwh": float(shed.sum() / n),
    }
    metric_error = max(abs(x - obj["metrics"][k]) for k,x in recomputed.items())
    transition_error = float(np.max(np.abs(restore[:,1:] - np.maximum(u[:,1:] - u[:,:-1],0))))
    first_hour_error = float(np.max(np.abs(restore[:,0] - u[:,0])))
    return metric_error, transition_error, first_hour_error

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--tol",type=float,default=1e-4)
    args=p.parse_args()
    paths=sorted(args.directory.glob("W??_m*.json"))
    if not paths:
        raise SystemExit("No case JSON files")
    results=np.asarray([verify(f) for f in paths])
    print("Verified cases:",len(paths),"max residuals:",np.max(results,axis=0).tolist())
    if np.any(results>args.tol):
        raise SystemExit("FAILED")
    print("PASSED")
