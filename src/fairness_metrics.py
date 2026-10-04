import numpy as np

def jain_fairness_index(x):
    x = np.asarray(x, dtype=float)
    denom = len(x) * np.sum(x**2)
    if denom <= 0:
        return np.nan
    return (np.sum(x)**2) / denom

def disparity(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(x) - np.min(x))
