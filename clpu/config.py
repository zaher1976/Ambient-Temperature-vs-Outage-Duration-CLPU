"""Every parameter of the study, in one place.

The values are those used for the reported results. Equation numbers refer to the
manuscript. Nothing here reads a notebook or a cloud drive: the dataset path comes
from the command line or from the CLPU_DATA environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# ─────────────────────────────────────────────── study design
GOVERNORATES = ["Anbar", "Babil", "Kirkuk", "Najaf", "Wasit"]
CONFIGS = ["S1", "S2", "S3", "S4", "S5"]

#: Percentiles of the hourly temperature distribution that define the six thermal
#: regimes. The regime temperatures themselves are computed from the record
#: (see :func:`clpu.pipeline.prepare`), never hard-coded.
REGIME_PERCENTILES = [10, 30, 50, 70, 90, 99]

# ─────────────────────────────────────────────── MILP parameters (Eqs. 1-18)
T_HORIZON = 24          # planning horizon in hours, T
N_SCENARIOS = 20        # |Omega|
REPORT_N = 20           # scenarios used for MaxRamp, disparity and JFI
K_SIGMA = 1.0           # P_max(f) = mu + k*sigma                        (Eq. 1)
C_CURT = 1.2            # curtailment cost coefficient                   (Eq. 2)
C_RAMP = 0.4            # ramping cost coefficient                       (Eq. 2)
C_DISC = 0.2            # discomfort price                          (Eqs. 2, 18)
EPS_FAIR = 0.05         # pairwise fairness tolerance               (Eqs. 16-17)

#: Upper edges of the ambient-temperature bins, in degrees Celsius.
TEMP_BINS = [-np.inf, 30.0, 35.0, 40.0, 45.0, np.inf]
#: lambda_b per bin: a surge of 1.30x to 1.80x of forecast demand.
LAMBDA_BY_BIN = [0.30, 0.45, 0.60, 0.70, 0.80]

#: Coefficient sets for the kernel sensitivity analysis (Section 5.7).
LAMBDA_RANGES = {
    "conservative": [0.10, 0.20, 0.30, 0.40, 0.50],   # 1.10x - 1.50x
    "baseline": LAMBDA_BY_BIN,                         # 1.30x - 1.80x
    "aggressive": [0.50, 0.60, 0.75, 0.90, 1.00],      # 1.50x - 2.00x
}
SCENARIO_GRID = [10, 20, 30, 50]     # Section 5.8
EPS_GRID = [0.03, 0.05, 0.10]        # Section 5.8

# ─────────────────────────────────────────────── solver
TIME_LIMIT = 300        # seconds per instance
GAP_REL = 0.03          # prescribed relative MIP gap

# ─────────────────────────────────────────────── forecasting
LOOKBACK = 24           # hours of history per training sequence
TEST_DAYS = 30          # chronologically held-out test period
SEED = 42

# ─────────────────────────────────────────────── dataset integrity
#: The cleaned record the reported results were produced from. load() refuses a
#: file that does not match, so a silently different dataset cannot be mistaken
#: for this one.
EXPECTED_ROWS = 206_040
EXPECTED_ROWS_PER_GOVERNORATE = 41_208

#: Window start and national mean temperature of each regime in the reported run.
#: verify_windows() checks a rebuilt pipeline against these.
EXPECTED_WINDOWS = {
    10.5: ("2020-01-08 11:00", 10.5),
    17.5: ("2020-11-16 01:00", 17.5),
    25.6: ("2024-04-08 09:00", 25.6),
    32.8: ("2024-09-16 16:00", 32.8),
    40.5: ("2022-06-19 17:00", 40.5),
    46.4: ("2020-07-29 12:00", 43.0),
}


@dataclass
class Paths:
    """Where the dataset is read from and where results are written."""

    data: str = ""
    out: str = "results"

    def __post_init__(self) -> None:
        if not self.data:
            self.data = os.environ.get("CLPU_DATA", "")
        if not self.data:
            raise ValueError(
                "No dataset path. Pass --data /path/to/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv "
                "or set the CLPU_DATA environment variable. The raw SCADA record is not "
                "redistributed with this package; see data/README_DATA.md."
            )
        if not os.path.exists(self.data):
            raise FileNotFoundError(f"dataset not found: {self.data}")
        os.makedirs(self.out, exist_ok=True)

    def __truediv__(self, name: str) -> str:
        return os.path.join(self.out, name)


@dataclass
class Settings:
    """Run-level choices that the command line can override."""

    n_scenarios: int = N_SCENARIOS
    time_limit: int = TIME_LIMIT
    gap_rel: float = GAP_REL
    eps_fair: float = EPS_FAIR
    k_sigma: float = K_SIGMA
    lambdas: list = field(default_factory=lambda: list(LAMBDA_BY_BIN))
    #: "double" reproduces the reported results, in which C_disc multiplies both
    #: the accumulation of Phi and its contribution to the objective (Section 4.7).
    #: "single" applies it once (Section 5.11).
    disc_mode: str = "double"

    def __post_init__(self) -> None:
        if self.disc_mode not in ("double", "single"):
            raise ValueError("disc_mode must be 'double' or 'single'")
