"""Loading and validating the SCADA record.

The raw record is governed by a Ministry of Electricity confidentiality arrangement
and is not redistributed. This module reads a local copy and refuses one that does
not match the dataset the reported results were produced from.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

COLUMNS = {
    "Governorate": "gov",
    "Timestamp": "ts",
    "Demand_Value": "demand_mw",
    "temperature_2m": "temp_c",
}


def load(path: str, strict: bool = True) -> pd.DataFrame:
    """Read the SCADA CSV and return it with the column names used throughout.

    With ``strict`` the file must have the shape of the cleaned record: 206,040
    rows, five governorates, 41,208 rows each and no duplicated
    (governorate, timestamp) pair. Set ``strict=False`` only to run the code on a
    different dataset, in which case the reported numbers will not be reproduced.
    """
    raw = pd.read_csv(path)
    missing = [c for c in list(COLUMNS) + ["Supply_Hours"] if c not in raw.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])

    counts = raw.Governorate.value_counts()
    problems = []
    if strict:
        if len(raw) != C.EXPECTED_ROWS:
            problems.append(f"expected {C.EXPECTED_ROWS:,} rows, got {len(raw):,}")
        if len(counts) != len(C.GOVERNORATES):
            problems.append(f"expected {len(C.GOVERNORATES)} governorates, got {len(counts)}")
        if len(counts) and counts.min() != C.EXPECTED_ROWS_PER_GOVERNORATE:
            problems.append(
                f"expected {C.EXPECTED_ROWS_PER_GOVERNORATE:,} rows per governorate, "
                f"minimum is {counts.min():,}"
            )
    dups = int(raw.duplicated(subset=["Governorate", "Timestamp"]).sum())
    if dups:
        problems.append(f"{dups} duplicate (Governorate, Timestamp) rows")
    if problems:
        raise ValueError(
            "SCADA file failed integrity checks:\n  - "
            + "\n  - ".join(problems)
            + f"\nRow counts per governorate: {counts.to_dict()}"
        )

    df = raw.rename(columns=COLUMNS)
    return df.sort_values(["gov", "ts"]).reset_index(drop=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-governorate demand and temperature summary, including P_max (Eq. 1)."""
    g = df.groupby("gov").agg(
        n=("demand_mw", "size"),
        mu=("demand_mw", "mean"),
        sigma=("demand_mw", "std"),
        t_min=("temp_c", "min"),
        t_max=("temp_c", "max"),
    )
    g["P_max"] = g["mu"] + C.K_SIGMA * g["sigma"]
    return g.round(2)


def describe(df: pd.DataFrame) -> str:
    return (
        f"rows               : {len(df):,}\n"
        f"governorates       : {sorted(df.gov.unique())}\n"
        f"period             : {df.ts.min()}  ->  {df.ts.max()}\n"
        f"temperature (degC) : {df.temp_c.min():.1f} .. {df.temp_c.max():.1f} "
        f"({df.temp_c.nunique():,} distinct values)\n"
        f"demand (MW)        : {df.demand_mw.min():.1f} .. {df.demand_mw.max():.1f}"
    )
