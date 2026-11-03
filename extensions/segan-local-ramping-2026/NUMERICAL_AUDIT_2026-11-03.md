# Numerical audit: 72-case SEGAN local-ramping CBC sensitivity (2026-11-03)

**Status: discrepancy detected; not submission clearance.** This is a documented audit of stored results; it is **not** a claim of fully independent recomputation of every model constraint.

## Provenance

- Primary evidence: [SEGAN_CBC_FULL_MILP_RESULTS Google Drive folder](https://drive.google.com/drive/folders/17oJknVKt6PWx0lMjRALBh5CS6GMChivk), with `case_metrics.csv`, `primary_11_windows_summary.csv`, `REPORT.md`, 72 individual `cases/Wxx_m*.json` solutions and CBC logs.
- Paper comparison: author's 2026-10-28 original DOCX (sensitivity table displayed as Table 8 in that DOCX; called Table 4 in some previous reviews). Values below are copied from that table, **not** from a different manuscript.
- Related public software record: [Zenodo 10.5281/zenodo.23685364](https://doi.org/10.5281/zenodo.23685364).
- 12 windows W00--W11 × six multipliers 0, 0.1, 0.25, 0.5, 1, 2 = 72 original solutions. Only W01--W11 (11 windows) enter paper means. W00 is supplementary, four-scenario.

## Observed validity/status from stored summary

- 72/72 cases present in the `case_metrics.csv` file; 72/72 marked `usable_feasible=True`; 66/72 marked `cbc_optimal_reported=True` (this is a reported solver classification, not proof of zero-gap global optimality).
- Minimum reported posthoc Jain index: 0.999968738; maximum recorded constraint violation: 2.150000e-4 (units depend on constraint; do not relabel this as MW).
- The source `REPORT.md` records at least one time-limited feasible case for multipliers 0, 0.25, 0.5, 1, 2; thus the manuscript column 'CBC reported optimal 11/11' is inconsistent for these multipliers.
- The six 900-s reruns are separate experiments and MUST NOT replace the 300-s primary results silently.

## Comparison with paper sensitivity table

All differences are shown to two decimal places. A difference of zero after rounding is **not** proof of complete agreement beyond two decimals.

| Multiplier | Metric | Manuscript | Original CSV | Delta (CSV - manuscript) |
|---:|---|---:|---:|---:|
| 0 | Local TV (MW) | 32221.85 | 32274.43 | +52.58 |
| 0 | Aggregate TV (MW) | 1449.64 | 1449.21 | -0.43 |
| 0 | Unserved energy (MWh) | 32605.97 | 32732.49 | +126.52 |
| 0 | CLPU energy (MWh) | 3284.41 | 3411.90 | +127.49 |
| 0.1 | Local TV (MW) | 6323.91 | 6323.91 | +0.00 |
| 0.1 | Aggregate TV (MW) | 1455.80 | 1455.80 | +0.00 |
| 0.1 | Unserved energy (MWh) | 32606.16 | 32606.16 | -0.00 |
| 0.1 | CLPU energy (MWh) | 3284.60 | 3284.60 | -0.00 |
| 0.25 | Local TV (MW) | 5909.73 | 5775.21 | -134.52 |
| 0.25 | Aggregate TV (MW) | 1413.75 | 1411.80 | -1.95 |
| 0.25 | Unserved energy (MWh) | 32607.45 | 32672.45 | +65.00 |
| 0.25 | CLPU energy (MWh) | 3285.89 | 3352.61 | +66.72 |
| 0.5 | Local TV (MW) | 5668.07 | 5870.76 | +202.69 |
| 0.5 | Aggregate TV (MW) | 1447.40 | 1446.60 | -0.80 |
| 0.5 | Unserved energy (MWh) | 32608.10 | 32679.17 | +71.07 |
| 0.5 | CLPU energy (MWh) | 3287.86 | 3359.33 | +71.47 |
| 1 | Local TV (MW) | 4669.91 | 4815.88 | +145.97 |
| 1 | Aggregate TV (MW) | 1722.12 | 1723.27 | +1.15 |
| 1 | Unserved energy (MWh) | 32612.86 | 32612.22 | -0.64 |
| 1 | CLPU energy (MWh) | 3292.63 | 3290.66 | -1.97 |
| 2 | Local TV (MW) | 3309.37 | 3309.37 | +0.00 |
| 2 | Aggregate TV (MW) | 2059.20 | 2059.20 | -0.00 |
| 2 | Unserved energy (MWh) | 32629.91 | 32629.91 | +0.00 |
| 2 | CLPU energy (MWh) | 3311.04 | 3311.04 | -0.00 |

**16 of 24 two-decimal metric cells differ by more than 0.01** between the current paper table and the original primary-window summary.

## Interpretation and remaining validation

1. Reconcile the manuscript sensitivity table to the frozen 300-second `case_metrics.csv` and `primary_11_windows_summary.csv` or document a formally identified newer result set.
2. Independently recompute served-power local and aggregate total variation from all 72 JSON variable arrays; check CLPU reconnection-hour activation, demand balance, fairness bounds, and initial conditions against frozen inputs and model equations. The CSV's self-reported flags are *not* independent proof.
3. Distinguish `CBC_REPORTED_OPTIMAL`, time-limit-feasible incumbents, target MIP gap, and 900-second reruns.
4. No restricted SCADA or hourly optimization inputs are included in this audit document. Do not upload raw SCADA or case JSON files publicly without explicit data-use authorization.
5. Do not claim that the GitHub extension contains the code archived on Zenodo unless source files are actually committed and reviewed.

The audit is a **corrective documentation update**, not a new peer-reviewed release or a Zenodo version update.
