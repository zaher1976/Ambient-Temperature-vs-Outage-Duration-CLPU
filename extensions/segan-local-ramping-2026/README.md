# SEGAN local-ramping CBC sensitivity extension (code-only deposit)

**Study**: *Fairness-Constrained, CLPU-Aware, Predictive–Prescriptive Load Management in Smart Grids under Scarcity Conditions* (2026).  
**Author**: Zaher Fadhil Raham.  
**Status**: code and independently checked aggregate results; the original ministry SCADA record and derived hourly frozen optimization arrays are **not publicly redistributed**.

## Scope and relation to the earlier CLPU repository

This is a **separate SEGAN-style local-ramping extension**, not the same experiment as *Ambient Temperature versus Outage Duration as Surge Drivers in a Chronically Constrained Grid*. It must not replace that study's documented S1–S5 configurations, DOI, results or model equations. A new Zenodo deposit should receive its own version-specific DOI.

The original 12-window sensitivity run solves 72 full MILP cases (11 five-scenario primary windows and one four-scenario supplement) for local penalty multipliers `[0, 0.1, 0.25, 0.5, 1, 2]` against base penalty 0.01. CBC first ran for at most 300 s per case; six time-limit feasible cases were re-solved for at most 900 s. The six resulting CBC logs all state `Result - Optimal solution found` and the decision arrays pass the archived feasibility checks. **Do not interpret a missing reported numerical gap as gap=0.**

## Contents

- `forecast_milp_v3.py` / `forecast_milp_v4.py`: archived MILP formulations. The v4 model handles reconnection at the **first Python index t=0**, consistent with manuscript hour t=1 and `initial_on=0`.
- `run_full_cbc.py`: complete binary MILP sensitivity sweep; no BiLSTM retraining.
- `sensitivity_metrics.py`: calculates variations and energy accounting from `served` and `shed` rather than auxiliary objective variables.
- `make_report.py`: summary and completeness checks.
- `verify_300_vs_900.py`: verifies provenance and recovers numeric metrics from case decision arrays **when the protected frozen input package is available**.
- `test_frozen_protocol.py`: frozen-input and arithmetic smoke checks, requires protected frozen arrays.
- `synthetic_smoke_test.py`: first-hour restoration test **without any private SCADA data**.
- `corrected_11_window_summary.csv` and `six_case_900_summary.csv`: aggregate audit outputs; not the confidential 24-hour operating traces.
- The frozen input, window manifest, scenario protocol, and experiment manifest are deliberately **not public**; these must be supplied together by an authorized data custodian for full replay.

## Exact environment

Use Python, NumPy, and **PuLP 3.3.2** with COIN-OR CBC (900-second logs: **CBC 2.10.3**, according to the binary's own banner; confirm your installation). Solver settings: one thread, `randomSeed 42`, relative gap target 0.001; 300 s primary sweep and six 900 s reruns. CBC `Optimal solution found` is the recorded solver termination, not a general statement of zero floating-point error.

```bash
python -m pip install -r requirements.txt
python synthetic_smoke_test.py
```

To reproduce the 72 optimizations **only with authorized access** to the unchanged `multiwindow_inputs_frozen.npz`, `multiwindow_inputs_manifest.json`, `multiwindow_scenario_protocol.json` and `experiment_manifest.json`, put all four next to the code. Verify SHA256:

```text
multiwindow_inputs_frozen.npz  c2700bc8361d3d5b5ed7cfd4eed075f13c5dcf7e3bf6ae1d2812944ff2c616de
forecast_milp_v4.py          def239830643f661ed7c231fff7f7f332021933620cb74225bd2cc2ac18b230a
```

Then run `python test_frozen_protocol.py`, `python run_full_cbc.py --dry-run`, and a new 72-case run with `python run_full_cbc.py --time-limit 300 --gap 0.001 --output-dir ./results_original` (the full runtime can be substantial). Six cases needing longer time can be rerun in a **separate empty directory**:

```bash
python run_full_cbc.py --windows 7 --multipliers 0,0.25,0.5,1,2 --time-limit 900 --gap 0.001 --output-dir results_900
python run_full_cbc.py --windows 8 --multipliers 0.5 --time-limit 900 --gap 0.001 --output-dir results_900
```

The quick metrics can be recomputed with:

```bash
python verify_300_vs_900.py --frozen multiwindow_inputs_frozen.npz --original results_original --rerun results_900 --output ./verified_results
```

*Important*: the public code-only archive is insufficient to independently reconstruct all operational schedules because frozen inputs are withheld subject to the originating data-use restrictions. Request institutional permission before distributing them in GitHub or Zenodo. The outputs are **simulated optimization** results, not measured grid stability or measured generation.

## Manuscript statistics and caveats

The *paired v3–v4 300-s comparison* gives a mean of per-window percentage reductions in local power TV of **78.16%**, and, including W06, mean v4-minus-v3 aggregate TV of **+273.63 MW** across eleven primary windows. That is a **different estimand** from the local TV reduction of **85.51%** between the mean λ=0 and λ=0.01 solutions in the mixed 300/900-s complete sensitivity audit. At λ=0 the selected local-TV value can differ among near-equivalent curtailment-optimal schedules.

## Archiving and citation

Do not reuse the earlier repository's concept DOI `10.5281/zenodo.23205868` as the DOI of this distinct study. Deposit this package as a **new software record** on Zenodo and add the allocated version-specific DOI to the manuscript and `CITATION.cff` after publication. For step-by-step instructions see `ZENODO_UPLOAD.md`.

## Zenodo version record (v1.0.0)

The depositor supplied the Zenodo version-record URL [zenodo.org/records/23683578](https://zenodo.org/records/23683578) with DOI [10.5281/zenodo.23683578](https://doi.org/10.5281/zenodo.23683578). This identifier is specific to the SEGAN code-only deposit; the earlier CLPU study has a different DOI. The version 1.0.0 Zenodo-ready ZIP consists of 17 files, as documented in `RELEASE_MANIFEST_SHA256.json`.

**Archival provenance:** the deposited code snapshot corresponds to GitHub squash-merge commit `bf425c82bbb5974e814913b04b239297248ec7ac`, under `extensions/segan-local-ramping-2026/`. This README was updated later to add the record link; these post-archive GitHub documentation edits are *not* part of the original 17-file ZIP. Zenodo publication status and file integrity should be checked against the public record itself before representing that external record as independently verified.

## License

MIT (code). Recorded derived results are made available for verification as attributed research tables; **no original operational SCADA dataset is licensed for public redistribution**.

## Public synthetic execution examples (added after archived v1.0.0)

For readers without authorized SCADA inputs, use the deterministic, explicitly synthetic examples under [`examples/`](examples/README.md). The small example has three artificial feeders, eight hourly steps, and two scenarios. A second example uses five artificial feeders, 24 hours, and three scenarios. Generate inputs from pure Python formulas, then validate and execute the archived v4 solver with:

```bash
python examples/test_synthetic_data.py
python synthetic_smoke_test.py
python examples/run_synthetic_example.py --case small --time-limit 60 --output-dir demo_results
```

The [GitHub Actions synthetic workflow](../../../.github/workflows/segan-synthetic-demo.yml) checks the small case on each change without protected data. **These synthetic files were added after the original 17-file v1.0.0 Zenodo archive.** To archive them on Zenodo, publish a separate **new software version (suggested v1.1.0)** with a complete new ZIP; do not overwrite or misdescribe the existing v1.0.0 record, and do not present synthetic demo outputs as research findings.
