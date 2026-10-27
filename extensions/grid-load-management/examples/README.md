# Public, fully synthetic Grid Load Management MILP execution examples

These example inputs are **100% artificial, created from explicit deterministic mathematical formulas** using the Python standard library. They do **not** use the Iraq SCADA record, frozen model windows, demand reconstructions, or fitted forecast residuals. **Never cite their numerical outputs as Grid Load Management manuscript results.**

## Contents

| Path | Purpose | Dimensions |
|---|---|---|
| `synthetic_small.json` | Short test suitable for first-time users | 3 synthetic feeders × 8 hourly steps × 2 scenarios |
| `synthetic_day24.json` | More realistic *size* / one-day tutorial, still artificial | 5 synthetic feeders × 24 hourly steps × 3 scenarios |
| `generate_synthetic_examples.py` | Reproduce both inputs exactly from a fixed seed (314159) | Python standard library |
| `test_synthetic_data.py` | Validate schema, dimensionality, positivity and byte-for-byte determinism | No solver needed |
| `run_synthetic_example.py` | Solve **the archived v4 MILP** with local penalty=0 and 0.01, validate output, export CSV+JSON | Requires PuLP 3.3.2/CBC and NumPy |

## Run locally or in Google Colab

From `extensions/grid-load-management`:

```bash
python -m pip install -r requirements.txt
python examples/test_synthetic_data.py
python synthetic_smoke_test.py
python examples/run_synthetic_example.py --case small --time-limit 45 --output-dir demo_results
```

For a larger teaching example:

```bash
python examples/run_synthetic_example.py --case day24 --time-limit 120 --gap 0.01 --output-dir demo_results_day24
```

The MILP benchmark may take longer on shared machines. The CLI deliberately fails if CBC does not report a solution of type `Optimal` within the budget. Independently validated feasibility is checked with residuals, and the reported target gap is **not** claimed to be the actual attained gap or to prove absolute global optimality. Use the CBC log to inspect termination and bounds.

To re-generate the JSON datasets and verify that no precomputed SCADA-derived data is hidden:

```bash
python examples/generate_synthetic_examples.py
python examples/test_synthetic_data.py
```

### Result files created **only after successful execution**

- `demo_results/small_with_local_penalty_schedule.csv` — solved feeder/hour allocations for each synthetic scenario
- `demo_results/small_without_local_penalty_schedule.csv` — second comparison point
- `demo_results/small_comparison.json` — objective, unserved demand, CLPU, local and aggregate total variation, fairness, and validation residuals

These files are created by your own run. There are no fabricated expected solver values in the archive. The included inputs do *not* reproduce published percentages (78.16% or 85.51%) or test SCADA data access. For full 72-case research reproduction, the data custodian must separately provide the restricted frozen input package.
