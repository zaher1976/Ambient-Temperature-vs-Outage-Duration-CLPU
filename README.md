# Fairness-Constrained Load-Shedding Optimisation with Cold Load Pickup

Reproducibility package for the manuscript

**Ambient Temperature versus Outage Duration as Surge Drivers in a Chronically
Constrained Grid**

built on SCADA telemetry from five Iraqi governorates (206,040 hourly records,
2020–2024).

The question the code answers: a common modelling assumption treats cold-load-pickup
(CLPU) severity as a function of how long the outage lasted. Five configurations vary the surge specification and fairness constraints.
These model-based sensitivity experiments do not identify an empirical outage-duration
response: S3 uses a fixed surge coefficient, not a duration-varying kernel, and
S3 versus S4 is confounded by the fairness setting.

| Configuration | CLPU surge kernel | Fairness constraint |
|---|---|---|
| S1 | none | no |
| S2 | none | yes |
| S3 | fixed, thermally blind (0.30) | no |
| S4 | thermal kernel, Eq. (10) | yes |
| S5 | equal-weight blend of the two | yes |

## Archived release

Concept DOI (identifies the record across versions; verify the version DOI on Zenodo before citing a specific release):
[10.5281/zenodo.23205868](https://doi.org/10.5281/zenodo.23205868)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pulp<4` is load-bearing. PuLP 4.0.0 rewrote `LpVariable` on a Rust core, where it
can no longer be constructed directly; every model here fails under it with
`TypeError: LpVariable.__init__() got an unexpected keyword argument 'cat'`.
`run.py` checks the installed version first and stops with that explanation rather
than failing deep inside a model build.

TensorFlow is needed only to retrain the forecaster. The optimisation reads the
cached forecasts in `results/` and never imports it.

## Check the install without the data

```bash
python tests/test_smoke.py
```

Builds a small synthetic record with the production schema, then builds and solves
all five configurations, checks that the fairness constraint is enforced,
and checks expected model relationships on the synthetic instance, and that the two discomfort
weightings give structurally identical instances. No confidential data and no GPU
required. It does **not** reproduce the published numbers — only the real record
does that.

## Run

The raw SCADA record is governed by a Ministry of Electricity confidentiality
arrangement and is **not** redistributed here (see `data/README_DATA.md`). With a
local copy:

```bash
export CLPU_DATA=/path/to/NATIONAL_MASTER_MATRIX_CLEAN_v2.csv

python run.py check          # load the record, rebuild the thermal windows, verify them
python run.py forecast       # train the BiLSTM per governorate (needs TensorFlow)
python run.py table5         # the 30 main instances
python run.py sensitivity    # kernel-coefficient sensitivity
python run.py robustness     # the 30 instances with C_disc applied once
python run.py all            # check, table5, robustness
```

Options: `--out DIR`, `--n-scenarios N`, `--time-limit S`, `--gap G`, `--eps-fair E`,
`--k-sigma K`, `--retrain`, `--benchmark`, `--any-dataset`.

Each sweep writes every instance to CSV as it finishes and skips what is already
recorded, so an interrupted run costs at most the instance in flight. `run.py table5`
writes to `results/table5_main.csv`; pass `--out results_rerun` to keep the recorded
results untouched.

`run.py check` verifies the rebuilt 24-hour windows against the run that produced the
published tables. If it reports a mismatch, nothing computed afterwards is comparable
with them, and the output says so.

## Layout

```
clpu/config.py        every parameter, with the equation it belongs to
clpu/data.py          loading and integrity checks on the SCADA record
clpu/forecasting.py   BiLSTM per governorate, plus the LSTM/GRU benchmark
clpu/pipeline.py      thermal regimes, synchronous national windows, scenarios
clpu/milp.py          Equations (1)-(18), the five configurations, the solver wrapper
clpu/analysis.py      what is read off a solved schedule
run.py                command-line entry point
tests/test_smoke.py   self-contained check, no confidential data needed
data/                 no SCADA records; see README_DATA.md
results/              recorded outputs of the reported run
```

## What reproduces what

| Manuscript | Produced by | Recorded in |
|---|---|---|
| Table 3, Figure 2 | `run.py forecast` | `bilstm_metrics.csv` |
| Table 4 | `run.py forecast --benchmark` | `architecture_benchmark.csv`, `architecture_wilcoxon.csv` |
| Table 5, Figure 3 | `run.py table5` | `table5_main.csv` |
| Table 6 | `run.py sensitivity` | `sensitivity_lambda.csv` |
| Table 7 | `run.py table5` | `table5_gaps.csv` |
| Table 10 | long-budget re-solves | `table5_longrun_900.csv` |
| Table 12, Section 5.11 | `run.py robustness` | `table5_single.csv`, `table5_single_scenario_costs.csv`, `disc_weight_check.csv` |
| Sections 5.8, 5.10, 5.6 | `run.py table5` | `sensitivity_scenarios*.csv`, `binding_constraints.csv`, `significance.csv`, `switching.csv` |

## Two details worth knowing before reading the results

**The attained gap is not the prescribed gap.** Most instances stop far from the 3%
criterion. Large gaps mean that global optimality and cost rankings are not certified.
Six anticipative-reference instances re-solved at three times the budget returned
unchanged incumbent objectives, but this does not establish their optimality or
prove that the gap arises solely from weak dual bounds. `milp.solve` therefore parses the lower bound out of
the CBC log and records it with every objective, and `analysis.certified` answers
whether a difference between two configurations is established by those bounds.
Without the log the bound cannot be recovered afterwards.

**Solver-status caution.** The recorded CSV `status` field can read `Optimal`
even for runs that reached their time limit with a large relative MIP gap.
Do not interpret this label as a certificate of global optimality. Check the
recorded lower bound, relative gap, and time-cap fields where available.

**The discomfort price enters the objective twice.** `C_disc` multiplies both the
accumulation of Φ (Eq. 18) and its contribution to the objective (Eq. 2). That is a
weighting, not a derived quantity, so `milp.build(..., disc_mode="single")` applies
it once and `run.py robustness` re-solves the whole set that way. Two of the three
principal findings and the fairness result hold under both weightings; Section 5.11
reports exactly which conclusions depend on the choice.

## Data availability

The original operational SCADA records are not redistributed through this repository.
Aggregated, de-identified summary statistics sufficient to verify the reported
findings may be requested from the corresponding author, subject to approval by the
Renewable Energy Research Centre at Al-Nahrain University.

## Environment

`results/environment.json` records the environment the reported results were produced
in. GPU training is not bit-for-bit reproducible across hardware and library
versions, which is why that file exists; the optimisation uses fixed inputs and seeds where supported; full numerical
reproduction of the reported results requires authorised access to the withheld
SCADA data and matching software and solver environments. `run.py` writes
`results/environment_rerun.json` for the environment you ran in, so the two can be
compared.

## Citation

> Raham, Z. F., & Mohamed Salleh, F. H. (2026). *Fairness-Constrained Load-Shedding
> Optimisation with Cold Load Pickup: Ambient Temperature versus Outage Duration as
> Surge Drivers in a Chronically Constrained Grid* (reproducibility package).
> Zenodo. https://doi.org/10.5281/zenodo.23205868

See `CITATION.cff` for machine-readable metadata.

## Licence

MIT. See `LICENSE`.
