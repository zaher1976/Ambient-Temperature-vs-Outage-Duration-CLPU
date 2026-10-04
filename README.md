# Fairness-Constrained Load Shedding with Cold Load Pickup

Repository accompanying the manuscript:

**Ambient Temperature versus Outage Duration as Surge Drivers in a Chronically Constrained Grid**

---

## Overview

This repository contains the complete reproducibility package used in the study, including:

- Regional BiLSTM forecasting workflows
- Forecast benchmarking experiments
- Scenario generation methods
- Fairness-constrained MILP optimisation
- Solver-gap diagnostics
- Budget-stability verification
- Sensitivity analyses
- Reproduced manuscript result tables

---

## Reproducibility

The repository reproduces all major results reported in the manuscript:

### Forecasting

- Table 3: Regional BiLSTM forecasting performance
- Table 4: Architecture benchmark comparison

### Optimisation

- Table 5: Main optimisation results
- Table 6: Lambda sensitivity analysis
- Table 7: Solver-gap diagnostics
- Table 10: Budget-stability verification

### Additional Analyses

- Scenario convergence studies
- Fairness-tolerance sensitivity
- Operator validation
- Binding-constraint analysis
- Switching-mechanism analysis

---

## Repository Structure

```text
data/
src/
results/
```

### data/

Contains data availability documentation only.

### src/

Contains source code modules:

- forecasting.py
- scenario_generation.py
- thermal_regimes.py
- milp_builder.py
- optimization_runner.py
- gap_analysis.py
- fairness_metrics.py

### results/

Contains all processed outputs required to reproduce manuscript figures and tables.

---

## Data Availability

The original SCADA records used in this study are not redistributed through this repository.

This repository provides:

- Source code
- Optimisation models
- Forecasting workflows
- Reproducibility artefacts
- Published result tables

Researchers requiring access to the original operational records should contact the original data owners.

---

## Computational Environment

Software versions and runtime information are archived in:

```text
results/environment.json
```

---

## Citation

Please cite the associated manuscript and this repository when using this work.
`# Fairness-Constrained Load Shedding with Cold Load Pickup

Repository accompanying the manuscript:

**Ambient Temperature versus Outage Duration as Surge Drivers in a Chronically Constrained Grid**

---

## Overview

This repository contains the complete reproducibility package used in the study, including:

- Regional BiLSTM forecasting workflows
- Forecast benchmarking experiments
- Scenario generation methods
- Fairness-constrained MILP optimisation
- Solver-gap diagnostics
- Budget-stability verification
- Sensitivity analyses
- Reproduced manuscript result tables

---

## Reproducibility

The repository reproduces all major results reported in the manuscript:

### Forecasting

- Table 3: Regional BiLSTM forecasting performance
- Table 4: Architecture benchmark comparison

### Optimisation

- Table 5: Main optimisation results
- Table 6: Lambda sensitivity analysis
- Table 7: Solver-gap diagnostics
- Table 10: Budget-stability verification

### Additional Analyses

- Scenario convergence studies
- Fairness-tolerance sensitivity
- Operator validation
- Binding-constraint analysis
- Switching-mechanism analysis

---

## Repository Structure

```text
data/
src/
results/
```

### data/

Contains data availability documentation only.

### src/

Contains source code modules:

- forecasting.py
- scenario_generation.py
- thermal_regimes.py
- milp_builder.py
- optimization_runner.py
- gap_analysis.py
- fairness_metrics.py

### results/

Contains all processed outputs required to reproduce manuscript figures and tables.

---

## Data Availability

The original SCADA records used in this study are not redistributed through this repository.

This repository provides:

- Source code
- Optimisation models
- Forecasting workflows
- Reproducibility artefacts
- Published result tables

Researchers requiring access to the original operational records should contact the original data owners.

---

## Computational Environment

Software versions and runtime information are archived in:

```text
results/environment.json
```

---

## Citation

Please cite the associated manuscript and this repository when using this work.
