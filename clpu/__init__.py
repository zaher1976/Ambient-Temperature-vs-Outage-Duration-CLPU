"""Fairness-constrained, CLPU-aware load-shedding optimisation.

Reproducibility package for the study of cold-load-pickup surge drivers under
chronic generation scarcity. Equation numbers in the source refer to the
manuscript.

Typical use::

    from clpu import config as C, data, forecasting, pipeline, milp, analysis

    scada = data.load("NATIONAL_MASTER_MATRIX_CLEAN_v2.csv")
    bilstm, forecast = forecasting.run(scada, "results")
    pipe = pipeline.Pipeline(scada, forecast)
    prob, h = milp.build(pipe, regime=32.8, config="S4")
    info = milp.solve(prob, log_path="results/cbc.log")
    metrics = analysis.report(prob, h)

or, from the command line::

    python run.py table5 --data PATH --out results
"""

__version__ = "1.2.0"

from . import analysis, config, data, milp, pipeline  # noqa: F401

__all__ = ["analysis", "config", "data", "milp", "pipeline", "forecasting"]
