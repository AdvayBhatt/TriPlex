# Day 1 exploratory notebooks

These notebooks are retained as the team's original exploration, not as the current submission workflow. Their saved outputs and historical line rankings do not establish a January 2008 forecast.

- `01_eda.ipynb`: historical data exploration; descriptive same-season trait relationships are not deployable preplant predictors.
- `02_preprocessing.ipynb`: older joins and identifier handling. Use the current loader for suffix normalization, C2 support and auditing conflicted identities.
- `03_modeling.ipynb`: within-population modeling and historical validation. This does not test transfer to the wholly new 2008 populations; moisture/stability composites from observed outcomes cannot be used for the January decision.

Use `python scripts/run_pipeline.py --sample` from the repository root for the current judge workflow, and read the root README before running full data. Implementation and verification are documented in `report/PERFORMANCE_EXPERIMENTS.md`.
