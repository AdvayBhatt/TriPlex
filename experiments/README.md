# Research commands

Run from the repository root. Normal predictions use the two commands in `scripts/`.

| File | Purpose |
|---|---|
| `mixed_model.py` + `trial_adjustment.R` | New historical-only joint trial adjustment; Python genomic forecast after R/lme4 REML |
| `stress_test.py` | Chronological model choice, conditional family bootstrap, site/roster sensitivity and interval audit |
| `model_experiments.py` | Original genomic comparison and shared marker/math helpers |
| `select_model.py` | Original first-stage selection and calibration experiment |
| `roster_experiment.py` | Original planned-site adjustment comparison and shared roster transformation |
| `audit_bridge.py` | Forensic comparison with reviewed legacy caches; not a prediction pipeline |

Examples:

```text
python experiments/mixed_model.py --output-dir outputs/my_mixed_model
python experiments/mixed_model.py --output-dir outputs/my_mixed_model --summarize
python experiments/mixed_model.py --years 2007 --clusters 1 --output-dir outputs/my_confirmation_C1
python experiments/mixed_model.py --output-dir outputs/my_mixed_model --summarize --confirmation-folder outputs/my_confirmation_C1
python experiments/stress_test.py --output-dir outputs/my_robustness
```

The mixed-model experiment needs R with `lme4` and `jsonlite`. Override `--rscript` if R is installed elsewhere. The default development years are 2004–2006, always trained on strictly earlier seasons starting in 2001. Use a separate output directory for 2007 confirmation, after freezing a choice from development results. No 2008 option is exposed.

Some existing predictor helpers still reside in the original comparison modules to avoid an unrelated statistical rewrite during this experiment. The production runners load them explicitly. Archived reports use the pre-reorganization paths. Historical artifact verifiers clearly report when they use the compressed historical source snapshot rather than current source.
