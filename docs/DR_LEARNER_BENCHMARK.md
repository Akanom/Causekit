# Honest DR-learner real-data benchmark

This is a one-run model comparison, not repeated cross-validation, a causal-forest study,
or evidence that any learner is universally best. It changes only the unweighted final
CATE regression inside the same `DRLearner` specification. The settled R-learner benchmark
artifacts were not rerun.

## Reproduction contract

- Data: `nsw_mixtape`, loaded through CauseKit's HTTPS/SHA-256 dataset registry.
- Source SHA-256: `fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4`.
- Outcome/treatment: 1978 earnings and randomized NSW assignment.
- Pre-treatment covariates: age, education, Black, Hispanic, married, no-degree, 1974
  earnings, and 1975 earnings.
- Observations: 445; construction: 222; evaluation: 223.
- Seed: `20260730`; outer folds: 3; evaluation fraction: 0.5; groups: 4; max-t draws: 999.
- Environment: Windows 11, Python 3.14.6, CauseKit 0.7.0a2 at benchmark execution,
  NumPy 2.4.6, pandas 3.0.3, optional scikit-learn 1.9.0.
- Repetitions: exactly one.

```powershell
$output = Join-Path $env:LOCALAPPDATA `
  "causekit\benchmarks\causekit-drlearner-nsw-v1.json"
python benchmarks/benchmark_drlearner.py --models all --output $output
```

All four rows have evaluation-index SHA-256
`3e95f6f725a36d48919850440e74c1d41c0ab7912788dad30cab0e864d8920df`.

## Results

| Final CATE learner | Honest DR loss | Gain over construction constant | Seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: |
| CauseKit native ridge-GCV | 148,166,882.7104 | 0.05359% | 0.9642 | 1.5757 |
| scikit-learn RidgeCV | 148,166,882.7104 | 0.05359% | 0.9135 | 2.5922 |
| scikit-learn histogram boosting | 184,272,268.3367 | -24.3014% | 4.5594 | 1.8084 |
| scikit-learn random forest | 163,459,384.1342 | -10.2620% | 1.9313 | 1.6699 |

The constant loss was `148,246,322.9693`. Native ridge-GCV selected the upper grid
boundary `alpha=10000`. Native and scikit-learn ridge produced the same displayed
predictions, loss, calibration, groups, and max-t critical value. This is expected because
both standardized ridge paths use the same six penalties and GCV/leave-one-out identities
in this design; it is not evidence that CauseKit invented a different ridge principle.

Native used about 39% less Python-managed peak memory and was about 5.5% slower than the
optional RidgeCV comparator. The heterogeneity-zero p-value was `0.8895` for both ridge
rows, so this split supplies no evidence that their weak positive loss gain corresponds to
meaningful effect heterogeneity. The tree results show that extra flexibility can overfit
the noisy DR score in a 222-row construction sample.

## Interpretation boundary

The randomized design supports conditional causal interpretation more directly than an
observational benchmark, but a single honest split remains noisy. DR loss is score loss,
not observed unit-level CATE error. Timing and `tracemalloc` values are descriptive and
machine-specific. The evidence supports native ridge-GCV as a small, auditable default; it
does not establish universal superiority, unit-level interval validity, RATE/policy value,
or publication-scale coverage.

## Scaling smoke

`benchmarks/benchmark_drlearner_scaling.py` generated 2,000 rows in 500 four-row clusters,
kept clusters intact across roles/folds, and used clustered inference. On the recorded
environment it completed in `1.1791` seconds with `0.9664 MiB` Python-managed peak memory.
The retained influence surface was `800 x 4`; nuisance predictions had exactly three
columns. The execution path constructs no observation-pair or distance matrix. This is a
linear-allocation smoke, not a universal timing guarantee.
