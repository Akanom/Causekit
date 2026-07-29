# Honest R-learner real-data benchmark

This is a one-run engineering and honest-evaluation record, not a benchmark repetition,
causal model-selection proof, or claim that the observed ranking identifies true CATEs.
It was run on 29 July 2026 after the public R-learner evaluation surface was implemented.
The earlier Cattaneo and NSW scalar-DML benchmark files were not rerun or overwritten.

## Design

- Dataset: National Supported Work experimental sample (`nsw_mixtape`), 445 rows.
- Source SHA-256:
  `fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4`.
- Outcome: 1978 earnings (`re78`).
- Treatment: randomized job-training assignment (`treat`).
- Pre-treatment covariates: age, education, Black and Hispanic indicators, marital
  status, no-degree indicator, and 1974/1975 earnings.
- Honest design: 222 construction and 223 evaluation observations, treatment-stratified
  with seed `20260729`; evaluation-index SHA-256
  `f2ce0c338c4331a4ba211e0e32f44aeaa6a775d78b3f821e20bebbaa4977032f`.
- Construction: three outer folds and native outcome/penalized-Logit propensity nuisances
  in every row. Only the weighted CATE learner changes across rows.
- Evaluation: four tie-preserving calibration groups and 999 seeded Rademacher max-t
  draws. Every model receives the same roles, folds, nuisance specification, and draws.
- Environment: Windows 11, Python 3.14.6, CauseKit 0.7.0a2, NumPy 2.4.6, pandas 3.0.3,
  scikit-learn 1.9.0.
- Repetitions: exactly one fit per model.

Because treatment was randomized, the separately validated randomized ATE remains the
primary average-effect analysis. This benchmark asks only whether a construction-fitted
CATE proxy improves held-out residual R-loss and calibrates on this split. Real data do
not reveal individual true treatment effects or PEHE.

## Results

| Weighted CATE learner | Honest R-loss | Constant R-loss | R-loss gain | Differential calibration | p-value vs 0 | Seconds | Python peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CauseKit native ridge-GCV | 40,564,476.953 | 40,167,364.991 | -0.9886% | -0.0151 | 0.9826 | **0.615** | **1.598** |
| CauseKit native adaptive spline-ridge | 40,741,158.012 | 40,167,364.991 | -1.4285% | 0.1109 | 0.8332 | 0.931 | 1.616 |
| scikit-learn standardized RidgeCV | **40,166,001.648** | 40,167,364.991 | **0.0034%** | 3.9100 | 0.9183 | 0.707 | 2.602 |
| scikit-learn histogram gradient boosting | 49,726,216.222 | 40,167,364.991 | -23.7976% | -0.2826 | 0.1040 | 2.760 | 1.811 |
| scikit-learn random forest | 47,042,475.606 | 40,167,364.991 | -17.1162% | -0.2888 | 0.1679 | 1.380 | 1.674 |

Lower held-out R-loss is preferable. Standardized scikit-learn RidgeCV has the smallest
loss on this split, but its improvement over the constant baseline is only 0.0034% and its
differential-calibration estimate is extremely imprecise. CauseKit native ridge-GCV is
0.99% worse than the constant baseline. The later opt-in native spline row is 1.43% worse;
it was run once after implementation without rerunning the four original rows. Both tree
learners are materially worse. None rejects a zero differential-calibration coefficient at
conventional levels, so this evidence does not support a stable heterogeneous ranking.

The native learner remains the default because it is dependency-free, its tuning occurs
only inside construction, and it is the fastest and smallest row here—not because it won
the real-data R-loss comparison. The external ridge selected much stronger shrinkage and
behaved almost like the constant comparator. Changing CauseKit's GCV rule after observing
the evaluation result would leak the honest role, so this result is retained as validation
evidence rather than used for retuning.

The adaptive spline learner is also dependency-free and does recover a known piecewise
nonlinear CATE in the seeded simulation contract. Its construction-only weighted GCV
selects among zero-, one-, and three-knot additive linear-spline bases. The negative NSW
result means it remains explicit opt-in through `cate_factory=NativeSplineRidgeCATE`, not
the default and not a claim of superiority over established nonlinear learners.

The saved JSON is external to the repository at
`%LOCALAPPDATA%/causekit/benchmarks/causekit-rlearner-nsw-benchmark-v1.json`; its SHA-256 is
`5faf504e404db7fdcf9fc60c85584b545227a5eac02bc2f6fe574249d16e7685`.
The separate one-row native spline artifact is
`%LOCALAPPDATA%/causekit/benchmarks/causekit-rlearner-nsw-native-spline-v1.json`; its
SHA-256 is
`693a45a2dfa6317a95fbf1f9b43c1c8794792ac5aa663f45f995a263c458b955`.
`tracemalloc` reports Python-managed allocations and may not include every native
allocation. Timing is machine-specific.

## Reproduction

With the verified source already cached:

```bash
python benchmarks/benchmark_rlearner.py --models all \
  --output causekit-rlearner-nsw-benchmark-v1.json
```

If absent, opt into the HTTPS download and mandatory hash verification with `--download`.
Comparator absence is recorded as unavailable and never changes CauseKit's dependencies.
