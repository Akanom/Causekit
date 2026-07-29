# Causal-ML real-data performance record

This is the frozen `0.7.0a1` baseline. Version `0.7.0a2` exposes fold diagnostics and uses
an algebraically equivalent, lower-allocation GCV calculation without changing the
six-point default grid; the external rows were not rerun or overwritten. A tested denser
grid slightly worsened the same real-data OOF errors and was rejected as the default.

This record compares CauseKit's native nuisance learner with optional external learners on
the same hash-verified Cattaneo maternal-smoking and birthweight data. It is a one-run
engineering benchmark, not a Monte Carlo study, causal model-selection rule, or claim that
the smallest prediction error identifies the most credible causal specification.

## Recorded design

- Dataset: Stata `cattaneo2`, 4,642 rows.
- Source SHA-256:
  `631e926eb9981828ba2e542b32c16ae08f336b9efa10621651a8a185405e0577`.
- Outcome: birthweight (`bweight`).
- Treatment: maternal smoking (`mbsmoke`).
- Covariates: marital status, maternal age, maternal education, and first-baby indicator.
- Estimator: `PartiallyLinearDML`, robust inference, three outer folds.
- Fold seed: `20260729`; every learner receives the identical outer fold assignment.
- Repetitions: exactly one fit per model. Fold-specific fits are intrinsic to cross-fitting
  and are not benchmark repetitions.
- Environment: Windows 11, Python 3.14.6, CauseKit 0.7.0a1, NumPy 2.4.6, pandas 3.0.3,
  scikit-learn 1.9.0.
- CauseKit commit: `0feb186`.
- Recorded: 2026-07-29.

## Results

| Nuisance learner | DML estimate | Standard error | Outcome OOF RMSE | Treatment OOF RMSE | Seconds | Python peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CauseKit native ridge-GCV | -224.831746 | 22.467589 | 569.120695 | 0.374450 | 0.0439 | 1.040 |
| scikit-learn RidgeCV | -224.832150 | 22.481367 | 569.155397 | 0.374469 | 0.2154 | 2.336 |
| scikit-learn histogram gradient boosting | -217.082879 | 22.721993 | 582.156481 | 0.375692 | 17.7650 | 2.155 |
| scikit-learn random forest | -216.322784 | 22.915655 | 582.872150 | 0.375629 | 2.9879 | 1.144 |

### Rejected dense-grid default

One `0.7.0a2` native-only decision run used the same data, seed, folds, environment, and
one-fit policy with 41 log-spaced penalties from `1e-6` through `1e4`. It was compared with
the frozen native row above; external learners were not rerun.

| Native grid | DML estimate | Standard error | Outcome OOF RMSE | Treatment OOF RMSE | Seconds | Python peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Six-point default | -224.831746 | 22.467589 | 569.120695 | 0.374450 | 0.0439 | 1.040 |
| 41-point candidate | -224.806044 | 22.473317 | 569.214832 | 0.374478 | 0.0469 | 0.996 |

The candidate improved its training-fold GCV objective but did not improve either held-out
RMSE and was marginally slower in this descriptive run. CauseKit therefore retained the
six-point default. This is a rejection decision for the default, not evidence that a dense
grid can never help another design; `ridge_alphas=` remains explicit.

On this declared design and single run, the CauseKit native path had the lowest elapsed
time and the lowest two observed-target OOF RMSE values. Its estimate was nearly identical
to external RidgeCV. This does not establish general superiority: the covariate set is
small, observed-target RMSE includes irreducible outcome/treatment variation, and no true
causal effect is known in real data. Different specifications may favor nonlinear learners.

`tracemalloc` reports Python-managed peak allocations and may not capture every native
allocation made by NumPy or a comparator. Timing was sequential on one machine; it is not
a hardware-independent performance guarantee.

## Reproduction

For the exact recorded values, check out commit `0feb186`. Running the harness from a
later tree intentionally records that tree's version and may change the native row.

The benchmark performs no download unless explicitly requested. With the verified source
already cached:

```bash
python benchmarks/benchmark_ml.py --models all \
  --output causekit-ml-real-benchmark-v1.json
```

To opt into the HTTPS download and mandatory hash check when the source is absent:

```bash
python benchmarks/benchmark_ml.py --models all --download \
  --output causekit-ml-real-benchmark-v1.json
```

The scikit-learn rows are optional comparators. Their absence records `unavailable`; it
does not prevent the CauseKit-native row from running and does not alter package
dependencies.
