# Causal-ML real-data performance records

These are deliberately one-run engineering records on two hash-verified datasets, not
benchmark repetitions. The original Cattaneo `0.7.0a1` baseline remains frozen. Version
`0.7.0a2` exposes fold diagnostics and uses an algebraically equivalent, lower-allocation
GCV calculation without changing the six-point default grid; the Cattaneo external rows
were not rerun or overwritten. A tested denser grid slightly worsened the same Cattaneo
out-of-fold (OOF) errors and was rejected as the default.

Each record compares CauseKit's native nuisance learner with optional external learners on
identical folds within that dataset. Neither record is a Monte Carlo study, causal
model-selection rule, or claim that the smallest prediction error identifies the most
credible causal specification.

## Cattaneo maternal-smoking design

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

## NSW job-training design

The second record tests the same CauseKit estimator in a different domain and a much
smaller sample without rerunning the Cattaneo benchmark.

- Dataset: National Supported Work experimental sample (`nsw_mixtape`), 445 rows.
- Source SHA-256:
  `fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4`.
- Outcome: 1978 earnings (`re78`).
- Treatment: job-training assignment (`treat`).
- Pre-treatment covariates: age, education, Black and Hispanic indicators, marital
  status, no-degree indicator, and 1974/1975 earnings.
- Estimator: `PartiallyLinearDML`, robust inference, three outer folds.
- Fold seed: `20260729`; every learner receives the identical outer fold assignment.
- Repetitions: exactly one fit per model.
- External ridge fairness rule: `StandardScaler` and `RidgeCV` are one fold-local pipeline,
  with the same six penalties as CauseKit. Scaling is fitted only on the outer training
  sample. The tree learners retain their declared Cattaneo configurations.
- Environment: Windows 11, Python 3.14.6, CauseKit 0.7.0a2, NumPy 2.4.6, pandas 3.0.3,
  scikit-learn 1.9.0.
- CauseKit estimator implementation commit: `ece84f7` (the benchmark-harness extension
  and this record were the only working-tree changes at execution).
- Recorded: 2026-07-29.

Because treatment was randomized, `RandomizedATE` remains the identification-appropriate
primary analysis for this dataset; its separately parity-validated unadjusted estimate is
1794.342382 (HC1 standard error 670.824491). DML is used here only to stress the nuisance
learning path on a second design. The randomized estimate is not the unknown ground truth
and is not a model-selection target.

| Nuisance learner | DML estimate | Standard error | Outcome OOF RMSE | Treatment OOF RMSE | Seconds | Python peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CauseKit native ridge-GCV | 1823.114035 | 668.916001 | **6608.467270** | 0.491269 | **0.0456** | **0.210** |
| scikit-learn standardized RidgeCV | 1790.295534 | 676.185984 | 6609.956633 | **0.490387** | 0.1255 | 1.351 |
| scikit-learn histogram gradient boosting | 1809.187334 | 672.085065 | 7013.913010 | 0.516963 | 6.6055 | 0.893 |
| scikit-learn random forest | 1774.886429 | 638.692982 | 6744.085770 | 0.495810 | 3.1711 | 0.402 |

The ridge comparison is effectively split on predictive error: CauseKit's outcome RMSE is
0.023% lower, while standardized scikit-learn RidgeCV's treatment RMSE is 0.180% lower.
CauseKit is 2.75 times faster and uses 6.43 times less Python-managed peak memory than the
external ridge pipeline in this single run. Both nonlinear configurations have higher OOF
RMSE for both nuisance targets. All four effect estimates lie between 1774.89 and 1823.11;
their proximity is useful sensitivity evidence but does not reveal which estimate is
closest to the unknown causal effect.

Together, the two datasets support retaining native ridge-GCV as the package default: it
is dependency-free, computationally smallest, and prediction-competitive across both
declared designs. They do not prove that ridge is best for nonlinear confounding or CATE
estimation. Honest R-learner evaluation remains a separate contract.

The saved NSW JSON is external to the repository at
`%LOCALAPPDATA%/causekit/benchmarks/causekit-ml-nsw-benchmark-v1.json`; its SHA-256 is
`a41758370e4ce0e2fd14003699f33154842ee35d867a3afb7547266e0b275a8c`.

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

Run the separate NSW design once with:

```bash
python benchmarks/benchmark_ml.py --dataset nsw_mixtape --models all \
  --output causekit-ml-nsw-benchmark-v1.json
```

To opt into the HTTPS download and mandatory hash check when the source is absent:

```bash
python benchmarks/benchmark_ml.py --models all --download \
  --output causekit-ml-real-benchmark-v1.json
```

Add `--dataset nsw_mixtape` to the download command when the verified NSW source is absent.

The scikit-learn rows are optional comparators. Their absence records `unavailable`; it
does not prevent the CauseKit-native row from running and does not alter package
dependencies.
