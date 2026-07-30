# Pairwise composition-robust DiD promotion evidence

This record promotes the implemented two-group, two-period
`RepeatedCrossSectionDiD(composition="robust")` slice through three independent gates:
hash-pinned real-data sensitivity, fixed-size performance, and publication-scale
pointwise coverage. It does not promote staggered composition robustness, a composition
diagnostic, survey weights, or estimator selection after a stationarity pretest.

## Hash-pinned real-data sensitivity

The data-preparation script verifies the official `compdid` source checkout at commit
`894bd65a952c30f01a4e0005efba4cb335065eb7`, Git blob
`6a6a8bbe9792bc6385849421a7fd0d76692cd79e`, and raw RDA SHA-256
`37f113f1c706a3b35c325b996884c843beb9a4f773c3928d505ca51b285a7953`.
It applies the authors' Sequeira preprocessing and writes an external, canonical-LF
analysis CSV with SHA-256
`48e3d0cc1bd2eb757e4ac0ad24a9cc60946c6041cb4c56debaac95e4aaebb492`.
The verified sample has 1,084 observations, 131 HS-code PSUs, and `(00,01,10,11)` cell
counts `(84,824,56,120)`.

The sensitivity run uses five immutable whole-PSU folds, a benchmark-owned standardized
ridge softmax with fixed penalty `0.5`, standardized ridge outcome regressions with fixed
penalty `1.0`, and CR1 inference. Every prediction is produced by a declared
`CrossFitter` task. The official paper values are retained only as descriptive references
because its local-polynomial nuisance and inference contract is different.

| Outcome | Robust estimate (SE) | Stationary estimate (SE) | Robust − stationary |
| --- | ---: | ---: | ---: |
| `bp` | -0.822589 (0.110794) | -0.591836 (0.107130) | -0.230752 |
| `lba` | -8.581601 (1.057825) | -6.037488 (0.819024) | -2.544113 |
| `lba_value` | -0.035854 (0.008426) | -0.026184 (0.007048) | -0.009671 |
| `lba_tonnage` | -4.069378 (0.470987) | -3.064027 (0.453524) | -1.005351 |

The differences are material. They are reported jointly as nuisance-and-target
sensitivity and are not used to select an estimator. The minimum robust generalized
probability is `0.01852`; the maximum normalized robust weight is `29.72`. These support
diagnostics warrant substantive scrutiny even though the mechanical overlap gate passes.

Reproduce from the repository root:

```bash
Rscript benchmarks/prepare_did_rcs_composition_real_data.R PATH_TO_COMPDID_CHECKOUT
python benchmarks/validate_did_rcs_composition_real_data.py
```

The canonical report is
`benchmarks/did_rcs_composition_real_data_evidence.json`. The derived CSV remains outside
the repository under the user's local CauseKit parity cache; the source RDA and prepared
data are not vendored.

## Fixed-size performance

`benchmarks/benchmark_did_rcs_composition.py` creates 100,000 rows with exactly 25,000
observations in each group-period cell. The two-fold task graph contains one four-class
probability fit and three nonlinear outcome fits per fold, or eight audited fits total.
On the recorded Python 3.14.6 / Windows 11 environment it completed in `0.229` seconds
with a `44.246` MiB Python allocation peak. The frozen gates are 10 seconds and 512 MiB;
time and memory remain machine-specific diagnostics. The implementation does not form an
observation-by-observation matrix.

Reproduce with:

```bash
python benchmarks/benchmark_did_rcs_composition.py --n-observations 100000 --n-splits 2
```

## Publication-scale coverage

The fixed-seed certificate crosses two unequal-cell designs with observation and
indivisible-PSU inference. Fold-fitted nuisances are saturated in binary `X` for the
group-period probabilities and use a nonlinear `(1, X, Z, Z^2)` outcome basis. The
favorable design satisfies stationary composition; the shifted design changes treated
composition while preserving overlap. Robust and stationary results are always retained.

The run uses 1,000 replications per design/sampling-unit cell, producing 8,000 estimator
fits and 72,000 fold-level nuisance fits with zero refusals. Coverage gates are
`[0.90,0.99]`, mean-SE/empirical-SD gates are `[0.75,1.25]`, the absolute-bias gate is
`0.08`, and every coverage Monte Carlo SE is below `0.0090`.

| Design | Sampling | Score | Bias | SE ratio | Coverage |
| --- | --- | --- | ---: | ---: | ---: |
| Stationary unequal | Observation | Robust | 0.0093 | 1.009 | 0.950 |
| Stationary unequal | Observation | Stationary | 0.0058 | 0.991 | 0.947 |
| Stationary unequal | PSU | Robust | 0.0043 | 1.003 | 0.953 |
| Stationary unequal | PSU | Stationary | -0.0018 | 1.016 | 0.951 |
| Composition shift | Observation | Robust | 0.0005 | 0.993 | 0.947 |
| Composition shift | Observation | Stationary | 0.0017 | 0.912 | 0.932 |
| Composition shift | PSU | Robust | -0.0040 | 0.959 | 0.943 |
| Composition shift | PSU | Stationary | -0.0040 | 0.882 | 0.912 |

Under stationarity, robust mean SE is `1.222` times the stationary mean SE for
observation inference and `1.182` times for PSU inference. This is the measured cost of
robustness in the favorable design. Under composition shift, the stationary score still
estimates its pooled-treated target accurately, but its bias when misreported for the
treated target-period estimand is `-0.7316` and `-0.7374`, close to the analytical
`-11/15`. Robust target-period bias is `0.0005` and `-0.0040`.

Reproduce with:

```bash
python benchmarks/validate_did_rcs_composition_promotion.py --replications 1000 --workers 4
```

The hash-bound report is
`benchmarks/did_rcs_composition_promotion_evidence.json`. The evidence supports the
pairwise score's pointwise inference under the contracted designs. It is not evidence for
unmeasured composition robustness, simultaneous inference in a longer robust event
study, survey transport, or an automatic stationarity decision rule.
