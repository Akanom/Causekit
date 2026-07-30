# Longer and staggered composition-robust DiD promotion evidence

This record promotes the longer-design `RepeatedCrossSectionDiD(composition="robust")`
path and the aligned `did_rcs_composition_test`. It covers task/fold orchestration,
full-sample influence alignment, target-share aggregation, conditional placebos,
simultaneous event-study bands, diagnostic behavior, real-data sensitivity, and fixed-size
performance. It does not promote survey-weighted targets or diagnostic-based estimator
selection.

## Contract and official diagnostic mapping

Hand tests reconstruct every pair estimate, normalized weight, `n/n_pair` influence
embedding, target-share aggregation term, HC1/CR1 covariance, conditional placebo, and
fixed-seed max-t band. Multiple cohorts pass under never-treated and not-yet-treated
controls. Refusal tests cover global/fold four-cell support, leakage, row/PSU role changes,
class/order errors, sample and target misalignment, covariance/cluster mismatch, and
singular difference covariance.

The official R diagnostic harness pins `compdid` 0.1.0 commit
`894bd65a952c30f01a4e0005efba4cb335065eb7` and stationarity-test blob
`e06750e061948b5c02f08741f651b0e8fd4155f5`. Base R 4.5.1 executes
`drdid_stationarity_test()` on the same fixed estimates and aligned influence vectors.
The point difference, empirical difference-influence variance, statistic, and p-value
agree after recording the official HC0-to-CauseKit-HC1 mapping
`W_HC0 = W_HC1 * n / (n - 1)`. The saved output SHA-256 is
`e77b5a8d11c28aa21060f99a0c00a449b2331b796a6cb0efb066faf6421309c8`.

Official `compdid` exposes a two-period nonstationary estimator, not the aligned
longer/staggered target-share vector. CauseKit therefore records that external comparator
cell as unavailable and does not manufacture parity from a different estimand.

Reproduce with:

```bash
Rscript benchmarks/validate_did_rcs_compdid_diagnostic_reference.R PATH_TO_COMPDID_CHECKOUT
pytest tests/validation/test_did_rcs_compdid_parity.py -q
```

## Hash-pinned real-data sensitivity

The real-data runner verifies Stata's `hospdd` analysis CSV at SHA-256
`db9d3b7182eb8abd2e1c5190305d6edce431b57f690a87550dba27e61df72990` and the source
artifact at SHA-256
`e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb`.
The sample has 322 hospital-period rows, 46 hospitals, and seven periods. Because hospitals
recur, inference clusters by hospital; this is a sensitivity execution and does not claim
the rows are independent repeated cross sections.

Five immutable whole-hospital folds use `hospital_scaled` as the observed covariate. The
robust path reports four post coordinates, two conditional placebos, six pairs, and 120
nuisance-fold fits; the stationary path reports 150. Robust ESavg is `0.829703` with SE
`0.056112`; stationary ESavg is `0.841683` with SE `0.052885`. The aligned four-coordinate
diagnostic is `F(4,45)=0.193036`, `p=0.940792`. This non-rejection does not establish
stationarity and does not select the stationary result. Robust generalized probabilities
range from `0.08378` to `0.38805`.

Reproduce with:

```bash
python benchmarks/validate_did_rcs_composition_longer_real_data.py
```

The canonical report is
`benchmarks/did_rcs_composition_longer_real_data_evidence.json`.

## Fixed-size performance

The recorded Python 3.14.6 / Windows 11 run fits 120,000 rows, 15 global cells, eight
pairs, five post effects, three conditional placebos, and 64 fold-level nuisance tasks.
It completes in `2.101` seconds with `346.961` MiB Python-managed peak memory. The frozen
gates are 30 seconds and 1,024 MiB; elapsed time and allocation peaks remain
machine-specific diagnostics.

Reproduce with:

```bash
python benchmarks/benchmark_did_rcs_composition_longer.py --n-observations 120000 --n-splits 2 --measure-memory
```

The canonical report is
`benchmarks/did_rcs_composition_longer_performance_evidence.json`.

## Publication-scale inference

The fixed-seed promotion simulation crosses favorable stationarity and composition shift
with observation and indivisible-PSU inference. Each cell uses 500 replications and 199
Rademacher draws, nonlinear correctly specified saturated probability and quadratic
outcome nuisances, and no refitting in post-estimation. The run completes 4,000 estimator
fits and 108,000 nuisance-fold fits with zero refusals.

| Design | Sampling | Bias range | SE-ratio range | Pointwise coverage | Joint coverage | Diagnostic rejection | Pretrend rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stationary | Observation | -0.0025 to -0.0017 | 0.986–0.995 | 0.942–0.946 | 0.946 | 0.068 | 0.054 |
| Stationary | PSU | -0.0030 to 0.0021 | 0.948–0.971 | 0.932–0.942 | 0.930 | 0.060 | 0.056 |
| Composition shift | Observation | -0.0003 to 0.0037 | 0.966–1.054 | 0.934–0.950 | 0.950 | 1.000 | 0.052 |
| Composition shift | PSU | 0.0002 to 0.0125 | 0.963–1.025 | 0.930–0.962 | 0.950 | 1.000 | 0.034 |

All preregistered bias, SE-calibration, pointwise and simultaneous coverage, Monte Carlo
uncertainty, overlap, conditional-pretrend size, diagnostic size/power, audit-count, and
zero-refusal gates pass. This evidence is conditional on the simulated nuisance and
identification contracts; it is not protection against unmeasured composition changes.

Reproduce with:

```bash
python benchmarks/validate_did_rcs_composition_longer_promotion.py --replications 500 --workers 4
pytest tests/validation/test_did_rcs_composition_longer_promotion.py -q
```

The hash-bound report is
`benchmarks/did_rcs_composition_longer_promotion_evidence.json`.
