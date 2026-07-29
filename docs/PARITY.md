# Cross-software parity register

This register is the package-wide Python/R/Stata promotion gate. A row is a parity pass
only when the external implementation targets the same estimand on the same estimation
sample with aligned options and finite-sample corrections. Similar labels or estimates
are not enough.

## Required evidence record

Every comparator harness must record:

- the causal estimand and identifying moments;
- data source or generator, seed, and retained sample;
- CauseKit commit/version and external software/package versions;
- treatment, outcome, covariate, instrument, cohort, time, and cluster mapping;
- intercept, weighting, support, caliper, tie, anticipation, nuisance, and aggregation
  options as applicable;
- covariance definition, degrees-of-freedom or debiasing correction, and reference
  distribution;
- compared fields, absolute/relative tolerances, and observed maximum discrepancies; and
- a one-command reproduction path.

Use `pass`, `fail`, `pending`, `unavailable`, or `non-comparable`. `Unavailable` means the
ecosystem has no identified implementation. `Non-comparable` means an apparent analogue
uses different estimands or moments. Neither counts as a pass.

## Current matrix

| CauseKit family | Python comparator | R comparator | Stata comparator | Current status |
| --- | --- | --- | --- | --- |
| `IV2SLS` | `linearmodels` aligned coefficient/covariance tests | real-data matrix/HC1 contract passes | Stata/IC 17 real-data native `ivregress` HC1 harness passes | pass |
| `RandomizedATE` | `statsmodels` regression/covariance identities | real NSW raw/Lin HC1 contracts pass | Stata/IC 17 real-data native robust regressions pass | pass |
| `IPWATE` / `AIPWATE` ATE/ATT/ATC | analytical score identities pass | real Cattaneo supplied-nuisance influence contracts pass | Stata/IC 17 same conditional influence contracts pass; `teffects` first-step variance is non-comparable | pass for aligned conditional-nuisance contract |
| `CrossFitter` | protocol and leakage/alignment tests; not itself an estimand | non-comparable | non-comparable | internal protocol |
| `NearestNeighborMatch` fixed-score ATT/ATC/ATE | hand/reuse/variance identities pass | CRAN `Matching` 4.10-15 fixture and real Cattaneo point estimates pass | Stata/MP 17 fixture and Stata/IC 17 real Cattaneo point harness pass | pass |
| `NearestNeighborMatch` estimated-Logit ATT/ATC/ATE | hand formulas and independent `statsmodels.Logit` pass | `Matching` conditions on supplied scores: non-comparable | Stata/IC 17 `teffects psmatch` fixture passes | pass for available estimand-aligned comparators |
| Conventional staggered DiD | hand/influence identities pass | real hospital group-time/influence contract passes | Stata/IC 17 same group-time/influence contract passes; `didregress` common-effect aggregation is non-comparable | pass for aligned contract |
| Efficient DiD, no covariates | native result checked on fixture and real data | pinned public `edid` commit passes fixture and real hospital data | unavailable: Stata heterogeneous DiD does not implement PT-All optimal weighting | pass for available aligned comparator |
| Efficient DiD, covariate adjusted | deterministic equation, refusal, and simulation evidence pass | unavailable: pinned public `edid` explicitly excludes covariates | unavailable: no identified Chen–Sant'Anna–Xie PT-All implementation | internal validation; external unavailable |
| `PartiallyLinearDML` residual stage | hand score plus independent Statsmodels HC1 pass | base R 4.5.1 matrix/HC1 contract passes | reviewed Stata/IC 17 no-intercept HC1 contract passes | pass |
| `RLearner` fixed honest evaluation | hand loss/calibration/group/max-t identities pass | base R 4.5.1 loss, HC1 calibration, and group covariance pass | reviewed Stata/IC 17 loss/calibration/group HC1 fixture passes | pass |

The DML parity fixture fixes already out-of-fold nuisance predictions and compares the
aligned residual-on-residual coefficient and HC1 standard error. It validates the public
orthogonal-score second stage, not another ecosystem's sample splitting or native
nuisance learner. `benchmarks/validate_dml_reference.R` passes under R 4.5.1. The manual
Stata harness is `benchmarks/validate_dml_stata.do`; its reviewed Stata/IC 17 output is
`benchmarks/validate_dml_stata_output.txt`. The estimate difference is
`2.220446049250313e-16` and the standard-error difference is
`5.551115123125783e-17`, both below the declared `1e-10` tolerance.
Those immutable comparator artifacts record `0.7.0a1`. Version `0.7.0a2` changes native
nuisance-grid selection and diagnostic transport, not the fixed-OOF residual-stage moment
used by the parity fixture, so the reviewed R/Stata second-stage evidence remains aligned.

The R-learner parity fixture conditions on immutable construction-fitted evaluation
predictions and compares honest R-loss, the constant comparator, differential calibration,
and tie-preserving group HC1 moments. `benchmarks/validate_rlearner_reference.R` passes
under base R 4.5.1. The manual Stata input is generated by
`benchmarks/prepare_rlearner_stata.py`; `benchmarks/validate_rlearner_stata.do` writes its
output before asserting. The reviewed output is
`benchmarks/validate_rlearner_stata_output.txt`; all loss, calibration, covariance, and
group fields pass at `1e-8`, with maximum absolute difference `1.55e-15`. Python separately
reconstructs the seeded max-t draws because cross-runtime random-number streams are not
silently treated as identical. The learner-specific native spline basis is validated by
hand identities and simulations; external parity conditions on fixed CATE predictions and
therefore validates the provider-neutral honest evaluation layer rather than claiming that
another package implements the same tuning algorithm.
The input SHA-256 is
`946a7a49d8fa5b1926030365ee03b60ff8d857992a1956c09e783e5755b20e2d`; the reviewed
Stata output SHA-256 is
`d16bd2b65dfcd69340f9327239f17f3b2f384811aff4bf591884c17e0441504c`.

The no-covariate efficient-DiD R harness is
`benchmarks/validate_edid_reference.R`; its maintained fixture compares every candidate
effect, the inverse-covariance weights, combined ATT, and HC1 standard error. This is not
evidence for the covariate-adjusted path or for Stata.

The fixed-score matching R harness is `benchmarks/validate_matching_reference.R`. It
pins CRAN `Matching` commit `1208eaa7bfa888b1fc903481dddfb8c0dffa40d5`
(version 4.10-15) and compares ATT, ATC, and ATE estimates plus Abadie-Imbens standard
errors on the no-tie, one-neighbor contract fixture. The maintained Stata script is
`benchmarks/validate_matching_stata.do`. A manual Windows run with Stata/MP 17 passed;
the reviewed output is `benchmarks/validate_matching_stata_17_output.txt`. The Stata ATC
mapping reverses treatment, estimates ATET, and negates the coefficient while retaining
its standard error. Stata requires at least two same-treatment neighbors in
`vce(robust, nn(#))`, so that harness compares CauseKit's otherwise identical
`variance_neighbors=2` contract; `nneighbor(1)` still governs the effect match.

The estimated-Logit Python fixture independently fits the treatment model with
`statsmodels.Logit`, passes that fitted result through the provider-neutral
`FittedPropensityMLEProtocol`, and compares all three effects and adjusted standard errors
with hand-recorded values. CRAN `Matching` accepts a supplied score but conditions on it
for uncertainty, so it is
not comparable to the fitted-score first-step correction. The maintained Stata harness is
`benchmarks/validate_matching_estimated_stata.do`; it uses `teffects psmatch`, one effect
neighbor, and `vce(robust, nn(2))`. Stata's two-observation local set includes the focal
observation, so its `nocorrection` variance maps to CauseKit `variance_neighbors=1`.
The first-step component maps to two-observation local covariance moments, two
leave-own-out outcome-regression neighbors, and one opposite-arm covariate neighbor. The
reviewed Stata/IC 17 output is
`benchmarks/validate_matching_estimated_stata_output.txt`. All component statuses pass;
the maximum absolute differences are `0` for estimates, `1.1102230246251565e-16` for the
known-score variance, `2.601257653722655e-9` for the first-step adjustment, and
`2.62713550913674e-9` for standard errors against a declared `1e-8` tolerance.

Both committed Stata script/output pairs were executed immediately before the CauseKit
rename. Their old `CausalKit`/`causalkit` labels and generator hashes are retained
unchanged as provenance; the executable Python parity tests now import `causekit` and
verify the same numerical contracts against those immutable artifacts.

The real-data certificate uses the four hash-pinned sources documented in
[Real-data validation](REAL_DATA_VALIDATION.md). `benchmarks/prepare_real_data.py`
materializes one shared CSV representation; `benchmarks/validate_real_data_reference.R`
then supplies independent base-R matrix/influence calculations, pinned CRAN `Matching`,
and pinned public `edid` comparisons. The R 4.5.1 run passed all 14 assertions. The
reviewed Stata/IC 17 artifact is
`benchmarks/validate_real_data_stata_output.txt`; all five comparable family statuses and
the overall status pass. Maximum absolute differences are `8.65e-7` for IV, `8.07e-7`
for randomized ATE, `5.64e-7` for supplied-nuisance IPW/AIPW, `7.39e-13` for matching,
and `1.11e-16` for conventional DiD.

## Completion rule

Before declaring all planned models release-complete, resolve every `pending` cell where
an estimand-aligned implementation exists. When none exists, retain evidence for the
ecosystem search and mark the cell `unavailable`; do not replace it with a different
estimator. Keep generated comparator outputs outside source control unless they are small,
reviewed, versioned fixtures produced by a committed harness.
