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
| `PanelIV2SLS` | explicit-dummy `linearmodels` 7.0 passes 12 balanced/unbalanced, one-/two-way, and covariance cells; hash-pinned wage-panel coefficient/covariance differences below `7.73e-13` | R 4.5.1 `AER` 1.2.16 plus `sandwich` 3.1.1 explicit-dummy CR1 passes six fields within `3.90e-13` | reviewed Stata/IC 17 `ivregress 2sls` explicit-dummy CR1 passes six fields within `1.22e-12` | pass |
| `RandomizedATE` | `statsmodels` regression/covariance identities | real NSW raw/Lin HC1 contracts pass | Stata/IC 17 real-data native robust regressions pass | pass |
| `IPWATE` / `AIPWATE` ATE/ATT/ATC | analytical score identities pass | real Cattaneo supplied-nuisance influence contracts pass | Stata/IC 17 same conditional influence contracts pass; `teffects` first-step variance is non-comparable | pass for aligned conditional-nuisance contract |
| `CrossFitter` | protocol and leakage/alignment tests; not itself an estimand | non-comparable | non-comparable | internal protocol |
| `NearestNeighborMatch` fixed-score ATT/ATC/ATE | hand/reuse/variance identities pass | CRAN `Matching` 4.10-15 fixture and real Cattaneo point estimates pass | Stata/MP 17 fixture and Stata/IC 17 real Cattaneo point harness pass | pass |
| `NearestNeighborMatch` estimated-Logit ATT/ATC/ATE | hand formulas and independent `statsmodels.Logit` pass | `Matching` conditions on supplied scores: non-comparable | Stata/IC 17 `teffects psmatch` fixture passes | pass for available estimand-aligned comparators |
| Conventional staggered DiD | hand/influence identities pass | real hospital group-time/influence contract passes | Stata/IC 17 same group-time/influence contract passes; `didregress` common-effect aggregation is non-comparable | pass for aligned contract |
| Repeated-cross-section DiD, stationary composition | hand, aggregation, HC1/CR1, and 32-cell publication coverage contracts pass | pinned `did` 2.5.0 `att_gt(panel = FALSE, est_method = "reg")` passes both control rules after explicit HC0-to-HC1 mapping | reviewed Stata/IC 17 `csdid` matches group-time estimates/SEs and aggregate points; aggregate SEs use non-comparable cell-share influence | pass for available aligned fields |
| Repeated-cross-section DiD, covariate adjusted | hand score/influence, leakage, double-robustness, hash-verified real-data, and 44-cell publication coverage contracts pass | fixed-OOF base-R score/HC1 reconstruction passes; estimator-level implementation unavailable | reviewed estimators target different moments: unavailable | internal publication evidence and score parity pass; estimator-level external unavailable |
| Repeated-cross-section DiD, composition robust | pairwise and longer hand influence/target-share/band identities, hash-pinned real data, 120,000-row performance, and observation/PSU pointwise/joint/diagnostic promotion pass | official `compdid` 0.1.0 pairwise point/IF and fixed-influence stationarity diagnostic pass; aligned longer estimator unavailable | no reviewed estimator targeting the same treated-target-period influence moment: unavailable | pairwise and longer/staggered internal promotion pass; survey combinations remain open |
| Efficient DiD, no covariates | native result checked on fixture and real data | pinned public `edid` commit passes fixture and real hospital data | unavailable: Stata heterogeneous DiD does not implement PT-All optimal weighting | pass for available aligned comparator |
| Efficient DiD, covariate adjusted | deterministic equation, refusal, and simulation evidence pass | unavailable: pinned public `edid` explicitly excludes covariates | unavailable: no identified Chen–Sant'Anna–Xie PT-All implementation | internal validation; external unavailable |
| Efficient DiD, direct cohort odds | hand score/`Omega_tilde`, nonlinear/weak-overlap publication, hash-pinned hospital, and 100,000-entity performance gates pass | base R 4.5.1 independently reproduces the fixed-OOF ordered-odds score, full influence, and HC1 SE; maintained estimator-level implementation unavailable | unavailable: no reviewed command accepts the same direct pairwise odds nuisances and PT-All moment | internal promotion and independent score parity pass; estimator-level external unavailable |
| `PartiallyLinearDML` residual stage | hand score plus independent Statsmodels HC1 pass | base R 4.5.1 matrix/HC1 contract passes | reviewed Stata/IC 17 no-intercept HC1 contract passes | pass |
| `RLearner` fixed honest evaluation | hand loss/calibration/group/max-t identities pass | base R 4.5.1 loss, HC1 calibration, and group covariance pass | reviewed Stata/IC 17 loss/calibration/group HC1 fixture passes | pass |
| `DRLearner` fixed honest evaluation | hand pseudo-outcome/loss/calibration/group/max-t identities pass | base R 4.5.1 loss, HC1 calibration, and group covariance pass | reviewed Stata/IC 17 loss/calibration/group HC1 fixture passes | pass |
| Sharp/fuzzy fixed-bandwidth RD | CauseKit hand/refusal identities and Python `rdrobust` 2.0.0 conventional point/RBC/robust-HC1 parity pass | R `rdrobust` 4.0.0 passes conventional point, RBC, robust-HC1, and fuzzy corrected first-stage fields within `2e-14` | reviewed Stata/IC 17 `rdrobust` 11.1.0 passes all aligned fields within `2.31e-14` | pass |

The RD parity fixture fixes triangular `p=1`, `q=2`, left/right point bandwidths
`(1.2, 1.4)`, bias bandwidths `(1.6, 1.7)`, HC1, and 800 deterministic observations.
`benchmarks/validate_rd_reference.R` writes the passing R artifact before assertions;
`benchmarks/validate_rd_stata.do` follows the same rule. Its reviewed output SHA-256 is
`29591cdf4a4ea3c4269838aafb02fec7942df8b870774c3aa221e04fbf429a0d`.
Native bandwidth selection is CauseKit-specific and is validated by recovery/coverage,
not represented as parity with `rdrobust` bandwidth selectors.

The Panel IV parity contract uses the hash-pinned Vella-Verbeek wage panel, 545 person
effects, seven retained year effects, person-clustered CR1, and full absorbed-rank finite-
sample correction. CauseKit absorbs effects compactly; Python, R, and Stata references use
explicit indicators. This validates numerical equivalence only. Lagged union status is
not asserted to satisfy exclusion or exogeneity. See
[Panel IV promotion evidence](PANEL_IV_PROMOTION_EVIDENCE.md). The reviewed Stata output
SHA-256 is `1ac883346889ff0d9c651864b8fdf17570cb663d392fc56e883d832961f70704`.

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

The DR-learner parity fixture likewise conditions on immutable construction-fitted
evaluation nuisance and CATE predictions. Python reconstructs the augmented
inverse-probability score, losses, HC1 calibration, group covariance, influence values,
and seeded max-t draws. `benchmarks/validate_drlearner_reference.R` passes under base R
4.5.1. `benchmarks/prepare_drlearner_stata.py` reproduces the fixed input and
`benchmarks/validate_drlearner_stata.do` writes diagnostic output before assertion. The
reviewed Stata/IC 17 artifact passes all fields at `1e-8`; the maximum absolute difference
is `4.440892098500626e-16`. The input SHA-256 is
`dd903ee5cfd916e0be5fabf234565ad8daae1a957f8c38ccb95e1e07d9c78e78`; the output SHA-256
is `fd70e45a700c25839602b787967c3b3634b35141942d01a21b7d69fa458598eb`.

The no-covariate efficient-DiD R harness is
`benchmarks/validate_edid_reference.R`; its maintained fixture compares every candidate
effect, the inverse-covariance weights, combined ATT, and HC1 standard error. This is not
evidence for the covariate-adjusted path or for Stata.

The direct-ratio score harness is
`benchmarks/validate_did_direct_ratio_reference.R`. Base R 4.5.1 independently rebuilds
the fixed-fold calibrated posterior odds, candidate score, every influence coordinate,
and HC1 standard error. This validates the scale-sensitive ordered-odds PT-All score, not
another ecosystem's nuisance learner. The reviewed R/Stata surfaces expose no maintained
estimator accepting the same pairwise odds nuisances and PT-All moment, so estimator-level
cells remain unavailable rather than being approximated with a different DiD command.

The repeated-cross-section hand harness is
`benchmarks/validate_did_rcs_reference.R`; base R 4.5.1 reproduces the 2-by-2 estimate,
all eight observation influence values, and HC1 standard error at `1e-12`. The manual
`benchmarks/validate_did_rcs_stata.do` writes its result before asserting. Its reviewed
Stata 17 output passes the estimate and HC1 standard error at `1e-12`; artifact SHA-256 is
`67cd686b409379a7dbcc58b8172d1defa6a132bb716458dfd0b0217d47288d95`.
The estimator-level R harness is `benchmarks/validate_did_rcs_did_reference.R`. It pins
`did` 2.5.0 commit `c449b8ce72029855d2de94b377f131be0e53e53a`, requests
`att_gt(panel = FALSE, est_method = "reg")`, and exercises never-treated and not-yet-
treated comparisons. CauseKit and R agree on all group-time, event, calendar, and ES-
average point estimates. R's analytical standard errors agree after multiplication by
`sqrt(54/53)`, the explicit mapping to CauseKit observation HC1. The saved reference is
`benchmarks/validate_did_rcs_did_output.txt`. The estimator-level Stata harness is
`benchmarks/validate_did_rcs_csdid.do`. Its reviewed Stata/IC 17 artifact exactly matches
all 16 point estimates and all six group-time analytical standard errors; maximum aligned
difference is zero. Aggregate SEs are non-comparable because `csdid` propagates period-
specific cell-share influence while CauseKit/R use pooled cohort-share influence. The
saved output SHA-256 is
`ad250fa9cdfd042e1989460ebcec6b1bac57ed301fae390f2c69331ccf47fd57`.

The covariate repeated-cross-section fixed-OOF harness is
`benchmarks/validate_did_rcs_covariate_reference.R`. Independent base R reconstructs all
eight normalized Sant'Anna-Zhao locally efficient score components, the ratio influence
terms, ATT, and CauseKit HC1 mapping at `1e-12`. This is score-level evidence under fixed
OOF nuisance predictions, not estimator-level equivalence to a particular classifier or
regressor. The audited pinned `did` path does not offer this covariate score, and reviewed
Stata repeated-cross-section estimators target different maintained moments, so those
estimator-level cells remain explicitly unavailable rather than labelled passing.
The separate hash-bound Python certificate
`benchmarks/did_rcs_covariate_promotion_evidence.json` records 4,000 estimator fits,
240,000 fold-local nuisance fits, zero refusals, 44 passing pointwise coverage cells, and
four passing joint conditional-pre-trend size cells. It strengthens internal inferential
evidence without manufacturing estimator-level parity where no aligned external surface
exists.
The observation/PSU simultaneous layer is likewise validated against direct retained-
influence reconstructions in Python. R/Stata pseudo-random streams or resampling commands
are not relabelled as exact parity for CauseKit's seeded Rademacher sequence. The separate
hash-bound `benchmarks/did_rcs_simultaneous_promotion_evidence.json` certificate records
16,000 estimator fits, 480,000 fold-local nuisance fits, 15,984,000 fixed-seed max-t
draws, zero refusals, and 16 passing publication-scale joint-coverage cells. This is
internal inferential evidence, not manufactured cross-language random-number parity.

The composition-robust row begins with the independent pairwise Python hand contract in
`tests/test_did_rcs_composition.py` with
`benchmarks/validate_did_rcs_compdid_reference.R`. The official comparator pins
`compdid` 0.1.0 commit `894bd65a952c30f01a4e0005efba4cb335065eb7` and the relevant
source blobs, then executes its exported `drdid_nonstationary()` implementation directly
from that checkout. It explicitly reorders CauseKit probabilities `(00,01,10,11)` to R
`(11,10,01,00)` and supplies R's required `m11`, which algebraically cancels from the
point score and influence. ATT is `2.2950387445451073`, SE is
`0.29206804206766818`, and every one of 16 ordered influence coordinates agrees within
`2e-14`. The canonical-LF fixture SHA-256 is
`eafc591363edf693b42adea3f56782def6addbb7bed67dc993260b1b2b17bf2f`; the saved R
output canonical-LF SHA-256 is
`680802078b1634cb5f5e43096f94867ea9ed1b4f9c95fa384d9c30b51fc057a0`.
No Stata command is labelled comparable without evidence that it targets the same treated
target-period population and influence moment.
The added composition-shift fixture recovers robust target ATT `5` and the distinct
stationary pooled-treated target `4`; it is a target-definition test rather than an
external parity claim. Exact class-schema refusals and row/labelled-column permutation
invariance pass before the official comparator is attempted.
The independent pairwise promotion certificate adds 8,000 fits, 72,000 genuinely
fold-fitted nuisance tasks, zero refusals, and eight passing observation/PSU coverage
cells. The associated Sequeira run is a hash-pinned sensitivity comparison under fixed
ridge nuisances, not external estimator parity; both targets are reported without
pretest selection. Details and reproduction commands are in
`docs/DID_RCS_COMPOSITION_PROMOTION_EVIDENCE.md`.

The longer-design extension preserves the pairwise score in a shared-fold task lattice,
zero-pads every pair influence on the complete sample, and aggregates with target-period
treated-cell shares plus their estimated-share influence. Official R 4.5.1 diagnostic
mapping passes on fixed aligned estimates/influences. The 322-row hospital sensitivity,
120,000-row performance certificate, and four-cell 4,000-fit publication certificate pass
conditional-pretrend size, pointwise/joint coverage, and diagnostic size/power with zero
refusals. Official `compdid` has no aligned longer/staggered estimator interface; R/Stata
estimator-level cells remain unavailable. Details are in
`docs/DID_RCS_COMPOSITION_LONGER_PROMOTION_EVIDENCE.md`.

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

Version `0.7.0a4` adds validation depth without changing those comparator moments. The
saved `benchmarks/matching_promotion_evidence.json` certificate exercises ATT, ATC, and
ATE for both maintained inference contracts across 1,000 replications in each of two
overlap designs. All 12 cells pass; coverage is `0.943–0.955`, maximum absolute bias is
`0.0073`, and the analytical-SE/empirical-SD ratio is `0.963–1.019`. Its separate
hash-verified Cattaneo sensitivity grid is point-estimation-only because support/caliper
selection can change the target. Settled R/Stata fixtures were not rerun for this
simulation-only promotion.

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
