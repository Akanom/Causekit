# Validation strategy

`causekit` separates implementation checks from evidence claims. A command returning
success in one environment supports the tested version, fixture, specification, comparator,
and tolerance. It does not establish universal numerical parity or validate an empirical
instrument.

This document defines the validation gate and how to report it. It does **not**
assert that the commands below have passed for the current checkout. Release notes and
review records must state the actual execution environment and outcome.

For supplied-nuisance IPW/AIPW, maintained tests must include the exact score identity,
influence-function centering and variance identity, deterministic recovery under a known
data-generating process, propensity-bound refusal, clipping disclosure, exact index
alignment, cluster-sum covariance, and a large vectorized smoke path. Nuisance predictions
used in empirical validation must record whether they are oracle, fixed low-complexity,
held-out, or cross-fitted; in-sample adaptive predictions cannot be presented as validated
cross-fitted inference.

For nearest-neighbor matching, point-estimation evidence begins with hand-computed ATT,
ATC, and bidirectional ATE examples. It must also reconstruct fractional tie weights,
effect-weight identities, comparison reuse, inclusive caliper and support boundaries,
target-population relabeling, weighted balance, row-permutation invariance, and exact
paired heterogeneous-effect recovery. Refusal tests cover invalid scores, alignment,
empty matches, unsupported replacement/metrics/ties/bias correction, ordinary bootstrap,
clustered inference, and unsupported analytical-variance paths. Changing outcomes while
holding the design inputs fixed must not change selected matches.

The known-score Abadie-Imbens path has hand-reconstructed reuse-aware ATT, ATC, and ATE
variance identities; exact same-arm conditional-variance values; a seeded ATE coverage
smoke; and estimate/standard-error parity against CRAN `Matching` 4.10-15 commit
`1208eaa7bfa888b1fc903481dddfb8c0dffa40d5`. It refuses estimated/cross-fitted scores,
support/caliper selection, expanded design or variance ties, inadequate same-arm samples,
and clustered inference. Its Stata/MP 17 `teffects nnmatch` output is recorded.

The estimated-Logit path has hand-reconstructed ATT, ATC, and ATE adjustment vectors,
target derivatives, normalized information, variances, and standard errors. An independent
`statsmodels.Logit` fixture verifies provider-neutral fitted-result consumption without
re-estimation.
Refusals cover missing first-step structure, nonstationary/penalized fits, convergence,
sample/feature/parameter/prediction/index drift, singular information, wrong score status
or metric, and support/caliper selection. A fixed-seed 40-replication ATE smoke checks
recovery, standard-error scale, and interval coverage. This evidence does not extend to
generic or cross-fitted learners. The reviewed Stata/IC 17 `teffects psmatch` artifact
passes the ATT/ATC/ATE point, known-variance, first-step, and final-standard-error
components at `1e-8`; R `Matching` is non-comparable because it treats the supplied score
as fixed.

For partially linear DML, maintained tests reconstruct the DML2 coefficient, orthogonal
score, influence function, residual-treatment Jacobian, HC1 covariance, cluster-summed CR1
covariance, and reference distribution. They also verify shared binary-stratified folds,
native fold-local ridge-GCV defaults, deterministic continuous-treatment recovery,
alignment/factory/fold/cluster refusals, and numerical residual-treatment identification.
A supplied dense-grid test proves that nesting the default candidates weakly lowers every
fold's selected GCV and improves at least one deterministic fold strictly. The dense grid
is not the default because the one-run real-data OOF errors did not improve. CrossFitter
contract tests reconstruct provider-neutral diagnostic rows and refuse malformed mappings.
Statsmodels and base R independently reproduce the fixed residual-stage coefficient and
HC1 standard error. The manually executed Stata/IC 17 harness writes its result before
asserting; the reviewed saved output passes both estimate and standard-error assertions at
the declared `1e-10` tolerance.

For the honest R-learner, hand contracts reconstruct immutable row/cluster roles, shared
cluster-preserving outer folds, `u/v` and `v^2`, the direct/weighted R-objective identity,
held-out R-loss, the construction-fitted constant comparator, differential calibration,
HC1/CR1 covariance, tie-preserving group moments, the full influence matrix, and the
seeded max-t critical value. Leakage logs prove that evaluation indices never reach a fit
target. Seeded simulations cover native linear and piecewise-nonlinear CATE recovery plus
null/power and complete-path band-coverage smokes. Base R 4.5.1 independently reproduces
fixed evaluation loss, calibration, and group covariance. The reviewed Stata/IC 17 do-file
reconstructs the same HC1 moments and passes at `1e-8`; its maximum absolute difference is
`1.56e-15`. The hash-pinned NSW benchmark retains the negative finding that no comparator
showed significant differential calibration. The separate Hillstrom randomized-email
record compares native linear and adaptive spline stages on one identical honest split.

For DiD, maintained evidence must keep the conventional and efficient estimators
separate. Conventional tests hand-compute every `ATT(g,t)`, event-time, calendar-time, and
ESavg aggregation for both never-treated and not-yet-treated comparisons. Efficient tests
must reconstruct every generated-outcome candidate, verify weights sum to one, reproduce
the inverse-covariance solution, reconstruct the efficient influence function, and refuse
singular systems without hidden regularization. Anticipation, control contamination,
balanced-panel validation, entity-level HC scaling, cluster-summed covariance, and random
cohort-share influence terms require direct tests.

Panel diagnostic tests hand-compute adjacent uncontaminated pre-period placebos, verify
that anticipation-window leads are excluded, reconstruct the full robust and cluster-
summed joint covariance, and retain individual placebos when a singular joint test is
unavailable. The PT-All/PT-Post Hausman contract reconstructs the common post-treatment
event-study difference influence function and covariance, hashes the complete estimation
design, and refuses not-yet-treated, covariate-adjusted, mismatched, reduced-moment, and
singular comparisons without pseudoinverse rank selection.

The covariate path additionally requires exact shared-fold alignment across multiclass
cohort probabilities, masked outcome-change tasks, and residual-product second-moment
tasks; a multi-moment fixture must exercise observation-specific conditional covariance
inversion. Tests must prove that low cohort probabilities and singular conditional
systems refuse without clipping or repair. Simultaneous-band tests reconstruct the seeded
Rademacher max-t critical value from entity scores and from cluster-summed scores. A fixed-
seed Monte Carlo smoke checks bias and empirical coverage without being presented as a
publication-scale coverage study.

The recorded efficient reference is `david-loeb/edid` commit
`f55a4a4aba14f0826f59ad7aa4af3bafaeba529b`. On the eight-entity orthogonal-score fixture,
the public R implementation returns candidate effects `(10, 10, 10)`, weights
`(16/21, 4/21, 1/21)`, efficient ATT `10`, and HC1 standard error
`0.46656947481584343`. `benchmarks/validate_edid_reference.R` reproduces the reference
values, and the optional validation test compares them with the native result.

## Claim boundary

Validation can provide evidence that:

- the estimator implements the declared 2SLS matrix problem;
- coefficient and covariance calculations agree with analytical identities or an aligned
  independent implementation;
- diagnostics use their stated definitions and reference distributions;
- strict input, alignment, schema, and refusal contracts behave as documented; and
- results remain internally aligned through prediction and reporting.

Validation cannot establish that an empirical instrument is independent, satisfies the
exclusion restriction, has no interference pathway, or identifies the analyst's preferred
causal estimand. Those claims require research-design evidence outside the software.

## Evidence layers

### 1. Contract and refusal tests

Fast tests should cover the complete public surface:

- accepted constructor and fit arguments;
- exact covariance labels and required cluster input;
- parameter order: constant, exogenous, endogenous;
- `params`, `covariance`, `standard_errors`, `test_statistics`, `pvalues`, `residuals`,
  `fitted_values`, `nobs`, `df_resid`, and covariance metadata;
- confidence intervals, summary frames, Markdown output, and prediction schema;
- missing and non-finite values in every numeric role;
- mismatched and permuted pandas indices, plus consistent repeated-index sequences;
- duplicate or changed column names and wrong prediction column order;
- inconsistent lengths and dimensionality;
- exact collinearity in structural and instrument designs;
- underidentification, exact rank failure, and too few residual degrees of freedom; and
- missing, misaligned, degenerate, or insufficient cluster labels; and
- joint complete-case filtering, retained-index preservation, and `dropped_rows` metadata
  under `missing="drop"`.

Refusal behavior is part of numerical correctness. A test that silently drops a bad row is
not equivalent to a test that confirms `missing="raise"`; the explicit `missing="drop"`
path must prove that every input used the same mask.

### 2. Analytical identities

Small deterministic fixtures should compare public results with independently assembled
matrix identities:

```text
theta_hat = (S' P_Q S)^(-1) S' P_Q y
fitted     = S theta_hat
residual   = y - fitted
```

The gate should separately reconstruct:

- homoskedastic covariance from the structural residual variance;
- HC1 sandwich covariance with multiplier `n / (n - k)`;
- one-way CR1 covariance from cluster-summed scores with multiplier
  `[G / (G - 1)] * [(n - 1) / (n - k)]`;
- coefficient statistics and confidence intervals using the recorded t or normal
  reference distribution;
- partial R-squared from residualized endogenous variables and instruments;
- classical first-stage excluded-instrument F tests;
- covariance-aware first-stage joint tests; and
- Sargan's `n R-squared` identity and `q - m` degrees of freedom when its homoskedastic
  conditions apply.

Tests should also verify covariance symmetry, finite diagonals, label alignment, the
identity `fitted_values + residuals == y` within numerical tolerance, and invariance to
consistent row permutation and cluster-label renaming.

### 3. Deterministic simulation

Simulations should exercise known data-generating processes rather than assert exact
sample recovery. Required designs include:

- a strong, valid, just-identified instrument;
- an overidentified model with multiple excluded instruments;
- multiple endogenous regressors with distinct relevant variation;
- heteroskedastic structural errors for HC1 inference;
- within-cluster dependence for CR1 inference;
- weak or nearly irrelevant excluded instruments; and
- deliberately invalid or collinear designs that must be rejected or warned about.

Use fixed random seeds. Recovery thresholds must be chosen before observing a convenient
draw and justified from the sample size, signal strength, and conditioning. A simulation
can show behavior under its known construction; it does not validate instruments in
uncontrolled data.

### 4. Independent implementation comparisons

The optional `validation` dependencies support aligned comparisons with maintained Python
implementations. `linearmodels` is the primary external 2SLS comparator because its IV
interface also distinguishes exogenous regressors from excluded instruments. `statsmodels`
can provide supporting regression and diagnostic identities where definitions align.

Every comparison must record:

- comparator package and version;
- data generator or data provenance and random seed;
- outcome, intercept, exogenous, endogenous, and excluded-instrument mapping;
- covariance definition and finite-sample/debiasing option;
- cluster definition and cluster count, when applicable;
- parameter and matrix label mapping;
- fields compared and numeric tolerances; and
- observed maximum absolute and relative differences.

Comparator defaults often differ. In particular, intercept handling, HC versus debiased
scaling, cluster corrections, degrees of freedom, and p-value reference distributions must
be aligned explicitly. Agreement after an undocumented option change is not reproducible
evidence.

## Required model matrix

The following matrix defines the intended release evidence. A row may be described as
passing only after its tests have executed successfully in the recorded environment.

| Area | Required cases | Required assertions |
| --- | --- | --- |
| Point estimates | Just identified, overidentified, multiple endogenous regressors | Estimates, labels, ordering, fitted values, residuals |
| Unadjusted inference | Homoskedastic fixture | Covariance, t statistics, p-values, intervals, `n-k` degrees of freedom |
| HC1 inference | Heteroskedastic fixture | Sandwich meat, `n/(n-k)` multiplier, normal reference |
| One-way CR1 | Unequal cluster sizes and within-cluster dependence | Cluster scores, CR1 multiplier, `G-1` reference degrees, cluster metadata |
| First stages | Strong, weak, multiple-instrument, multiple-endogenous cases | R-squared, partial R-squared, classical F, covariance-aware joint test and metadata |
| Overidentification | Just identified, overidentified, every covariance label | Sargan only for overidentified unadjusted fit; correct statistic, p-value, and `q-m` df |
| Data contract | NumPy and pandas, missing/non-finite, misaligned indices, duplicate names | Stable success behavior or precise refusal; never silent sample drift |
| Post-estimation | In-sample and new-data prediction, tables, Markdown | Schema enforcement, index preservation, numerical alignment |
| Integrations | Optional dependency present and absent | Stable adapter output or actionable optional-dependency error |
| Matching point estimate | ATT, ATC, ATE, ties, support/caliper attrition, reuse | Hand identities, design invariance, target labels, exact audit weights |
| Matching uncertainty | Fixed score and supported estimated-score provenance | Abadie–Imbens variance identities, first-step adjustment, coverage, explicit refusals |
| Matching performance | Balanced/imbalanced arms, ties, attrition, reuse at scale | Sorted-scalar behavior, elapsed time, peak memory, no quadratic distance matrix |
| Conventional DiD | Single/staggered cohorts, never/not-yet controls, anticipation | Hand `ATT(g,t)`, event/calendar/ESavg targets, uncontaminated controls, influence identities |
| Efficient DiD | Multiple pre-periods and auxiliary cohorts under PT-All | Candidate effects, inverse-covariance weights, efficient influence, singular refusal, R parity |
| Covariate-efficient DiD | Multiple moments, valid/invalid overlap and covariance systems | OOF alignment, equation (4.4) scores, equation (3.12) weights, exact refusal boundaries |
| DiD inference | Entity and higher-level clustered sampling | HC1/cluster score identities, pointwise metadata, robust/cluster max-t band identities |
| DiD diagnostics | Clean/no-clean/singular pre-period paths; aligned/misaligned PT-All and PT-Post | Placebo influence/covariance identities, anticipation exclusion, strict Hausman difference-IF/refusal contract |
| Partially linear DML | Binary/continuous treatment, native/custom nuisance, robust/clustered inference | DML2 score, OOF fold alignment, native ridge-GCV, fold tuning audit, direct GCV identity, influence/Jacobian identities, strict weak-signal refusal |
| Honest R-learner | Binary treatment, row/cluster honesty, native/custom learners, overlap, robust/clustered calibration | Leakage/refusals, exact R-objective, held-out loss/constant gain, HC1/CR1 differential calibration, tie-preserving group moments/influence/max-t bands, simulations, R/Stata fixed-evaluation parity, real-data comparator record |
| Causal-ML performance | One hash-verified real dataset, one identically folded run per model | Estimate, standard error, OOF outcome/treatment RMSE, elapsed time, Python peak memory, versions, no runtime comparator dependency |

## Cross-software parity matrix

Every model family must ultimately record estimand-aligned comparisons in Python, R, and
Stata. A parity row records the exact software/package version, fixture provenance,
estimation options, sample and parameter mapping, covariance corrections, seed,
tolerances, and observed discrepancy. “Unavailable” and “non-comparable” are valid states
when a platform lacks the estimator or implements different identifying moments; neither
state is a pass.

The maintained no-covariate efficient DiD row pins the public R `edid` implementation.
The published `edid` reference has no covariate-adjusted estimator, and the reviewed
Stata estimators target different identifying moments; those covariate-efficient cells
are therefore recorded as unavailable rather than fabricated parity. Adding an arbitrary
regression that happens to return a similar number does not satisfy this gate.

The live status and completion rule are maintained in [Cross-software parity
register](PARITY.md).

## Numerical tolerances

Do not use one global tolerance. Exact integer metadata and labels require equality.
Closed-form values on well-conditioned synthetic designs should use tight floating-point
tolerances. Reference-package comparisons may need a documented wider tolerance where
linear solvers, finite-sample conventions, or tail-probability implementations differ.

For every tolerance, record:

- whether it is absolute, relative, or both;
- the scale and conditioning of the fixture;
- the expected source of numerical variation; and
- the maximum observed discrepancy in the recorded run.

Never loosen a tolerance merely to absorb an unexplained regression. Diagnose matrix
conditioning, sample alignment, parameter order, and comparator conventions first.

## Reproduction commands

Install the development environment from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Run the complete project gate:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
python -m build
python -m twine check --strict dist/*
```

Run marked evidence subsets explicitly when reviewing them:

```bash
python -m pytest -m validation
python -m pytest -m simulation
python benchmarks/benchmark_matching.py --scenario balanced_ate --n 100000 --measure-memory
python benchmarks/benchmark_matching.py --scenario known_score_ate_inference --n 100000 --measure-memory
python benchmarks/benchmark_matching.py --scenario estimated_score_ate_inference --n 100000 --measure-memory
python benchmarks/benchmark_did.py --scenario all --n-entities 20000
python benchmarks/benchmark_ml.py --models all
python benchmarks/benchmark_ml.py --dataset nsw_mixtape --models all
python benchmarks/benchmark_rlearner.py --models all
python benchmarks/benchmark_rlearner_hillstrom.py --data /path/to/reviewed/hillstrom.csv
Rscript benchmarks/validate_dml_reference.R
```

The ML/R-learner commands are separate one-run real-data records, not benchmark
repetitions. Their frozen results and interpretation limits are documented in
[Causal-ML real-data performance](ML_BENCHMARK.md),
[NSW honest R-learning](R_LEARNER_BENCHMARK.md), and
[Hillstrom honest R-learning](R_LEARNER_HILLSTROM_BENCHMARK.md).

Stata is manual: from the repository root run
`do "benchmarks/validate_dml_stata.do"`. The harness persists
`benchmarks/validate_dml_stata_output.txt` before any parity assertion.

Prepare and run the pinned real-data certificate against the exact R reference checkouts:

```bash
python benchmarks/prepare_real_data.py --download
CAUSEKIT_REAL_DATA_DIR=/path/to/prepared/data \
CAUSEKIT_EDID_REFERENCE=/path/to/edid \
CAUSEKIT_MATCHING_REFERENCE=/path/to/Matching \
python -m pytest tests/validation/test_real_data_parity.py
```

Stata is a manual final gate. From the repository root, run
`do "benchmarks/validate_real_data_stata.do"`; it writes a machine-readable result before
asserting. Dataset provenance, comparator boundaries, and exact commands are recorded in
[Real-data validation](REAL_DATA_VALIDATION.md).

Run pinned R `edid` parity from a checkout at the recorded commit:

```bash
CAUSEKIT_EDID_REFERENCE=/path/to/edid python -m pytest tests/validation/test_edid_parity.py
Rscript benchmarks/validate_edid_reference.R /path/to/edid
```

Run pinned R `Matching` parity from the CRAN mirror checkout at tag 4.10-15, then
run the Stata script manually when Stata is available:

```bash
CAUSEKIT_MATCHING_REFERENCE=/path/to/Matching python -m pytest tests/validation/test_matching_parity.py
Rscript benchmarks/validate_matching_reference.R /path/to/Matching
stata -b do benchmarks/validate_matching_stata.do
stata -b do benchmarks/validate_matching_estimated_stata.do
```

The Stata harness uses one opposite-arm effect match and two same-arm variance
neighbors. This is the smallest robust-variance contract accepted by Stata and maps to
`NearestNeighborMatch(..., variance_neighbors=2)`; the R harness separately validates
CauseKit's supported one-neighbor conditional-variance contract. The reviewed
Stata/MP 17 result is preserved in
`benchmarks/validate_matching_stata_17_output.txt`; its maximum absolute standard-error
difference from CauseKit is `4.440892098500626e-16`.

The estimated-score Stata harness maps raw fitted propensity distance and one effect
match. Stata's `vce(robust, nn(2))` local set includes the focal observation: its
`nocorrection` variance maps to CauseKit `variance_neighbors=1`, while the first-step
moments use two-observation local covariances, two leave-own-out outcome-regression
neighbors, and one opposite-arm covariate neighbor. The reviewed Stata/IC 17 artifact is
`benchmarks/validate_matching_estimated_stata_output.txt`; all point-estimate,
known-variance, first-step-adjustment, and final-standard-error statuses pass at the
declared `1e-8` tolerance. Its maximum absolute standard-error difference is
`2.62713550913674e-9`.

Run the README example in a clean installation and inspect both wheel and source
distribution before release. Archive the commands, operating system, Python version,
resolved dependency versions, commit identifier, test output, and artifact hashes with a
release decision.

## Reporting validation status

Use bounded statements:

- “matched `linearmodels` for the recorded specification and covariance options within the
  declared tolerances”;
- “recovered the generating coefficient across the recorded deterministic simulations”;
- “the strict index-mismatch refusal test passed”; or
- “not run in this environment; exact command: ...”.

Avoid statements such as “fully validated,” “proven correct,” “identical to all software,”
or “the instruments are valid.” If a required gate is unavailable, distinguish an
environment limitation from a code failure and list the remaining definition-of-done gap.

## Adding evidence

New evidence belongs in maintained tests or a documented validation harness, not only in a
notebook or screenshot. Generated artifacts must record their script, source data or
generator, configuration, comparator versions, mappings, tolerances, date, and reproduction
command. Do not commit restricted data, local paths, or unreviewed binary output.
