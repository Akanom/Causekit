# Repeated-cross-section DiD first-slice validation

## Contract evidence

The public estimator was implemented test-first. `tests/test_did_rcs.py` initially failed
during collection because `RepeatedCrossSectionDiD` and its result type did not exist.
The retained two-period fixture has treated means 2 and 7 and comparison means 3 and 4,
so its ATT is `(7 - 2) - (4 - 3) = 4`. Its ordered observation influence vector is
`[4, -4, -4, 4, -4, 4, 4, -4]`; HC1 therefore uses
`sum(psi^2) / (8 * 7)` and gives standard error `1.5118578920369088`.

The focused contracts also reconstruct staggered never-treated and not-yet-treated
effects, pooled estimated-share contributions, event/calendar/ESavg aggregation,
anticipation, independent-cell placebos, unequal sample sizes, row permutation, and
cluster-summed CR1 variance. Malformed roles, unsupported composition values and
inference, missing clean baselines, weak cells, invalid PSUs, covariates without an
explicit `CrossFitter`, and sampling weights refuse without row deletion or numerical
repair.

The covariate contract in `tests/test_did_rcs_covariate.py` was also observed failing
before implementation. Its fixed-OOF fixture has ATT `2.4875`, influence
`[-.975, -.575, .225, 1.025, -.775, -.175, .425, .825, 0, ..., 0]`, and HC1 standard
error `0.12706625568314087`. It reconstructs all eight normalized score components and
ratio influence terms. Separate contracts verify both double-robustness legs, shared
folds, row/PSU non-leakage, hard overlap refusal, conditional placebos, staggered
never/not-yet-treated aggregation, fixed-seed reproduction, and OutputHub transport.

The pairwise composition-change contract in `tests/test_did_rcs_composition.py` was
observed with four failures at the prior `composition="robust"` refusal before the runtime
path was added. Its 16-row fixture has estimate `2.295038744545108` and independently
reconstructs the complete centered EIF, HC1 standard error, and all four normalized cell
weights. The retained refusals cover missing covariates, more than two periods, more than
one treated cohort, and a generalized-propensity cell at the hard probability floor.
Separate audit assertions keep construction/holdout rows and declared PSUs disjoint,
require identical folds across the four-class and three outcome tasks, and verify that no
unused target-cell outcome regression is fitted. This evidence promotes only the pairwise
score; it is not staggered, real-data, or publication-scale evidence. External score
parity is promoted separately below.

The next 32-row deterministic design changes treated `X` composition from `0.25` at
baseline to `0.75` in the target period and sets the conditional effect to `2+4X`.
CauseKit's robust score recovers the target-period ATT `5`; the stationary score recovers
its different pooled-treated target `4`, so using it as the post-period target creates a
known bias of `-1`. The class-schema suite was observed with two intended failures before
protocol hardening: extra probability columns were ignored and duplicate labels reached
a generic matrix-shape error. Missing, extra, duplicate, and unlabeled probability
schemas now refuse explicitly; labelled column permutations realign safely. A complete
row permutation leaves the estimate, standard error, group-time result, weights,
influence, nuisance predictions, and design fingerprint unchanged.

## Cross-language hand reference

Run the independent base-R reconstruction:

```bash
Rscript benchmarks/validate_did_rcs_reference.R
```

The reviewed local R 4.5.1 run returned estimate `4`, HC1 standard error
`1.5118578920369088`, the exact eight-element influence vector above, and
`parity_status=pass`. The separate estimator-level comparator pins R `did` 2.5.0 commit
`c449b8ce72029855d2de94b377f131be0e53e53a`, expands the deterministic cells only to clear
that package's five-observation-per-group guard, and calls
`att_gt(panel = FALSE, est_method = "reg")` for both control rules. Every group-time,
event, calendar, and ES-average point estimate agrees. Multiplying the R analytical
standard errors by `sqrt(54/53)` maps its convention exactly to CauseKit observation HC1;
the saved artifact and live comparator tests pass.

Run the independent fixed-OOF covariate-score reconstruction:

```bash
Rscript benchmarks/validate_did_rcs_covariate_reference.R
```

Base R reconstructs the eight-component locally efficient repeated-cross-section score,
ATT, full influence vector, and HC1 at `1e-12`. It validates the causal score with fixed
out-of-fold nuisance predictions, not any particular nuisance-model implementation.

Run the official composition-change comparator from a checkout fixed at commit
`894bd65a952c30f01a4e0005efba4cb335065eb7`:

```bash
Rscript benchmarks/validate_did_rcs_compdid_reference.R PATH_TO_COMPDID_CHECKOUT
```

The reviewed R 4.5.1 run executes the pinned official `compdid` 0.1.0
`drdid_nonstationary()` R source. It matches CauseKit ATT, the sample-SD/HC1 standard
error, and all 16 ordered influence values at `2e-14`. The saved output is
`benchmarks/validate_did_rcs_compdid_output.txt`; the test also fixes the nuisance-column
mapping and canonical-LF SHA-256 hashes for both input and output artifacts.

Stata must be run manually from the repository root:

```stata
do "benchmarks/validate_did_rcs_stata.do"
```

The do-file writes `benchmarks/validate_did_rcs_stata_output.txt` before asserting. The
reviewed Stata 17 artifact returned estimate `4`, HC1 standard error
`1.5118578920369088`, and `parity_status=pass`; its SHA-256 is
`67cd686b409379a7dbcc58b8172d1defa6a132bb716458dfd0b0217d47288d95`.

The estimator-level Stata comparator is separate:

```stata
do "benchmarks/validate_did_rcs_csdid.do"
```

It requests repeated cross sections by omitting `ivar()`, uses `method(reg) long2`, runs
both control rules, and writes `benchmarks/validate_did_rcs_csdid_output.txt` before its
assertions. The reviewed Stata/IC 17 artifact matches all 16 point estimates and all six
group-time analytical standard errors exactly. Aggregate SEs are explicitly non-comparable:
`csdid` propagates period-specific cell-share influence, whereas CauseKit and R `did` use
pooled cohort-share influence. The saved artifact SHA-256 is
`ad250fa9cdfd042e1989460ebcec6b1bac57ed301fae390f2c69331ccf47fd57`.

## Publication-scale inference certificate

Run:

```bash
python benchmarks/validate_did_rcs_promotion.py --replications 1000 --workers 8
```

The deterministic 2026-07-30 certificate made 4,000 estimator fits and passed all 32
group-time, event-time, calendar-time, and ES-average cells across balanced and sharply
unequal period sizes and both control rules. Coverage was `0.928–0.958`, mean analytical
SE divided by empirical SD was `0.938–1.029`, maximum absolute bias was `0.0159`, and no
fit refused. Coverage Monte Carlo SE was at most `0.0082`. The run took 29.77 seconds on
the recorded Python 3.14.6 / Windows 11 environment; timing is descriptive.

The covariate-adjusted certificate is separate:

```bash
python benchmarks/validate_did_rcs_covariate_promotion.py --replications 1000 --workers 8
```

It made 4,000 estimator fits and 240,000 fresh fold/task nuisance fits across favorable
balanced and stressed-overlap unequal-period designs, both control rules, three
group-time effects, five public aggregates, and three conditional placebos. All 44
effect/placebo cells passed: pointwise coverage was `0.940–0.964`, mean analytical SE
divided by empirical SD was `0.954–1.048`, maximum absolute bias was `0.0089`, and
coverage Monte Carlo SE was at most `0.0075`. All four joint conditional-pre-trend size
cells passed at rejection rates `0.049–0.057`. There were no refusals, unavailable tests,
clipped probabilities, trimmed rows, or oracle nuisance predictions; the realized fitted
probability range was `0.0290–0.9065`.

The first stressed design used the same conditional cohort probabilities with one-third
of the final sample sizes and correctly produced 11 fold-support refusals in 1,000 draws.
The publication design therefore preserves the probabilities, unequal-period ratio,
estimands, seed, and acceptance gates but requires at least 32 expected observations in
every period-X-cohort cell. Any realized failure still refuses and enters the saved reason
ledger; the promoted run's ledger is empty. The recorded Python 3.14.6 / Windows 11 run
took 120.54 seconds; timing is descriptive. The hash-bound certificate is
`benchmarks/did_rcs_covariate_promotion_evidence.json`.

## Simultaneous event-study contract

The observation-level hand test reconstructs 513 seeded Rademacher max-t draws directly
from the retained event-study influence matrix and HC1 standard errors. A separate PSU
test first sums scores within the declared clusters, draws exactly one multiplier per
indivisible PSU, and reconstructs the CR1-studentized critical value and endpoints. The
implementation uses batches of at most 256 draws; seeded equality and repeat fitting are
part of the contract. The analytic path exposes an empty band and `None` configuration
metadata rather than a pointwise table relabelled as simultaneous.

Refusal tests cover an ordinary-bootstrap label, fewer than 99 iterations, Boolean
iteration counts, noninteger seeds, invalid levels, and nonpositive event standard errors.
The fixed-OOF covariate fixture verifies that the multiplier layer consumes the retained
event influence after exactly the original ten nuisance fold/task fits. A deterministic
100-replication two-event observation-level smoke passed its preregistered joint-coverage
gate of at least 88 covered paths.

The publication certificate is reproduced with:

```bash
python benchmarks/validate_did_rcs_simultaneous_promotion.py --replications 1000 --band-iterations 999 --workers 8
```

It crosses favorable/stressed designs, unadjusted/cross-fitted scores, observation/PSU
sampling, and both control rules. All 16 complete-event-vector cells pass with joint
coverage `0.931–0.961` and Monte Carlo SE at most `0.0081`. All 16,000 estimator fits,
480,000 fold-local nuisance fits, and 15,984,000 multiplier draws complete with zero
refusals; all PSU counts and whole-cluster fold roles match, and minimum realized cell
support is 31 PSUs. The recorded Python 3.14.6 / Windows 11 run took 451.00 seconds;
timing is descriptive. This is internal inferential evidence, not external-package
random-number parity.

## Public-data and performance smoke

`examples/real_world_causal_workflow.py` preserves the hash-pinned Stata `hospdd` rows,
maps hospital adoption time, and fits clustered repeated-cross-section DiD with hospital
as the PSU. The 2026-07-30 local execution returned ESavg `0.867369` and CR1 standard error
`0.042337`. The source itself labels the hospital data artificial; this validates the
workflow, indexing, and clustered execution path, not a substantive causal conclusion.

Reproduce the fixed-period large-sample smoke with:

```bash
python benchmarks/benchmark_did_rcs.py --n-observations 100000 --periods 6 --measure-memory
```

On Python 3.14.6 / Windows 11, the 2026-07-30 run completed in `0.2551` seconds with
`34.656` MiB Python-managed peak memory, six group-time effects, and four event-time
effects. These figures are descriptive and environment-specific.

The large-sample multiplier path is reproducible with:

```bash
python benchmarks/benchmark_did_rcs.py --n-observations 100000 --periods 6 --inference multiplier_bootstrap --bootstrap-iterations 999 --measure-memory
```

The 2026-07-30 Python 3.14.6 / Windows 11 run completed 999 four-coordinate max-t draws
in `2.117` seconds with `208.581` MiB Python-managed peak memory and critical value
`2.4571086724809366`. The bounded batches avoid an iterations-by-observations allocation;
time and memory remain descriptive and environment-specific.

The covariate real-data test uses the same hash-verified 7,368 rows, `frequency` as an
observed covariate, two whole-hospital folds, and hospital CR1 inference. It returns ESavg
`0.8676486723638247`, standard error `0.04274270577554287`, 46 PSUs, and 60 fold/task
diagnostic rows. The source is artificial and the affine/empirical nuisances are an
execution contract, not evidence that the nuisance models or identifying assumptions are
substantively correct. The narrow pairwise composition-robust path has hand-contract,
official R `compdid` point/influence evidence, hash-pinned real-data sensitivity,
fixed-size performance, and eight-cell observation/PSU publication-scale coverage.
Staggered composition robustness, longer robust bands, its aligned diagnostic, and
survey weights remain open. Covariate stationary-composition publication-scale
pointwise and simultaneous
joint-coverage evidence is promoted separately above.

Reproduce the covariate large-sample path with:

```bash
python benchmarks/benchmark_did_rcs_covariate.py --n-observations 100000 --periods 6 --folds 2 --measure-memory
```

The 2026-07-30 Python 3.14.6 / Windows 11 run completed the 100,000-row, 100-fit task plan
in `1.270` seconds with `244.743` MiB Python-managed peak memory. This is a fixed-period
vectorization smoke with deliberately simple nuisance providers, not a learner benchmark;
time and memory are environment-specific.
