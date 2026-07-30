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
cluster-summed CR1 variance. Malformed roles, unsupported composition and inference,
missing clean baselines, weak cells, invalid PSUs, covariates without an explicit
`CrossFitter`, and sampling weights refuse without row deletion or numerical repair.

The covariate contract in `tests/test_did_rcs_covariate.py` was also observed failing
before implementation. Its fixed-OOF fixture has ATT `2.4875`, influence
`[-.975, -.575, .225, 1.025, -.775, -.175, .425, .825, 0, ..., 0]`, and HC1 standard
error `0.12706625568314087`. It reconstructs all eight normalized score components and
ratio influence terms. Separate contracts verify both double-robustness legs, shared
folds, row/PSU non-leakage, hard overlap refusal, conditional placebos, staggered
never/not-yet-treated aggregation, fixed-seed reproduction, and OutputHub transport.

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

The covariate real-data test uses the same hash-verified 7,368 rows, `frequency` as an
observed covariate, two whole-hospital folds, and hospital CR1 inference. It returns ESavg
`0.8676486723638247`, standard error `0.04274270577554287`, 46 PSUs, and 60 fold/task
diagnostic rows. The source is artificial and the affine/empirical nuisances are an
execution contract, not evidence that the nuisance models or identifying assumptions are
substantively correct. Composition-change robustness, survey weights, and simultaneous
bands remain open; covariate publication-scale pointwise coverage is promoted separately
above.

Reproduce the covariate large-sample path with:

```bash
python benchmarks/benchmark_did_rcs_covariate.py --n-observations 100000 --periods 6 --folds 2 --measure-memory
```

The 2026-07-30 Python 3.14.6 / Windows 11 run completed the 100,000-row, 100-fit task plan
in `1.270` seconds with `244.743` MiB Python-managed peak memory. This is a fixed-period
vectorization smoke with deliberately simple nuisance providers, not a learner benchmark;
time and memory are environment-specific.
