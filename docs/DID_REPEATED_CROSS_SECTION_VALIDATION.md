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
missing clean baselines, weak cells, invalid PSUs, covariates, and sampling weights refuse
without row deletion or numerical repair.

## Cross-language hand reference

Run the independent base-R reconstruction:

```bash
Rscript benchmarks/validate_did_rcs_reference.R
```

The reviewed local R 4.5.1 run returned estimate `4`, HC1 standard error
`1.5118578920369088`, the exact eight-element influence vector above, and
`parity_status=pass`. This checks the frozen arithmetic across languages; it is not a
substitute for the still-open estimator-level `did::att_gt(panel = FALSE)` comparison.

Stata must be run manually from the repository root:

```stata
do "benchmarks/validate_did_rcs_stata.do"
```

The do-file writes `benchmarks/validate_did_rcs_stata_output.txt` before asserting. The
reviewed Stata 17 artifact returned estimate `4`, HC1 standard error
`1.5118578920369088`, and `parity_status=pass`; its SHA-256 is
`67cd686b409379a7dbcc58b8172d1defa6a132bb716458dfd0b0217d47288d95`.

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
effects. These figures are descriptive and environment-specific. Publication-scale
coverage, estimator-level R/Stata parity, covariate adjustment, compositional-change
robustness, survey weights, and simultaneous bands remain open.
