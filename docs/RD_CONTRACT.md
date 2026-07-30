# Regression discontinuity contract

This document freezes CauseKit's first continuity-based regression discontinuity (RD)
contract. The estimator is identification-aware: it reports a local cutoff effect only under
the declared sharp or fuzzy assignment design and never treats a visible discontinuity,
bandwidth choice, or density diagnostic as sufficient evidence of causality.

## Public estimands

Let `X` be the running variable, `c` the known cutoff, and `Y` the outcome. The right side is
defined as `X >= c`; the left side is `X < c`.

For a sharp design, treatment must equal `D = 1[X >= c]`. The estimand is

```text
tau_sharp = lim E[Y | X=x] as x -> c+ - lim E[Y | X=x] as x -> c-.
```

Its treatment-assignment jump is known to equal one. If a treatment vector is supplied, it is
audited against the assignment rule and is not reinterpreted as a fuzzy first stage.

For a fuzzy design, `D` must be binary and have a positive discontinuity at the cutoff. The
estimand is the local Wald ratio

```text
tau_fuzzy = jump(Y) / jump(D).
```

The causal label is the cutoff complier local average treatment effect. It additionally requires
an exclusion restriction and monotone take-up response to eligibility. CauseKit refuses a
nonpositive or numerically negligible first stage. It reports, but does not automatically reject,
a weak-first-stage warning when the corrected jump is below `0.10` or its statistic is below
`3.29` in absolute value.

## Local-polynomial point and bias fits

The running variable is centered at the cutoff. Separate weighted polynomial regressions are fit
on the left and right. The point order is `p`; the bias order is `q > p`. CauseKit permits
`0 <= p <= 3` and `p < q <= 4`; local linear `p=1` with local quadratic `q=2` is the default.
The supported kernels are triangular (default), uniform, and Epanechnikov.

Manual bandwidths can be symmetric or a `(left, right)` pair. Bias bandwidths must be at least
as large as their corresponding point bandwidth. No observation is clipped, winsorized,
silently moved across the cutoff, or admitted with a zero kernel weight.

The conventional estimate uses the order-`p` intercept difference. The leading omitted
order-`p+1` term is estimated by the order-`q` fit. CauseKit forms the resulting
design-conditional bias-corrected linear weights directly, so its robust variance includes the
additional bias-estimation variability.

For fuzzy RD, the field-standard first-order ratio correction is used:

```text
B_tau = {B_Y - tau_us B_D} / jump_us(D)
tau_bc = tau_us - B_tau.
```

It is not the generally different ratio `jump_bc(Y) / jump_bc(D)`. The robust fuzzy score uses
the corrected intercept weights and the conventional local-Wald residual combination, matching
the maintained `rdrobust` fixed-bandwidth definition.

## Native bandwidth selection

`bandwidth="native_mse"` is the default. It is CauseKit's bounded, deterministic,
design-conditional MSE grid; it is not labeled as CCT `mserd`, IK, or coverage-error optimal.

1. A robust running-variable scale is the smaller of its standard deviation and normalized
   interquartile range. The reference width is `2.5 * scale * n^(-1/5)`.
2. Candidate distances run from the minimum estimable local support through the smaller of the
   80th percentile and `1.5` reference widths. This prohibits an accidental global fit when one
   side has a very long tail.
3. A fixed order-`q` pilot extends through the smaller of the 90th percentile and `2.5` reference
   widths and estimates curvature, residual variation, and—under fuzzy RD—the local treatment
   jump and outcome/treatment residual covariance.
4. Every bounded left/right pair is scored by estimated leading squared bias plus variance. A
   fuzzy candidate with a nonpositive local treatment jump remains in the ledger but is
   inadmissible for selection.
5. The minimizing admissible pair is selected deterministically. The pilot bandwidths become the bias
   bandwidths.

The entire candidate ledger, objective components, sample counts, and selected row are returned
in `result.bandwidth_selection`. Selection at a lower or upper candidate boundary is explicitly
flagged by side and should trigger bandwidth sensitivity analysis; it does not silently expand
the local-support cap. Manual fixed bandwidths remain the reproducibility and
cross-software parity contract. A bandwidth is a bias/variance decision, not an identification
test; sensitivity across credible bandwidths remains the analyst's responsibility.

## Inference

The primary public estimate is the robust bias-corrected estimate. CauseKit also exposes the
conventional point estimate and conventional HC1 standard error, but it does not recommend a
conventional interval at a data-driven MSE bandwidth.

For independent observations, the primary covariance is side-specific HC1 applied to the exact
bias-corrected score weights. For one-way clusters, score contributions are summed by local
cluster and a CR1 correction is applied; at least two represented clusters are required on each
side. Robust inference uses the normal reference distribution. Clustered inference uses a
`t(G-1)` reference distribution.

The estimator never creates an observation-by-observation projection matrix. Local moment
matrices have dimension at most five, score aggregation is vectorized, and cluster aggregation
uses integer factor codes.

## Running-variable support and mass points

The cutoff must lie strictly inside observed support. Each point and bias fit must have more
positive-weight observations than parameters, enough unique running values for its polynomial
order, full local rank, and a condition number below the declared ceiling.

`mass_points="check"` reports duplicate local running values. `mass_points="raise"` refuses
them. The first release does not claim discrete-running-variable asymptotics, donut-hole
identification, or honest inference after analyst-driven support searches.

Pandas inputs must have exactly equal indices in the same order. Missing values are either
jointly refused (`missing="raise"`) or jointly removed (`missing="drop"`). Row labels are
preserved on weights, local support, and score contributions.

## Manipulation diagnostic

Every result includes a separate one-sided triangular boundary-kernel log-density diagnostic.
It reports left/right density estimates, their log difference, an influence-based standard error,
and a two-sided p-value. At least five positive-weight observations are required per side.

This diagnostic is intentionally named
`one_sided_boundary_kernel_density`. It is **not** the robust local-polynomial density test of
Cattaneo, Jansson, and Ma and is not represented as one. Failure to reject does not establish an
absence of precise sorting; rejection does not automatically select another estimator or delete
observations. Density continuity, covariate balance, placebo cutoffs, institutional knowledge,
and graphical evidence remain distinct validity checks.

## Returned evidence

`RegressionDiscontinuityResult` contains:

- primary labeled parameters, covariance, standard errors, tests, and confidence intervals;
- conventional and bias-corrected effect layers;
- outcome and treatment jumps plus fuzzy first-stage diagnostics;
- point/bias bandwidths, selection ledger, polynomial orders, kernel, and condition numbers;
- local left/right counts, unique support, mass-point count, and cluster degrees of freedom;
- exact conventional/bias-corrected weights and robust score contributions;
- a local plotting table and optional binned diagnostic plot; and
- the manipulation diagnostic and identification assumptions.

OutputHub exports the model, bandwidth ledger, and manipulation diagnostic without refitting or
exporting raw outcomes by default.

## Validation and parity boundary

Fixed-bandwidth triangular `p=1`, `q=2`, HC1 sharp and fuzzy effects are compared against the
maintained Python `rdrobust` 2.0.0 and R `rdrobust` 4.0.0 implementations. Conventional and
bias-corrected point estimates, robust standard errors, and the fuzzy corrected treatment jump
match at numerical precision. Reviewed Stata/IC 17 `rdrobust` 11.1.0 uses the same deterministic
fixture and passes every aligned field with maximum absolute difference `2.31e-14`; its do-file
writes output before assertions.

The native bandwidth selector is CauseKit-specific. It must be validated by deterministic
recovery, coverage, sensitivity, performance, and refusal gates; it is not expected to reproduce
another package's selector.

Artifact hashes, exact promotion results, comparator versions, and reproduction commands are
recorded in [Regression discontinuity promotion evidence](RD_PROMOTION_EVIDENCE.md).

## Explicitly unavailable in this contract

- covariate-adjusted RD;
- regression kink, geographic, multi-score, or multi-cutoff designs;
- discrete-running-variable or local-randomization inference;
- donut-hole automation, placebo-cutoff multiplicity correction, or specification searching;
- survey weights, frequency weights, multiway clustering, or cluster leverage corrections;
- coverage-error-optimal bandwidths or the official Cattaneo-Jansson-Ma density statistic; and
- a claim that the density diagnostic or local fit verifies causal identification.

These require separately derived scores, targets, and validation evidence before implementation.

## Pre-mortem and promotion gates

The most likely failure modes are weak fuzzy first stages, selected bandwidths at a grid boundary,
insufficient unique support, manipulation/heaping near the cutoff, ill-conditioned higher-order
fits, and nominal intervals that undercover nonlinear designs. Promotion therefore requires:

- hand-computed sharp and fuzzy identities and explicit refusal paths;
- row-order and labeled-index invariance;
- maintained Python/R fixed-bandwidth parity and reviewed Stata output;
- nonlinear sharp/fuzzy recovery and pointwise coverage with zero unexpected refusals;
- a hash-pinned real-data sensitivity record;
- large-sample performance without quadratic storage; and
- formatting, typing, packaging, dependency, and full-suite checks.

Methodological anchors are Hahn, Todd, and van der Klaauw's local-Wald identification result and
Calonico, Cattaneo, and Titiunik's robust bias-corrected local-polynomial inference. The maintained
software comparator and references are available from the
[RD Packages project](https://rdpackages.github.io/rdrobust/).
