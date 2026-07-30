# Composition-change-robust repeated-cross-section DiD contract

## Status and boundary

`composition="robust"` is implemented through `RepeatedCrossSectionDiD` for pairwise,
longer, and staggered repeated-section designs with non-empty covariates and an explicit
`CrossFitter`. Every pair targets its treated target-period population, retains four
normalized cell weights and a full-sample influence, and supports observation/PSU
analytical and optional multiplier inference.

Pairwise and longer external-boundary, real-data, performance, pointwise, diagnostic, and
simultaneous-coverage gates pass. The public aligned equality diagnostic is implemented;
it reports both estimators and does not select between them. Survey-population combinations
remain open under their separate contract.

Composition robustness is not a weighting option on the stationary score. It changes the
target population, generalized propensity nuisance, efficient influence function,
aggregation shares, overlap checks, and interpretation. The balanced-panel direct
cohort-ratio nuisance and repeated-section survey designs remain independently
contracted.

## Pairwise estimand

For treated cohort `g`, target period `t`, clean baseline `b`, and fixed eligible
comparison population `C(g,t)`, restrict the data to the two groups and two periods. Let
`D=1` identify cohort `g`, `D=0` the comparison population, and `S=1` the target-period
sample. The target is

```text
ATT_cc(g,t)
  = E[Y_t(1) - Y_t(0) | D=1, S=1].
```

It is explicitly the effect for treated members of the target-period population. It is
not the effect for treated rows pooled across periods. This distinction must appear in
the fitted result, summary, OutputHub metadata, graph data, and aggregation tables.

Identification requires consistency, no interference, the declared no-anticipation
window, conditional parallel trends for the target population, and strong support for
all four `(D,S)` cells given the baseline covariates. Unlike the stationary path, it does
not assume `(D,X)` is independent of sample period. Covariates are therefore required and
may not be an empty cosmetic matrix.

## Nuisance functions and score

For `d,s in {0,1}`, define

```text
p_ds(X) = P(D=d, S=s | X)
m_ds(X) = E[Y | D=d, S=s, X].
```

The target-cell weight is

```text
w_11 = D*S / E[D*S],
```

and each non-target cell uses the normalized generalized-propensity ratio

```text
w_ds = I(D=d,S=s) * p_11(X)/p_ds(X)
       / E[I(D=d,S=s) * p_11(X)/p_ds(X)].
```

With the target-cell pseudo-outcome

```text
tau_target(Y,X) = Y - m_10(X) - m_01(X) + m_00(X),
```

the maintained efficient influence representation is

```text
psi_cc
  = w_11 * (tau_target(Y,X) - ATT_cc)
    + sum[(d,s) != (1,1)] (-1)^(d+s) * w_ds * (Y - m_ds(X)).
```

The implementation must derive its estimate and centered observation influence from this
same normalized moment. It may use the equivalent outcome-regression identification

```text
E[Y | D=1,S=1]
  - E[m_10(X) + m_01(X) - m_00(X) | D=1,S=1]
```

only as an independent test oracle. No stationary binary propensity or pooled treated
distribution may enter the robust score.

The score has a rate-doubly-robust nuisance remainder under the paper's regularity and
cross-fitting conditions: appropriate mean-square errors must vanish and their product
must be sufficiently small. CauseKit will use that precise label. It will not claim
unqualified model double robustness, finite-sample unbiasedness, or protection from
unmeasured composition changes.

## CrossFitter execution

The implementation reuses the public `CrossFitter` rather than owning learners. Each pair
uses one four-class probability task predicting
the ordered cells `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`. Three masked scalar outcome tasks
predict `m_10`, `m_01`, and `m_00`; `m_11` is not a nuisance in the efficient score and
must not be fitted as unused work. An external comparator's optional fourth column may be
retained only in the comparator artifact, not the runtime task graph.

A single immutable outer-fold plan is shared by all nuisance tasks, group-time effects,
and conditional pre-trend placebos. It is stratified by cohort-period cell and keeps each
declared PSU whole. Training masks and prediction indices remain auditable. Inner tuning
may use only an outer training partition. All four probabilities must be aligned,
strictly inside the declared floor, and sum to one within tolerance; CauseKit does not
clip or renormalize them.

The longer-design path uses the comparison-set, anticipation, and max-t mechanics through
the promoted global-fold, pair-ledger, zero-padded influence, and target-period share
contract. Stationary pooled cohort shares are not reused.

## Composition diagnostic

The detailed
[diagnostic and influence-alignment contract](DID_RCS_COMPOSITION_DIAGNOSTIC_ALIGNMENT_CONTRACT.md)
freezes a robust-minus-stationary Hausman-type equality comparison. It computes covariance
from the aligned difference influence, not by subtracting marginal variances, and uses
the existing HC1 or PSU-CR1 finite-sample convention. Singular systems and every sample,
role, pair, fold, covariance, or cluster mismatch refuse without numerical repair.

The diagnostic is model validation, not model selection. CauseKit never selects the
stationary estimator after failure to reject and then reports ordinary post-selection
inference. Users concerned about composition change report both estimates and the aligned
sensitivity comparison.

## Required refusals

The robust path must refuse:

- missing or empty covariates, or covariates without an explicit `CrossFitter`;
- fewer than the required observations or independent PSUs in any of the four cells,
  globally or within a training fold;
- generalized probabilities with missing, extra, or duplicate classes, ambiguous or
  unlabeled column order, index drift, nonfinite values, row sums outside tolerance, or
  any cell at/below the probability floor; labelled permutations are realigned safely;
- own-observation predictions, mutable row/PSU roles, or cross-task fold drift;
- a target-period treated cell with a nonpositive denominator;
- direct-ratio-only or binary-propensity nuisances presented as the four-cell score;
- survey weights or survey designs until the combined population-and-composition score
  has its own theorem, hand contract, and coverage gate;
- diagnostic inputs with different samples, targets, folds, support rules, covariance,
  or aggregation definitions; and
- requests to interpret non-rejection as proof of stable composition.

No cell pooling, clipping, row trimming, implicit baseline substitution, fallback to the
stationary score, or variance repair is permitted.

## Current failing-first evidence

`tests/test_did_rcs_composition.py` was added and observed with four failures at the old
public refusal before runtime code changed. The retained tests now reconstruct the exact
two-by-two estimate, all four normalized weights, the complete efficient influence
vector, and HC1 standard error. They also verify the target and nuisance schema, absence
of an unused `m_11` fit, non-empty-covariate refusals, hard four-cell
overlap refusal without clipping, unsupported pair/fold refusal, and immutable
row/whole-PSU cross-fitting. Existing
stationary tests remain unchanged apart from allowing the now-supported robust label.

The retained deterministic gate uses 32 observations with binary `X` and exact four-cell
probabilities. Treated baseline composition has `E[X]=0.25`, while treated target-period
composition has `E[X]=0.75`; the heterogeneous effect is `2+4X`. The robust score exactly
recovers target-period ATT `5`, whereas the stationary score equals its distinct pooled-
treated target `4`, demonstrating one unit of target bias if it is misreported as the
post-period ATT. A second failing-first run recorded two intended failures: extra class
columns were silently ignored and duplicates produced only an incidental shape error.
The reusable protocol now refuses missing, extra, duplicate, and unlabeled class schemas,
safely realigns labelled permutations, and preserves estimates, standard errors, weights,
influences, nuisance predictions, and design fingerprint under row permutation.

The official external gate now passes. The R harness pins `compdid` 0.1.0 source commit
`894bd65a952c30f01a4e0005efba4cb335065eb7` and the exact `att-core.R` and
`dp-grooming.R` Git blobs, maps the persisted CauseKit probability order
`(00,01,10,11)` into the official `(11,10,01,00)` contract, and calls
`drdid_nonstationary(stabilized=TRUE, boot=FALSE, inffunc=TRUE)`. CauseKit and R agree on
the point estimate, sample-SD/HC1 standard error, and all 16 ordered influence records at
absolute tolerance `2e-14`. The official API's required `m11` column cancels algebraically
between its target residual and regression contrast, matching CauseKit's deliberate
three-outcome-nuisance execution path.

The pairwise promotion gates now also pass. The hash-pinned Sequeira sensitivity reports
robust and stationary estimates together under five whole-PSU folds; it deliberately
shows material nuisance-and-target sensitivity rather than claiming reproduction of the
paper's different local-polynomial contract. The 100,000-row benchmark completes the
eight-fit nonlinear task graph without a quadratic observation matrix. The fixed-seed
publication certificate completes 8,000 estimator fits and 72,000 fold-level nuisance
fits with zero refusals; all eight bias, SE-calibration, and pointwise-coverage cells pass.
See [the full promotion evidence](DID_RCS_COMPOSITION_PROMOTION_EVIDENCE.md).

## Completed extension sequence

Hand-computed scalar/vector diagnostics and alignment refusals, conditional pre-trends,
fixed-seed max-t identities, target-share staggered aggregation, and publication-scale
coverage now pass. The completed simulations cover favorable stationarity and composition change,
nonlinear fitted nuisances, unequal cell probabilities, observation/PSU inference, bias,
SE calibration, pointwise coverage, overlap, audit counts, zero fallbacks, runtime, and
peak memory. They report diagnostic size under stationarity, power under composition shift,
and the efficiency cost of robustness without converting the diagnostic into selection.
See [the longer promotion evidence](DID_RCS_COMPOSITION_LONGER_PROMOTION_EVIDENCE.md).

The primary external comparator is the authors' official R `compdid` implementation.
Its pinned point/influence gate and subsequent standard-error comparison now pass through
`benchmarks/validate_did_rcs_compdid_reference.R`; the saved artifact is validated in
`tests/validation/test_did_rcs_compdid_parity.py`. Stata is recorded unavailable unless a
reviewed command targets the same post-period treated ATT and influence moment. The hash-
pinned real-data sensitivity examples report robust and stationary estimates together
without pretest-based selection. Official `compdid` has no maintained aligned longer or
staggered interface, so that comparator cell remains unavailable rather than manufactured.

The remaining repeated-section extension is survey-population combination. It stays under
the separate survey contract because design weights change the target and variance; raw
sampling weights continue to refuse.

## Pre-mortem

The principal failure is accidentally reusing the stationary binary propensity and
pooled treated distribution under a new label. Four-class task audits, a shifting-
composition hand example, and explicit target metadata address it. Other likely failures
are empty fold cells, inconsistent generalized-probability column order, a Hausman test
computed from marginal variances instead of the difference influence, and pretest-based
model selection. Each is a hard contract or refusal above.

## Primary methodology and comparator

- Pedro H. C. Sant'Anna and Qi Xu (2026), [*Difference-in-Differences with Compositional
  Changes*](https://doi.org/10.1016/j.jeconom.2025.106147).
- The authors' official R [`compdid` nonstationary
  estimator](https://psantanna.com/comp_did/reference/drdid_nonstationary.html) and
  [`drdid_stationarity_test`](https://psantanna.com/comp_did/reference/drdid_stationarity_test.html).
- Pedro H. C. Sant'Anna and Jun Zhao (2020), [*Doubly Robust
  Difference-in-Differences Estimators*](https://doi.org/10.1016/j.jeconom.2020.06.003),
  for the distinct stationary-composition benchmark.
