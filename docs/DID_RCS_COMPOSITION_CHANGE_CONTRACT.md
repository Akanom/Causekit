# Composition-change-robust repeated-cross-section DiD contract

## Status and boundary

The first `composition="robust"` slice is implemented through
`RepeatedCrossSectionDiD`: exactly two periods, one treated cohort, a fixed never-treated
comparison population, non-empty covariates, and an explicit `CrossFitter`. It targets the
treated target-period population, retains its four normalized cell weights, and supports
the existing observation/PSU analytical and optional multiplier-inference machinery.

Staggered group-time aggregation, conditional pre-trends with additional periods, the
stationary-versus-robust composition diagnostic, survey designs, external estimator
parity, publication-scale coverage, and promotion beyond this first slice remain open.
No placeholder diagnostic or staggered wrapper is exported.

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

The implemented first phase is the pairwise two-group, two-period score. It reuses the
public `CrossFitter` rather than owning learners. One four-class probability task predicts
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

Only after the pairwise score and inference are promoted may the staggered wrapper reuse
the current comparison-set, anticipation, aggregation, and max-t infrastructure. It must
retain target-period treated-population cohort shares and their full estimated-share
influence. Stationary pooled cohort shares are not reusable.

## Composition diagnostic

A future `RepeatedCrossSectionCompositionDiagnostic` compares aligned robust and
stationary estimates using the empirical second moment of their difference influence:

```text
delta = ATT_cc - ATT_stationary
psi_delta = psi_cc - psi_stationary
V_delta = E[psi_delta^2]
W = n * delta^2 / V_delta.
```

For a vector, CauseKit uses the full covariance of `psi_delta` and a rank-preserving
linear solve. Singular systems refuse without ridge, pseudoinverse, dropped coordinates,
or changed rank. Both inputs must share the analysis rows, target, comparison rule,
covariates, baseline/target periods, anticipation, stabilization convention, fold plan,
and observation/PSU inference role.

The diagnostic is model validation, not model selection. CauseKit will never select the
stationary estimator after a failure to reject and then report ordinary post-selection
inference. Users concerned about composition change should report the robust result and
the aligned sensitivity comparison.

## Required refusals

The robust path must refuse:

- missing or empty covariates, or covariates without an explicit `CrossFitter`;
- fewer than the required observations or independent PSUs in any of the four cells,
  globally or within a training fold;
- generalized probabilities with missing classes, wrong order, index drift, nonfinite
  values, row sums outside tolerance, or any cell at/below the probability floor;
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
of an unused `m_11` fit, non-empty-covariate and pairwise-scope refusals, hard four-cell
overlap refusal without clipping, and immutable row/whole-PSU cross-fitting. Existing
stationary tests remain unchanged apart from allowing the now-supported robust label.

## Remaining promotion sequence

Promotion beyond the implemented first slice still requires:

1. exact recovery when four-cell composition changes under the maintained conditional
   parallel-trends restriction and a contrasting stationary-score bias example;
2. row-permutation invariance and broader generalized-propensity class/order refusals;
3. hand-computed scalar and vector composition diagnostics, including singular and
   misaligned-input refusals;
4. conditional pre-trends for longer designs and fixed-seed max-t identities using robust
   influence records without nuisance refitting; and
5. staggered group-time aggregation only after the pairwise evidence below passes.

Promotion requires separate favorable-stationarity and composition-change simulations,
including nonlinear nuisances and unequal period sizes. Preregistered cells must cover
bias, empirical/analytical SE calibration, pointwise and simultaneous coverage,
composition-test size and power, overlap, support, audit counts, zero silent fallbacks,
runtime, and peak memory. Efficiency loss under true stationarity must be reported rather
than hidden.

The primary external comparator is the authors' official R `compdid` implementation,
pinned by released version and source commit. CauseKit must map its four probability and
outcome columns explicitly and compare point estimates and retained influence records
before comparing standard errors. Stata is recorded unavailable unless a reviewed command
targets the same post-period treated ATT and influence moment. A hash-pinned real-data
sensitivity example must report robust and stationary estimates together without
pretest-based selection.

Staggered group-time aggregation becomes a later promotion gate after the pairwise score
passes. It requires both control rules, anticipation, target-period treated shares and
share influence, conditional placebos, event/calendar/ESavg aggregation, observation/PSU
inference, simultaneous bands, publication-scale coverage, OutputHub, and graph-data
parity.

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
