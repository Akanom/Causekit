# Composition diagnostic and longer-design influence-alignment contract

## Status and claim boundary

This is a design-only contract. CauseKit currently implements and promotes only the
two-group, two-period `RepeatedCrossSectionDiD(composition="robust")` score. This document
freezes the next two implementation layers:

1. an aligned Hausman-type equality diagnostic comparing robust and stationary scores;
2. the pair lattice, folds, influence records, aggregation shares, placebos, and bands
   required before composition robustness can support longer or staggered designs.

No diagnostic class, test function, `composition="both"` mode, longer robust estimator,
or staggered wrapper is exported by this contract. Public names are implemented only
after failing behavioral/API tests freeze them. The proposed first names are
`did_rcs_composition_test` and `RepeatedCrossSectionCompositionDiagnostic`.

The authors' maintained `compdid` implementation compares its two-period nonstationary
and stationary estimates through the empirical second moment of their difference
influence. CauseKit follows that comparison principle. The multi-period and staggered
pair lattice below is a CauseKit extension of the promoted pairwise score; it is not
attributed to the paper or official package as an already established staggered
estimator.

## Distance from the full composition-robust design

| Layer | Current status | Remaining work |
| --- | --- | --- |
| Pairwise estimand, score, weights, HC1/CR1, multiplier path | Promoted | Maintain unchanged |
| Pairwise hand/refusal tests and official R point/influence parity | Promoted | Add diagnostic parity |
| Real-data sensitivity, performance, pairwise pointwise coverage | Promoted | Do not rerun for unrelated work |
| Composition diagnostic | Contracted here | Failing-first implementation, R parity, size/power promotion |
| Longer-design pair and influence alignment | Contracted here | Runtime task lattice and hand identities |
| Conditional robust pre-trends and simultaneous bands | Formula frozen here | Implementation and joint coverage |
| Staggered event/calendar/ES-average aggregation | Formula frozen here | Implementation, parity where available, promotion |
| Survey-population combination | Separate design-only contract | Not part of this sequence |

Thus the statistically difficult pairwise score is complete, but three delivery blocks
remain for a full sample-target design: the diagnostic, longer/staggered runtime
orchestration, and publication promotion of the resulting vector. Completion is better
described by those gates than by a line-count percentage.

## Longer-design pair lattice

For each treated cohort `g`, admissible target period `t`, and declared anticipation
window `a`, define the effective treatment boundary and clean baseline

```text
e(g) = g - a
b(g) = e(g) - 1.
```

For post-treatment group-time effects, the pair is `(g, b(g), t)` for every
`t >= e(g)`. The comparison population `C(g,t)` follows the existing public rule:

- `never_treated`: cohorts coded as never treated;
- `not_yet_treated`: cohorts untreated and outside the anticipation window at `t`.

Comparison membership is fixed using `t` and then used unchanged in both `b(g)` and `t`.
CauseKit does not allow a later-treated cohort to enter only the baseline side of a pair.

Within a pair, retain only cohort `g` and `C(g,t)` at `b(g)` and `t`. Let `D=1` identify
cohort `g`, `D=0` the fixed comparison population, `S=1` target period `t`, and `S=0`
baseline `b(g)`. Every group-time effect is the already promoted pairwise estimand

```text
ATT_cc(g,t) = E[Y_t(1) - Y_t(0) | G=g, T=t].
```

The four generalized probabilities are pair-conditional:

```text
p_ds^(g,t)(X)
  = P(D=d, S=s | X, row belongs to pair (g,b(g),t)).
```

Their common conditioning probability cancels from each target-to-source ratio. A giant
multinomial over every cohort-period cell is not equivalent and is not used. Each pair
fits one ordered four-class probability nuisance and exactly three outcome nuisances
`m00`, `m01`, and `m10`; `m11` remains absent.

## One immutable global fold plan

All robust and stationary pair tasks in one comparison workflow use one fold vector over
the complete validated analysis sample. It is generated once from the full
cohort-by-period strata and keeps every declared PSU whole. Each pair-specific task then:

- trains only on pair rows outside the held-out global fold;
- predicts only relevant pair rows in that held-out fold;
- retains the same row or PSU role in every other pair, nuisance, placebo, and score;
- records pair key, nuisance name, fold, training support, relevant holdout support,
  model identity, and provider diagnostics.

Generating folds independently inside each pair is prohibited. A row can contribute to
several group-time effects; changing its role across pairs compromises the intended joint
out-of-fold vector and makes leakage auditing and covariance comparisons unreliable.

The required reusable orchestration addition is a masked multiclass task operation,
provisionally `ClassProbabilityCrossFitTask`, executed by `CrossFitter` on the shared fold
plan. It must retain exact labelled class schemas and fresh estimators per task/fold. It
does not create a CauseKit-owned classifier.

Every training fold must contain all four pair cells and the support required by its
provider. An unsupported pair refuses the complete requested fit; CauseKit does not
reduce folds, merge cells, borrow another pair's nuisance, or silently omit an effect.

## Full-sample influence alignment

Let `R_j` identify rows in pair `j=(g,b,t)`, `n` be the complete sample size, and `n_j`
the pair size. Evaluate the promoted pair score on pair rows and embed it into the common
sample index:

```text
phi_ij = (n / n_j) * R_ij * phi_ij,pair.
```

Rows outside the pair receive exact zero, not missing values. Each full-sample influence
column must be finite, centered within numerical tolerance, and labelled by the complete
pair key `(cohort, target_time, base_period)`. Its corresponding point estimate,
comparison-cohort tuple, four cell counts, target-cell count/share, probability bounds,
weight diagnostics, and nuisance-task keys are retained in a pair ledger.

All `group_time_influence`, placebo, event, calendar, ES-average, diagnostic, and
simultaneous-band objects use this identical row order. Clustered paths additionally use
the identical PSU label series. Pairwise row deletion, independent complete-case
processing, or alignment by positional coincidence refuses.

## Target-period aggregation shares

Stationary pooled cohort shares are invalid for the robust target. For group-time cell
`j=(g,t)`, define its target-cell indicator and sample mass

```text
r_ij = I(G_i=g, T_i=t)
q_j  = E[r_ij].
```

For any event-time or calendar-time set `A`, use target-period treated shares

```text
Q_A       = sum[j in A] q_j
omega_j,A = q_j / Q_A
theta_A   = sum[j in A] omega_j,A * ATT_cc(j).
```

The full influence includes estimated-share uncertainty:

```text
phi_i,A
  = sum[j in A] omega_j,A * phi_ij
    + (1 / Q_A) * sum[j in A]
        (ATT_cc(j) - theta_A) * (r_ij - q_j).
```

Tests reconstruct both terms. Treating target shares as fixed, substituting pooled cohort
frequencies, or using only period-specific sample proportions without their share
influence is prohibited.

Event-time estimates aggregate all cells with a common event time. Calendar-time
estimates aggregate treated cells in the same target period. To preserve the existing
public ES-average definition, ESavg is the arithmetic mean over the retained nonnegative
event-time estimates; its influence is the same fixed arithmetic mean of their complete
influences. The result explicitly records these two weighting stages.

## Conditional robust placebos

For each admissible pre-treatment target period `s < e(g)`, use the adjacent clean pair
`(g, s-1, s)` and the comparison population fixed at `s`. Apply the same four-cell robust
score, pair-specific nuisance tasks, global folds, zero-padded influence alignment, and
target-period cohort population. The null is an adjacent conditional pre-treatment
effect of zero for that target population.

The placebo table retains every coordinate even when the joint covariance is singular.
The joint test then reports unavailable with the exact reason; it does not drop a
placebo, alter rank, add ridge, or use a pseudoinverse. Failure to reject neither proves
conditional parallel trends nor rules out unmeasured composition changes.

## Composition equality diagnostic

The first public diagnostic accepts an aligned robust result and an aligned,
covariate-adjusted stationary result. Its maintained comparison vector is the complete
sorted group-time support shared by both results:

```text
delta       = ATT_robust - ATT_stationary
Phi_delta   = Phi_robust - Phi_stationary
V_delta     = Cov(Phi_delta)
statistic   = delta' solve(V_delta, delta).
```

The orientation is always robust minus stationary. `V_delta` is computed directly from
the difference influence, never by subtracting the two marginal covariance matrices.
For auditability, the result also exposes robust covariance, stationary covariance, and
their cross-covariance.

Observation inference uses the package's HC1 cross-covariance convention and a chi-square
reference with degrees of freedom equal to the number of group-time restrictions. PSU
inference first sums every influence column within the shared PSU, applies the package's
CR1 correction, and uses the existing finite-cluster `F(q, G-1)` reference. A scalar
pairwise comparison is the one-coordinate special case. The official comparator's HC0
second-moment statistic is recorded with its explicit HC0-to-HC1 mapping rather than
silently called numerically identical.

The covariance must be positive definite at the declared tolerance. A singular or nearly
singular vector refuses without ridge, pseudoinverse, coordinate deletion, rank choice,
or conversion to separate unadjusted tests. The first implementation has no automatic
bootstrap fallback. A later difference-influence multiplier test requires its own
fixed-seed identity and size contract.

The proposed immutable result contains:

- statistic, p-value, rejection flag, confidence level, distribution, numerator and
  denominator degrees of freedom, and PSU count;
- the ordered robust, stationary, and difference estimates;
- aligned difference influence, robust/stationary/cross/difference covariance matrices;
- pair keys, estimation index, inference clusters, covariance declaration, and design
  fingerprint;
- null hypothesis and notes that distinguish equality from proof of stationarity.

It contains no estimator-selection recommendation. Rejection indicates evidence against
equality of the maintained robust and stationary moments. Failure to reject does not
verify stationary composition, and neither outcome authorizes ordinary post-selection
inference. Applied output reports both estimates and the diagnostic together.

## Alignment requirements and refusals

The diagnostic refuses unless both inputs have:

- exact `RepeatedCrossSectionDiDResult` types, with `composition` equal to `robust` and
  `stationary` in the required argument order;
- the same ordered estimation index, outcome, covariates, time and treatment-time roles,
  period support, anticipation, control rule, never-treated code, and probability floor;
- the same pair keys, baseline periods, fixed comparison-cohort tuples, and group-time
  coordinate order;
- the same immutable nuisance-fold series, split count, and whole-PSU roles;
- cross-fitted covariate adjustment for both paths; an unadjusted stationary result is
  not an aligned comparator;
- the same covariance declaration and exactly equal inference-cluster labels; and
- finite, centered influence matrices with identical row/coordinate labels.

It also refuses invalid confidence levels, empty common support, duplicate coordinates,
nonpositive scalar variance, insufficient PSUs, singular vector covariance, modified or
reduced influence records, mismatched fingerprints, and any request to select or refit an
estimator inside the diagnostic.

The longer robust estimator refuses unsupported pair cells/folds, multiple baselines for
one post-treatment cell, changing comparison membership across the two periods,
pair-local fold regeneration, missing influence rows, inconsistent zero padding,
stationary aggregation shares, unsupported survey combinations, or nuisance refitting in
post-estimation bands and diagnostics.

## Alternatives rejected

- **One global cohort-period multinomial:** class dimension and support grow with the
  design, while its ratios do not encode the maintained pair-specific comparison target.
- **Independent pair-local folds:** computationally convenient but changes row/PSU roles
  across coordinates and weakens honest joint auditing.
- **Marginal variance subtraction:** does not equal the covariance of the estimator
  difference and can be negative or misleading.
- **Automatic stationary/robust selection:** turns a diagnostic into a pretest estimator
  and invalidates ordinary inference.
- **Pooled cohort-share aggregation:** targets a different population when treated
  composition changes by period.

## Failing-first implementation order

### Gate A — pairwise diagnostic

1. Hand-computed scalar difference, influence, HC1 variance, statistic, and p-value.
2. A clustered fixture reconstructing PSU sums, CR1, and the finite-cluster reference.
3. Official R `drdid_stationarity_test()` parity under fixed influence inputs, with the
   documented HC0/HC1 mapping.
4. Argument-order, sample, fold, covariate, role, support, covariance, cluster, and
   singularity refusals.
5. OutputHub and summary evidence that always presents both estimators and never selects.

### Gate B — longer influence lattice

1. One cohort and at least three periods: hand-reconstruct every post pair, zero-padded
   influence column, and adjacent pre-treatment placebo.
2. Multiple cohorts under never-treated and not-yet-treated controls: verify fixed pair
   membership, anticipation, exact task counts, and one global row/PSU fold role.
3. Hand-computed event/calendar target shares and complete estimated-share influence.
4. Row, labelled-column, pair-order, and PSU permutation invariance.
5. Strict four-cell global/fold overlap and unsupported-pair refusals.

### Gate C — joint post-estimation

1. Direct covariance reconstruction for group-time, event, calendar, ESavg, and placebo
   vectors under observation and PSU inference.
2. Fixed-seed max-t identity from the retained robust event influence with no refit.
3. Scalar and vector equality diagnostics on longer aligned results.
4. OutputHub and graph-data parity for every point, interval, share, and diagnostic field.

### Gate D — promotion

Preregistered simulations cross favorable stationarity and composition shift, nonlinear
nuisances, unequal waves, heterogeneous effects, never/not-yet controls, observation/PSU
sampling, and multiple cohorts. They report robust target bias, stationary target bias,
SE calibration, pointwise and simultaneous coverage, diagnostic size and power, Monte
Carlo uncertainty, overlap, task/fold counts, refusals, runtime, and peak memory. Fixed
seeds and zero silent fallbacks are mandatory.

The pairwise Sequeira, performance, and coverage certificates are already settled and are
not rerun merely to implement this extension. Longer-design real-data evidence must report
both paths and all diagnostics without pretest selection. Where no maintained R or Stata
estimator exposes the aligned staggered target and influence, the comparator cell remains
unavailable rather than being manufactured.

## Pre-mortem

The most likely failures are a row changing folds between overlapping pairs, comparison
cohorts drifting between baseline and target, aggregation silently reverting to pooled
shares, a Hausman covariance formed from marginal variances, and an applied helper
selecting the stationary estimator after non-rejection. Pair-ledger hashes, a global fold
vector, zero-padded labelled influences, explicit target-share terms, difference-influence
covariance, and a no-selection result contract are the required defenses.

## Primary sources

- Pedro H. C. Sant'Anna and Qi Xu (2026), [*Difference-in-Differences with Compositional
  Changes*](https://doi.org/10.1016/j.jeconom.2025.106147).
- The authors' official [`compdid` package](https://psantanna.com/comp_did/) and
  [`drdid_stationarity_test`](https://psantanna.com/comp_did/reference/drdid_stationarity_test.html).
- The official clustered comparison uses
  [`drdid_cluster_bootstrap`](https://psantanna.com/comp_did/reference/drdid_cluster_bootstrap.html);
  CauseKit's first contracted diagnostic instead retains its existing analytical HC1/CR1
  conventions and records that difference explicitly.
