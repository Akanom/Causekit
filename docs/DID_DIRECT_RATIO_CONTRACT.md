# Direct cohort-ratio nuisance contract for efficient DiD

## Status and boundary

This is a design-only contract. It freezes the estimand, nuisance semantics,
`CrossFitter` operations, diagnostics, refusals, and promotion evidence required before
code is added. CauseKit does not currently export a direct-ratio protocol or accept direct
ratio predictions in `EfficientDiD`. The implemented covariate-adjusted PT-All path
continues to construct cohort ratios from one out-of-fold multiclass probability matrix.

The first target is the covariate-adjusted `EfficientDiD` score of Chen, Sant'Anna, and
Xie. Direct ratios are an alternative nuisance route, not a new estimator. They do not
relax PT-All, no anticipation, cohort overlap, outcome or conditional-covariance nuisance
conditions, or the positive-definite efficient-weight system. They do not authorize
composition-robust repeated sections or survey weighting, which have separate contracts.

## Required statistical object

For target cohort `g` and denominator cohort `h`, the maintained score requires the
calibrated conditional cohort-odds ratio

```text
rho_g:h(X) = P(G=g | X) / P(G=h | X).
```

This is also the odds of `G=g` in the binary population restricted to cohorts `{g,h}`.
It is not merely the group-conditional density ratio
`f(X|G=g)/f(X|G=h)`: Bayes' rule relates the two by the cohort prior odds
`P(G=g)/P(G=h)`. A provider that estimates conditional densities must apply and document
that calibration before returning predictions. CauseKit will not guess a missing prior,
infer orientation from column order, or normalize a raw ratio until a score happens to
fit.

For candidate `(g_prime, t_pre)`, equation (4.4) uses exactly

```text
rho_g:never(X)
rho_g:g_prime(X).
```

Unlike Hájek-normalized stationary repeated-section weights, these score terms are divided
by the target cohort share `pi_g` and are not normalized by their own sample means.
Multiplying a returned ratio by an arbitrary positive constant generally changes the
estimand. The implementation tests this sensitivity rather than promising scale
invariance.

The conditional efficient-weight covariance can avoid absolute multiclass probabilities.
Every element of the current `Omega_i` contains a common factor `1/p_g(X)`. Define

```text
Omega_tilde_i = p_g(X) * Omega_i.
```

Then its components use the treated conditional variance, `rho_g:never` times never-
treated conditional covariances, and `rho_g:g_prime` times the relevant auxiliary-cohort
covariances. Because

```text
w_i = solve(Omega_i, 1) / [1' solve(Omega_i, 1)]
    = solve(Omega_tilde_i, 1) / [1' solve(Omega_tilde_i, 1)],
```

the common positive factor cancels in the efficient weights. This algebra must be a hand
identity before the multiclass matrix is optional.

## Public nuisance protocol

The implementation phase will first add a provider-neutral runtime-checkable prediction
protocol:

```python
class CohortOddsRatioResultProtocol(Protocol):
    def predict_odds_ratio(self, X: object) -> object: ...
```

Fitting continues through the existing `NuisanceEstimatorProtocol.fit(X, y)`. For an
ordered pair `(g,h)`, `CrossFitter` supplies only rows from those two cohorts, with `y=1`
meaning numerator cohort `g` and `y=0` denominator cohort `h`. The prediction must be the
held-out posterior odds for that exact orientation. A factory, not a fitted singleton, is
supplied so every pair and outer fold receives a fresh object.

The planned `CrossFitter` extension is an optional `cohort_ratio_factory` and, if needed,
an explicit `cohort_ratio_predict` adapter. Exactly one cohort-weighting route is active:
the existing `propensity_factory` for a multiclass probability matrix or the new pairwise
ratio factory. There is no data-dependent fallback. `outcome_factory` and
`second_moment_factory` retain their current meanings.

CauseKit owns the ordered-pair task graph, alignment, ratio validation, fold plan, and
audit records; it will not copy a density-ratio learner from another package. Provider
diagnostics may be retained through `NuisanceDiagnosticsProtocol`, but no provider class
becomes the CauseKit default merely to enable the protocol.

The fitted result adds `nuisance_weighting="direct_cohort_odds"` and a labelled
`cohort_ratios` frame with ordered `(numerator, denominator)` columns. Its existing
`cohort_probabilities` frame is empty on this route rather than populated with fabricated
probabilities. Summary and OutputHub adapters must preserve that distinction.

## Cross-fitting and immutable roles

The existing entity-level outer-fold assignment is shared by all ratio, outcome-change,
and residual-product tasks for the fit. Declared higher-level clusters stay whole. For
each unique ordered pair needed across all candidates, the ratio model is fitted only on
the two cohorts in an outer training partition and predicts every outer-holdout entity.
Inner tuning may use only that outer training partition.

The ratio is predicted for every outer-holdout entity because the refactored conditional
efficient weights are observation-specific, including for entities outside the fitted
pair. Those predictions are mathematically used, not cosmetic extrapolations, and require
common covariate support. The pair orientation, target and denominator cohorts, candidate
cells, entity order, training and holdout hashes, fold, and cluster roles are retained.
Reversing a requested pair must either fit the reversed task or take the checked
reciprocal of an already validated ratio; it cannot relabel the same predictions silently.

## Support diagnostics

Every ordered pair reports raw ratio minimum, median, maximum and upper quantiles;
log-ratio quantiles; numerator and denominator counts and PSUs by fold; denominator-row
importance effective sample size; and maximum normalized importance share. Diagnostics
are reported by pair and candidate use so a good common summary cannot hide a failed
auxiliary cohort.

Direct estimation does not solve overlap. Scale-sensitive ratio and scale-invariant
concentration gates are both required: calibrated ratio bounds diagnose conditional odds,
while normalized shares and effective sample size diagnose dominance. Public threshold
names and defaults must be preregistered in failing tests and publication simulations.
CauseKit will not clip, winsorize, trim, renormalize, or replace a failed direct ratio with
multiclass probabilities.

## Required refusals

The direct-ratio path must refuse:

- simultaneous multiclass propensity and direct-ratio factories, or absence of either
  weighting route when covariates are requested;
- a provider without `fit(X,y)` and `predict_odds_ratio(X)` behavior;
- a factory that returns the same estimator instance across calls;
- an unknown numerator/denominator orientation or an uncalibrated group-conditional density
  ratio;
- prediction length, index, task, pair, column, or fold drift;
- missing, nonnumeric, complex, nonfinite, zero, or negative ratios;
- own-entity predictions, inner tuning on holdout rows, or mutable entity/cluster roles;
- a training fold missing either cohort in any required ordered pair;
- failure of a preregistered ratio, normalized-share, effective-size, or PSU-support gate;
- a refactored conditional-covariance system that is nonfinite, nonpositive-definite, or
  inconsistent with its pair ratios; and
- attempts to pass the PT-All ratio protocol into repeated-section, matching, survey, or
  causal-ML scores without a separately derived adapter and contract.

No numerical repair or fallback may turn these conditions into warnings.

## Failing-first implementation gates

Before runtime code is written, tests must fail for:

1. a hand-computed PT-All candidate in which direct cohort odds reproduce the existing
   multiclass-ratio point score, candidate influence, and target effect;
2. the `Omega`/`Omega_tilde` solve identity, conditional efficient weights, full efficient
   influence, HC1/CR1 covariance, and event aggregation;
3. a calibrated toy example distinguishing posterior cohort odds from a raw conditional-
   density ratio and its omitted prior-odds error;
4. reciprocal orientation, row permutation, labelled-index, and shared-fold identities;
5. fresh-factory, entity-fold, higher-level-cluster, and inner-tuning leakage sentinels;
   and
6. every invalid prediction, weak-support, and unsupported-combination refusal above.

Promotion then requires nonlinear and weak-overlap designs comparing direct ratios,
multiclass probability ratios, and oracle cohort odds on identical folds and nuisance
factories where meaningful. Preregistered gates cover candidate and aggregate bias,
empirical-to-analytical SE calibration, pointwise and simultaneous coverage, conditional
weight recovery, support, refusal rate, runtime, and peak memory. The direct route must
show a documented stability, calibration, or performance benefit in at least one frozen
design without materially degrading the other gates; novelty alone is insufficient.

A hash-pinned real-data sensitivity workflow reports both routes under the same analysis
sample, folds, candidates, outcome/second-moment nuisances, and seeds. It cannot choose a
route after inspecting the target estimate. Independent base-R reconstruction is
required. R/Stata package parity is reported only when a maintained command accepts the
same direct pairwise cohort-odds nuisances and PT-All moment; otherwise the cell is
explicitly unavailable.

## Not covered by this contract

A stationary repeated-section score is insensitive to a common positive rescaling of its
treated-to-comparison density ratio because each importance component is
Hájek-normalized. That is a different operation with different diagnostics and scale
behavior. It is not exported by this contract. If later justified, it receives a small
separate adapter contract after this PT-All route, rather than overloading
`predict_odds_ratio` with two meanings.

## Pre-mortem

The leading failure is accepting an uncalibrated density ratio where the PT-All score
requires posterior cohort odds. An omitted cohort prior can produce a clean-looking but
wrong effect. Ordered pair metadata, the prior-odds hand example, and scale-sensitive
tests address it. Other risks are quadratic pair-task growth, holdout leakage through
provider tuning, and refactoring `Omega` incorrectly. Unique-pair planning, fold hashes,
fixed-T/vectorized-n benchmarks, and exact `Omega_tilde` identities are promotion gates.

## Primary methodology

- Xiaohong Chen, Pedro H. C. Sant'Anna, and Haitian Xie (2025), [*Efficient
  Difference-in-Differences and Event Study Estimators*](https://arxiv.org/abs/2506.17729),
  arXiv:2506.17729.
- Yuta Tsuboi, Hisashi Kashima, Shohei Hido, Steffen Bickel, and Masashi Sugiyama
  (2009), [*Direct Density Ratio Estimation for Large-scale Covariate Shift
  Adaptation*](https://doi.org/10.2197/ipsjjip.17.138), for direct ratio estimation as a
  nuisance strategy rather than separate density estimation.
