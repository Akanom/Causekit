# Development handover

This file records the active development sequence and implementation constraints so a
future task can resume without reconstructing intent from chat history.

## Completed milestones

1. `0.1.0a1`: cross-sectional `IV2SLS`, strict excluded-instrument semantics, robust and
   clustered inference, identification checks, first-stage diagnostics, and OutputHub.
2. `0.2.0a1`: two-arm `RandomizedATE`, raw difference in means, fully interacted Lin
   adjustment, balance diagnostics, and robust/clustered inference.
3. `0.3.0a1`: supplied-nuisance `IPWATE` and `AIPWATE`, overlap/weight diagnostics,
   influence-function inference, and explicit clipping.
4. `0.4.0a1`: reusable deterministic `CrossFitter`, public nuisance fit/prediction
   protocols, and ATE/ATT/ATC support.
5. `0.5.0a1` matching point alpha: supplied logit-propensity ATT/ATC/ATE matching with
   replacement, sorted scalar neighbor search, support/caliper attrition, deterministic
   fractional ties, effect/reuse weights, balance, strict refusals, deterministic recovery
   tests, and a reproducible benchmark harness.

## Active milestone

6. `0.6.0a1` DiD alpha: conventional DiD remains a separate first-class estimator with
   never-treated or not-yet-treated comparisons. The Chen–Sant'Anna–Xie efficient
   estimator is opt-in under PT-All. Both use one strict balanced-panel/treatment-timing
   validator, explicit anticipation, cohort/event/calendar effects, entity influence
   functions, HC1-style or higher-level clustered pointwise inference, and OutputHub.

   Contract and hand-computed tests were written and observed failing before `did.py`
   existed. The efficient fixture recovers weights `16/21`, `4/21`, and `1/21`, and matches
   the public R `edid` implementation at commit
   `f55a4a4aba14f0826f59ad7aa4af3bafaeba529b` for candidate effects, weights, the combined
   ATT, and HC1 standard error.

   This milestone is a no-covariate short-panel alpha, not the full paper implementation.
   Covariate-adjusted public nuisance integration, the conditional covariance estimator,
   multiplier-bootstrap simultaneous bands, pre-trend/Hausman diagnostics, repeated
   cross-sections, coverage simulations, and broader parity remain open.

7. `0.6.0a2` DiD nuisance/inference promotion: `CrossFitter` now provides reusable
   multiclass class-probability and masked scalar-regression tasks. The covariate-adjusted
   `EfficientDiD` path uses those tasks for cohort probabilities, group-specific outcome
   changes, and residual-product conditional covariances; it then solves the paper's
   observation-specific equation (3.12) systems without hidden repair. Both DiD classes
   support reproducible entity- or declared-cluster Rademacher max-t simultaneous
   event-study bands. Exact recovery, multi-moment, robust/cluster band, singular-refusal,
   and seeded coverage-smoke tests are maintained.

8. `0.6.0a3` matching known-score inference promotion: the Abadie-Imbens path now
   estimates same-arm conditional variances and applies exact ATT/ATC/ATE comparison-
   reuse formulas for a declared fixed scalar score. It reports normal inference and full
   variance audit fields, refuses estimated/cross-fitted scores and target-changing
   support/caliper selection, exports OutputHub design tables, matches pinned CRAN
   `Matching` 4.10-15 and Stata/MP 17 `teffects nnmatch`, and retains both comparator
   harnesses plus the reviewed Stata output. The 100,000-row inference scenario completes
   in linear memory without a distance matrix.

## Open promotion gates

- Matching still defaults to `inference="none"`. Known-score reuse-aware inference and
  R/Stata parity are implemented; the estimated-propensity first-step adjustment remains
  open. Do not substitute a generic sandwich, ordinary bootstrap, or cluster wrapper.
- Efficient DiD owns no nuisance model classes. The implemented covariate path must keep
  consuming public cross-fitting factories; direct density-ratio regression remains a
  possible future stability enhancement over ratios of multiclass probabilities.
- Pre-trend/Hausman diagnostics, repeated-cross-section DiD, publication-scale coverage,
  covariate-path external parity, and a larger covariate benchmark remain open.

## Required implementation patterns

- Reuse the strict alignment, labelled-result, covariance, post-estimation, OutputHub,
  release, and parity conventions established by `limiteddepkit` and `systemgmmkit`.
- Keep numerical fast paths vectorized. Normalize cluster labels once and aggregate scores
  by integer code; do not introduce observation-level Python loops or quadratic matrices.
- A fold loop is permitted where model fitting is inherently fold-specific. Each fold must
  receive fresh nuisance estimators from factories, and every reported nuisance prediction
  must be out of fold.
- Do not copy binary, count, censoring, duration, or ordinal estimators from
  `limiteddepkit`. Integrate their public fit/predict results through protocols or explicit
  prediction adapters. Move model ownership only through a separately reviewed migration.
- Do not claim that overlap diagnostics prove exchangeability, that AIPW repairs unmeasured
  confounding, or that clipping is an innocuous numerical operation.
- Every model addition requires analytical identities, deterministic recovery,
  refusal-path tests, aligned external/reference evidence where available, performance
  smoke coverage, documentation, packaging checks, and an intentional commit.

## Next development order

1. Finish matching promotion from `docs/MATCHING_CONTRACT.md`: add a separately supported
   estimated-propensity first-step contract if its required model information can be
   exposed cleanly, and extend publication-scale sensitivity/coverage evidence without
   weakening the fixed-score refusal boundaries.
2. Return to DiD for pre-trend/Hausman diagnostics, repeated cross-sections,
   publication-scale coverage, direct-ratio nuisance support if justified, and broader
   reference evidence.
3. Regression discontinuity: sharp/fuzzy design, bandwidth, polynomial order,
   manipulation checks, bias correction, and local estimand.
4. Panel IV: reuse public `systemgmmkit` panel validation, entity/time indexing, fixed
   effects, and clustered covariance contracts.

Continue to preserve the matching tie/inference, target-population, and no-quadratic-
matrix decisions. For DiD, preserve the conventional/efficient separation, PT-All label,
entity-level influence records, no-hidden-regularization refusal, and fixed-T/vectorized-n
execution path. Do not relabel either alpha as promoted merely to make an early demo run.

## Cross-software parity gate

Before a model family is release-complete, maintain estimand-aligned Python, R, and Stata
comparisons where all three ecosystems expose the same estimator. Record software and
package versions, source or fixture, option and parameter mapping, covariance corrections,
seed, tolerances, and maximum discrepancies. If an ecosystem lacks the estimator or uses
different identifying moments, record the row as non-comparable or unavailable rather
than manufacturing a parity pass. The existing no-covariate efficient DiD R comparator is
pinned; Python/R/Stata coverage for every model remains a package-wide release gate.
