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

## Continued completed milestones

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

9. `0.6.0a4` matching estimated-Logit inference promotion: the public
   `FittedPropensityMLEProtocol` consumes a provider-neutral regular full-sample
   unpenalized Logit result. CauseKit validates fit/sample/schema/score/information
   identities and applies separate Abadie–Imbens ATE/ATT/ATC first-step
   corrections. Hand contracts, strict refusals, independent `statsmodels` parity,
   seeded coverage smoke, OutputHub audit fields, and a 100,000-row 85.20 MiB benchmark
   are implemented. Reviewed Stata/IC 17 parity
   establishes that `vce(robust, nn(2))` maps to CauseKit's one-neighbor known-score
   variance and two leave-own-out local first-step regression neighbors. ATT/ATC/ATE
   point estimates, variance components, and final standard errors pass at `1e-8`.

10. `0.6.0a4` ownership and real-data release gate: the unreleased distribution/import
    identity is now `causekit`, with no shim under the occupied `causalkit` name. CauseKit
    owns all migrated causal models and has no LimitedDepKit runtime or validation
    dependency. Four opt-in HTTPS datasets are pinned by SHA-256, a runnable example covers
    every model family, and the real-data Python/R certificate passes 14 estimand-aligned
    comparisons. The reviewed Stata/IC 17 saved-output certificate also passes every
    comparable family; efficient PT-All DiD remains explicitly unavailable in Stata.

11. `0.7.0a1` native causal-ML alpha: `PartiallyLinearDML` implements the pooled DML2
    score with shared `CrossFitter` folds, a CauseKit-owned standardized ridge-GCV default
    fitted separately inside every outer training fold, HC1/one-way CR1 inference, strict
    residual-treatment identification, full OOF audit records, OutputHub adaptation, and
    a hash-verified Cattaneo real-data workflow. Hand, refusal, simulation, Statsmodels,
    base-R 4.5.1, and manually executed Stata/IC 17 contracts pass. A one-run real-data
    performance comparison records native and optional external learner results without
    adding an ML dependency.

12. `0.7.0a2` ridge audit and heterogeneous-effect design: the native GCV path now uses
    SVD shrinkage identities without fitted-vector reconstruction. A denser grid lowered
    training GCV but slightly worsened the same real-data OOF errors and was rejected as
    the default. `NuisanceDiagnosticsProtocol` carries scalar per-task/fold
    tuning records through CrossFitter, DML results, and OutputHub. The five-fold Cattaneo
    smoke was revalidated without rerunning external comparators. The honest R-learner
    construction/evaluation, R-loss, calibration, grouping, uncertainty, graphing, and
    refusal contract is recorded; no placeholder estimator is exported.

    A separate one-run NSW job-training benchmark now exercises the same scalar DML path
    on 445 observations and eight pre-treatment covariates. CauseKit native ridge-GCV had
    the lowest outcome OOF RMSE (6608.467270), runtime (0.0456 seconds), and
    Python-managed peak memory (0.210 MiB); fold-local standardized scikit-learn RidgeCV
    had a 0.180% lower treatment OOF RMSE. Histogram boosting and random forest were worse
    on both nuisance targets. This supports keeping native ridge-GCV as the efficient
    default, not a universal superiority or CATE claim. The Cattaneo rows were not rerun.

13. `0.7.0a2` R-learner prerequisites: public `WeightedCATEEstimatorProtocol` and
    `CATEResultProtocol` define the provider-neutral weighted-fit/prediction boundary.
    Internal native components implement standardized ridge-penalized binary Logit with
    deterministic stratified training-only CV, plus weighted ridge-GCV whose transformed
    loss is exactly the direct R-objective. Both retain construction indices and tuning
    diagnostics. Hand score/objective, integration, leakage-audit, weight, schema, index,
    malformed-provider, and prediction refusals pass. No `RLearner` placeholder is
    exported; honest splitting and every evaluation/promotion gate remain open.

## Open promotion gates

- Matching still defaults to `inference="none"`. Known-score reuse-aware inference and a
  separately validated full-sample Logit-MLE correction are implemented. Generic,
  cross-fitted, penalized, and unsupported-link scores still refuse. Fixed-score R/Stata
  parity is recorded, including the estimated-score Stata decomposition; R `Matching`
  is non-comparable because it conditions on the supplied score. Do not substitute a
  generic sandwich, ordinary bootstrap, or cluster wrapper.
- Efficient DiD owns no nuisance model classes. The implemented covariate path must keep
  consuming public cross-fitting factories; direct density-ratio regression remains a
  possible future stability enhancement over ratios of multiclass probabilities.
- Pre-trend/Hausman diagnostics, repeated-cross-section DiD, publication-scale coverage,
  and a larger covariate benchmark remain open. Covariate-efficient external parity was
  audited: the pinned public R implementation has no covariate path and reviewed Stata
  estimators target different moments, so those cells remain explicitly unavailable.
- R/DR learners and heterogeneous-effect diagnostics/graphs are not yet implemented. The
  R-learner contract and native probability/weighted-CATE prerequisites are complete, but
  honest role splitting, cross-fitted CATE construction, leakage enforcement at the full
  estimator boundary, evaluation, calibration, groups, bands, graphs, simulations,
  parity, and real-data promotion remain open. The partially linear DML residual-stage
  parity row passes in Python, base R 4.5.1, and a reviewed manual Stata/IC 17 run.

## Required implementation patterns

- Reuse the strict alignment, labelled-result, covariance, post-estimation, OutputHub,
  release, and parity conventions established by `limiteddepkit` and `systemgmmkit`.
- Keep numerical fast paths vectorized. Normalize cluster labels once and aggregate scores
  by integer code; do not introduce observation-level Python loops or quadratic matrices.
- A fold loop is permitted where model fitting is inherently fold-specific. Each fold must
  receive fresh nuisance estimators from factories, and every reported nuisance prediction
  must be out of fold.
- Do not copy binary, count, censoring, duration, or ordinal estimators into CauseKit.
  General nuisance integration stays provider-neutral through protocols or explicit
  prediction adapters. The small native ridge-GCV learner is owned only as the default
  component of `PartiallyLinearDML`, not as a general regression surface.
- LimitedDepKit is owned by a separate agent/location. Do not inspect, edit, test, build,
  commit, clean, or release that repository from this CauseKit worktree.
- Do not claim that overlap diagnostics prove exchangeability, that AIPW repairs unmeasured
  confounding, or that clipping is an innocuous numerical operation.
- Every model addition requires analytical identities, deterministic recovery,
  refusal-path tests, aligned external/reference evidence where available, performance
  smoke coverage, documentation, packaging checks, and an intentional commit.

## Next development order

1. Implement the honest R-learner construction/evaluation split and cross-fitted
   R-objective using the completed native probability/weighted-CATE prerequisites.
2. Implement R-loss, differential calibration,
   group bands, and graph-data surface; run simulation/parity/real-data gates before
   promotion.
3. Add the doubly robust learner only after its propensity, pseudo-outcome, honest second-
   stage, and uncertainty contracts are settled.
4. Continue matching promotion from `docs/MATCHING_CONTRACT.md`: extend publication-scale
   sensitivity/coverage evidence without weakening either analytical refusal boundary.
5. Return to DiD for pre-trend/Hausman diagnostics, repeated cross-sections,
   publication-scale coverage, direct-ratio nuisance support if justified, and broader
   reference evidence.
6. Regression discontinuity: sharp/fuzzy design, bandwidth, polynomial order,
   manipulation checks, bias correction, and local estimand.
7. Panel IV: reuse public `systemgmmkit` panel validation, entity/time indexing, fixed
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
