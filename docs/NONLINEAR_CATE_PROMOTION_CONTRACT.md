# Native nonlinear CATE promotion contract

## Status and standard

This is the design contract for CauseKit's next nonlinear weighted-CATE stage. No new
learner is exported by this document. `NativeSplineRidgeCATE` remains the opt-in additive
spline implementation and native linear ridge-GCV remains the default until the proposed
learner passes every gate below.

There is no uniformly best CATE learner over all data-generating processes. CauseKit will
therefore use “best” to mean a preregistered best-in-class engineering and evidence
standard: orthogonal target loss, immutable honest evaluation, construction-only adaptive
selection, safe low-complexity fallbacks, multiple nonlinear function classes, bounded
runtime/memory, complete diagnostics, and competitive performance over a frozen suite
rather than one favorable dataset.

The proposed public component is `NativeOrthogonalStackedCATE`. It will implement the
existing `WeightedCATEEstimatorProtocol`; it will not become a general regression API.

## Objective and honesty boundary

For cross-fitted construction residuals

```text
u_i = Y_i - m_hat^(-k(i))(X_i)
v_i = W_i - e_hat^(-k(i))(X_i),
```

every candidate and the stack target the direct R-objective

```text
L_R(tau) = sum_i [u_i - v_i tau(X_i)]^2.
```

The equivalent transformed regression uses response `u_i / v_i` and weight `v_i^2`, but
reported loss is always reconstructed on the direct scale. Near-zero `v_i` values receive
their algebraic small weight; no unstable pseudo-outcome is treated as an equally weighted
target.

The R-learner's evaluation role remains untouched until final loss, calibration, and group
assessment. Candidate construction, basis discovery, screening, tuning, stacking, and
refitting never inspect evaluation outcomes, treatments, nuisance residuals, or metrics.
Whole declared clusters retain one immutable role and one fold wherever clustering is
used.

## Candidate library

Every fit includes nested candidates so complexity can collapse safely:

1. a constant weighted R-projection;
2. standardized linear ridge-GCV;
3. feature-adaptive additive hinge splines with zero-knot linear candidates;
4. strong-heredity pairwise hinge interactions whose parent main effects are retained; and
5. a bounded seeded random-feature kernel candidate for smooth non-additive structure,
   promoted only after a separate approximation and reproducibility test.

The first implementation phase stops after candidate 4. Candidate 5 is a later gate, not
a placeholder. CauseKit will not copy a random forest, gradient booster, neural network,
or another package's general-purpose estimator into the repository. Those models remain
optional benchmark comparators through the public protocol.

Candidate bases are derived inside each candidate-training fold. Feature-wise knots come
from weighted training quantiles. Interaction screening uses only training-fold direct
R-score correlations and must obey strong heredity. The same selection is repeated from
scratch in each fold; a full-construction refit occurs only after stack weights are fixed.

Hard limits cover input features, knots per feature, screened interactions, total basis
columns, and candidate count. Limit violations refuse with an estimated required size;
they do not silently drop columns. Matrix construction remains `O(n p_basis)` and no
`n by n` kernel or distance matrix is permitted.

## Cross-fitted orthogonal stacking

A shared, deterministic inner fold plan on the construction role produces out-of-fold
predictions from every candidate. Let `tau_hat_ik` be candidate `k`'s prediction for row
`i` from a model that did not train on row `i`. Stack weights solve

```text
minimize_a  sum_i [u_i - v_i sum_k a_k tau_hat_ik]^2
subject to  a_k >= 0 and sum_k a_k = 1.
```

The simplex contains every single candidate, so the solved construction OOF objective
must be no larger than the best candidate OOF objective up to a declared numerical
tolerance. The solver uses a deterministic convex method with KKT residual diagnostics;
failure to meet feasibility or optimality tolerance refuses. There is no unconstrained
negative extrapolation and no post-hoc evaluation tuning.

After selecting weights, every candidate with nonzero weight is freshly refit on all
construction rows using only its construction-selected configuration. Future and honest
evaluation predictions use the fixed weighted sum. Exact zero weights are retained in the
audit table rather than deleting evidence about rejected candidates.

## Public diagnostics

The result must expose:

- immutable construction and evaluation indices or clusters;
- shared inner fold assignments;
- per-fold and full-refit candidate basis schemas;
- knots, interaction parents, screening scores, penalties, dimensions, and condition
  numbers;
- candidate OOF direct R-loss, loss standard error by fold, runtime, and peak memory;
- stack weights, KKT residuals, simplex feasibility, stacked OOF loss, and best-single-
  candidate loss;
- final full-construction refit metadata;
- unchanged honest R-loss, constant comparison, differential calibration, groups,
  influence records, and simultaneous bands from the public `RLearner`; and
- exact graph data with no plotting-time refit.

The output must distinguish model-selection diagnostics from evaluation evidence. A
selected nonlinear basis does not establish nonlinear CATE truth, and lower R-loss does
not prove causal identification.

## Refusals

The learner will refuse evaluation leakage, mutable role or fold assignments, malformed
weighted-CATE providers, non-finite inputs, non-binary treatment at the R-learner boundary,
overlap failure, folds missing an arm, basis-size overflow, unsupported categorical raw
inputs, interaction requests without parent main effects, singular unpenalized columns,
invalid weights, non-converged simplex optimization, inconsistent prediction schemas,
cluster splitting, and any request to select candidates from evaluation loss.

No prediction clipping, evaluation-based early stopping, hidden ridge, pseudoinverse,
silent basis deletion, or automatic external-ML fallback is allowed.

## Promotion suite

Tests are written and observed failing before implementation:

1. hand direct/transformed R-loss and simplex-stacking identities;
2. exact constant, linear, additive-kink, and interaction recovery fixtures;
3. KKT, non-negativity, sum-to-one, best-candidate containment, schema, and deterministic-
   seed identities;
4. row and whole-cluster leakage logs proving that every candidate and screening step is
   construction-only and out of fold;
5. refusal tests for every limit and malformed optimizer/provider path;
6. null, linear, additive, interaction, weak-overlap, irrelevant-feature, and high-noise
   simulations with frozen seeds and thresholds;
7. semisynthetic benchmarks using hash-pinned real covariate distributions and known
   nonlinear CATE functions, so pointwise ranking and integrated CATE error are observable;
8. real randomized-outcome evidence on Hillstrom visit and conversion, retaining negative
   results and honest R-loss rather than treating the realized outcome as known CATE;
9. one-run, identically split comparisons against linear ridge, the existing spline,
   histogram boosting, random forest, and an aligned generalized random forest where
   available, with no runtime dependency;
10. repeated-split robustness as a validation artifact, while the public single-split
    result remains explicit;
11. fixed-sample base-R identities for the stack and reviewed Stata algebra where its
    matrix language can reproduce the same fixed candidate predictions; unavailable
    learner-fit parity is recorded honestly;
12. 100,000-row fixed-feature and increasing-feature performance smokes; and
13. full API, OutputHub, documentation, lint, format, type, build, dependency, and security
    checks.

## Preregistered performance decisions

Promotion requires all correctness and leakage gates, not merely a leaderboard win. On
the frozen nonlinear simulation/semisynthetic suite, the stack must improve mean normalized
integrated CATE error and direct R-loss over native linear ridge, and must not regress by
more than the preregistered tolerance on constant or linear designs. On real randomized
outcomes, results are descriptive honest loss/calibration evidence: a loss does not become
a pass because the basis is more complex.

External comparators are evaluated under identical construction/evaluation roles, nuisance
predictions, covariates, seeds, and resource measurement. Conclusions report per-design
wins, median regret, uncertainty across registered replications, runtime, and memory. No
claim of universal superiority, “state of the art,” or “best” is permitted from one dataset.

The current Hillstrom evidence illustrates the rule. Visit selected the zero-knot fallback.
Conversion selected one knot, but spline honest R-loss was 0.0019% worse than linear and
both were worse than the construction-fitted constant. These are useful negative tests,
not reasons to tune against the evaluation role.

## Alternatives considered

- Making splines more complex globally was rejected because one knot count for every
  feature cannot adapt selectively and can inflate variance.
- Selecting the best candidate directly on honest evaluation was rejected as leakage.
- Shipping a forest copy was rejected because it duplicates mature general ML software
  and would create a maintenance surface unrelated to CauseKit's causal contracts.
- Choosing a single external learner as the default was rejected because it creates a
  runtime dependency and no learner dominates across smooth, sparse, and interaction
  designs.
- Unconstrained stacking was rejected because negative weights can extrapolate sharply and
  remove the best-single-candidate containment guarantee.

## Pre-mortem

Likely failure modes are meta-overfitting to too many candidates, leakage during knot or
interaction discovery, quadratic basis growth, weak-overlap pseudo-outcome instability,
and claiming success from noisy real-outcome rankings. The contract controls these with a
small frozen library, nested fold-local discovery, strong heredity and hard dimensions,
the direct weighted R-objective, immutable evaluation, semisynthetic known-truth gates,
and explicit negative-result reporting.

## Primary methodology

- Xinkun Nie and Stefan Wager (2021), [*Quasi-Oracle Estimation of Heterogeneous Treatment
  Effects*](https://doi.org/10.1093/biomet/asaa076).
- Mark J. van der Laan, Eric C. Polley, and Alan E. Hubbard (2007), [*Super
  Learner*](https://doi.org/10.2202/1544-6115.1309).
- Susan Athey, Julie Tibshirani, and Stefan Wager (2019), [*Generalized Random
  Forests*](https://doi.org/10.1214/18-AOS1709).
