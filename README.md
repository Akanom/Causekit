# CauseKit

CauseKit (installed and imported as `causekit`) is an identification-aware Python package for causal inference and
instrumental-variable workflows. The `0.7.0a3` surface provides linear two-stage least
squares, randomized-experiment effects, reusable nuisance cross-fitting, IPW/AIPW ATE,
ATT, and ATC, scalar propensity-score matching with separate fixed- and estimated-score
analytical inference paths, conventional staggered DiD, and cross-fitted covariate-adjusted
Chen-Sant'Anna-Xie efficient DiD for short panels, and CauseKit-native partially linear
double machine learning plus separately contracted honest heterogeneous-effect R- and
DR-learning.

This is alpha research software. A successful fit is not evidence that an instrument is
valid, and an IV coefficient is not automatically an average treatment effect. State the
estimand and defend the identifying assumptions before using causal language.

The project was renamed before its first release because the intended `causalkit`
distribution name is already used by an unrelated project. CauseKit does not install a
`causalkit` compatibility namespace; use `pip install causekit` and `import causekit`.

## Current scope

The retained IV API provides:

- `IV2SLS` with one or more endogenous regressors and excluded instruments;
- an excluded-instrument API: exogenous regressors are added to the instrument matrix
  internally and must not be repeated in `instruments`;
- homoskedastic (`"unadjusted"`), HC1 (`"robust"`), and one-way cluster-robust CR1
  (`"clustered"`) covariance estimators;
- strict missing-value, non-finite-value, index-alignment, shape, name, and rank checks;
- classical and covariance-aware first-stage excluded-instrument diagnostics; and
- Sargan's overidentification test only for overidentified fits using
  `covariance="unadjusted"`.

The `0.2.0a1` randomized-experiment layer adds:

- `RandomizedATE(adjustment="none")` for the raw difference in arm means;
- `RandomizedATE(adjustment="lin")` for fully interacted, mean-centered Lin adjustment;
- HC1 or one-way CR1 inference, arm counts, balance diagnostics, strict binary assignment,
  and explicit causal-assumption metadata.

The `0.3.0a1` observational layer adds `IPWATE` and `AIPWATE`. It deliberately accepts
propensity and potential-outcome predictions through provider-neutral public boundaries
rather than owning nuisance-model likelihoods.

```python
from causekit import AIPWATE

result = AIPWATE().fit(
    outcome,
    treatment=treated,
    propensity=cross_fitted_propensity,
    outcome_treated=cross_fitted_mu1,
    outcome_control=cross_fitted_mu0,
)
```

For adaptive nuisance models, predictions should be cross-fitted: every observation's
prediction must come from a model that did not train on that observation. AIPW is doubly
robust to one nuisance family being correctly specified under regularity conditions; it
is not robust to unmeasured confounding, positivity failure, leakage, or both nuisance
families being invalid. Propensity clipping is never silent and requires sensitivity
reporting because it changes the estimating equation.

### Reusable cross-fitting

`CrossFitter` accepts factories so every fold receives fresh models. A model may return
an immutable fitted result or follow the scikit-learn convention where `fit` returns the
estimator. Default fitted results expose `predict_proba` for propensity models and
`predict` for outcome models; explicit adapters support other public APIs.

```python
from causekit import AIPWATE, CrossFitter

nuisance = CrossFitter(
    propensity_factory=make_propensity_model,
    outcome_factory=make_outcome_model,
    n_splits=5,
    random_state=2026,
).fit_predict(X, treatment=treated, outcome=outcome)

att = AIPWATE(estimand="att").fit(
    outcome,
    treatment=treated,
    propensity=nuisance.propensity,
    outcome_treated=nuisance.outcome_treated,
    outcome_control=nuisance.outcome_control,
)
```

Folds are stratified by treatment, predictions retain the original index, and each arm
must contain at least `n_splits` observations. Fold assignment is deterministic when
`random_state` is fixed. Supplying `clusters=` keeps every cluster wholly within one fold
and refuses any allocation that cannot retain every treatment stratum in every fold.

The same orchestrator exposes multiclass class-probability prediction and masked scalar
regression tasks. Those operations let panel estimators request cohort-specific outcome
changes and conditional second moments without owning or copying model implementations.
Every task receives a fresh model per fold. A separate `second_moment_factory=` is
optional; when omitted, the outcome factory is reused.

### Native causal machine learning

`PartiallyLinearDML` estimates the DML2 orthogonal-score coefficient for a binary or
continuous scalar treatment. Its default nuisance path is CauseKit's own dependency-free,
standardized ridge learner with generalized-cross-validation selection nested separately
inside each outer training fold.

```python
from causekit import PartiallyLinearDML

dml = PartiallyLinearDML(
    n_splits=5,
    random_state=2026,
    covariance="robust",
).fit(
    outcome,
    treatment=treatment,
    covariates=baseline_covariates,
)

print(dml.summary_frame())
print(dml.nuisance_predictions)
print(dml.nuisance_diagnostics)
print(dml.residual_treatment_second_moment)
```

CauseKit imports no third-party ML implementation for this path. Optional outcome and
treatment factories can replace either native nuisance learner when substantively
necessary, while `CrossFitter` retains the shared out-of-fold plan. The reported `theta`
is an ATE only under a credible constant-effect partially linear model, consistency, no
interference, conditional exchangeability, residual treatment variation, and the DML
nuisance-rate/regularity conditions. See the [causal-ML contract](docs/ML_CONTRACT.md).
The native learner reports one tuning row per task and outer fold, including its selected
penalty, effective degrees of freedom, GCV score, training RMSE, numerical rank, grid size,
and whether selection reached a grid boundary.

`RLearner` estimates heterogeneous effects through the residualized R-objective while
keeping construction and evaluation roles honest. Rows are split within treatment arms;
with `covariance="clustered"`, whole clusters are assigned to roles and outer folds. Only
construction data tune or fit the outcome, propensity, and weighted CATE learners.

```python
from causekit import RLearner

rlearner = RLearner(
    n_splits=5,
    evaluation_fraction=0.5,
    random_state=2026,
    calibration_groups=5,
    bootstrap_iterations=999,
).fit(
    outcome,
    treatment=treated,
    covariates=X,
)

print(rlearner.summary_frame())          # differential calibration
print(rlearner.honest_r_loss)
print(rlearner.honest_constant_r_loss)
print(rlearner.calibration_plot_data())  # pointwise and simultaneous group bands
```

The R-loss gain compares the CATE learner with a constant effect fitted only on
construction; it is not predictive R-squared. Calibration groups preserve score ties and
report overlap-weighted residual-moment effects, not ordinary group ATEs. The first public
contract provides no unit-level CATE interval, RATE, policy value, or repeated-split
aggregation. `plot_calibration()` and `plot_cate_distribution()` are available through the
optional `plot` extra; both use the exact retained honest tables.

The R-loss is an established method, not a new CauseKit model. CauseKit's differentiation
is its integrated honest-role, audit, refusal, cluster, calibration, and simultaneous-band
contract; the [full contract](docs/R_LEARNER_CONTRACT.md) avoids unsupported claims of
global algorithmic novelty.

For nonlinear CATEs, `NativeSplineRidgeCATE` is an opt-in weighted final stage:

```python
from causekit import NativeSplineRidgeCATE

nonlinear = RLearner(
    cate_factory=NativeSplineRidgeCATE,
    random_state=2026,
).fit(outcome, treatment=treated, covariates=X)

print(nonlinear.cate_diagnostics)  # selected knots, basis size, alpha, weighted GCV
```

It selects zero-, one-, or three-knot additive linear-spline bases and the ridge penalty
using construction-only weighted GCV, then evaluates once on the honest role. Linear
ridge-GCV remains the default: the nonlinear learner recovers a known piecewise effect in
simulation, safely matches linear performance on the Hillstrom randomized-email data, and
is worse on the separate NSW split. Pairwise interactions are explicit opt-in and a strict
basis-size ceiling prevents accidental feature explosion.

`DRLearner` is a separately contracted heterogeneous-effect estimator. It cross-fits the
propensity and both treatment-arm outcome regressions inside construction, forms the
augmented inverse-probability pseudo-outcome without clipping, and fits an unweighted CATE
regression. The evaluation role remains untouched by every fit and tuning operation.

```python
from causekit import DRLearner

drlearner = DRLearner(
    n_splits=5,
    evaluation_fraction=0.5,
    random_state=2026,
    calibration_groups=5,
    bootstrap_iterations=999,
).fit(outcome, treatment=treated, covariates=X)

print(drlearner.summary_frame())
print(drlearner.honest_dr_loss, drlearner.honest_constant_dr_loss)
print(drlearner.calibration_plot_data())
```

The score is doubly robust only in the precise sense that its conditional mean is correct
when the propensity is correct or both arm outcome regressions are correct, alongside the
documented identification and rate conditions. It does not repair unmeasured confounding.
HC1/CR1 calibration, tie-preserving mean-score groups, and max-t group bands are
split-conditional; no unit-level interval, RATE, or policy-value claim is exposed. See the
[honest DR-learner contract](docs/DR_LEARNER_CONTRACT.md).

On one hash-verified NSW split, native ridge-GCV and optional scikit-learn RidgeCV produced
the same displayed CATE predictions and honest DR loss; the native path used less
Python-managed peak memory, while RidgeCV was slightly faster. Boosting and random forest
were worse than the construction-fitted constant. Ridge-GCV is therefore the auditable,
dependency-free default, not a claimed novel ridge method or universal winner.

### Nearest-neighbor matching

`NearestNeighborMatch` consumes a supplied propensity through a provider-neutral boundary.
The implemented slice supports ATT, ATC, and bidirectional-imputation ATE;
logit-propensity distance; replacement; inclusive numeric or automatic calipers;
intersection common support; deterministic fractional boundary ties; effect/reuse weights;
and before/after covariate balance.

```python
from causekit import NearestNeighborMatch

matched = NearestNeighborMatch(
    estimand="att",
    inference="none",
).fit(
    outcome,
    treatment=treated,
    propensity=nuisance.propensity,
    covariates=X,
    propensity_provenance="CrossFitter:5-fold",
)

print(matched.summary_frame())
print(matched.balance)
print(matched.match_table)
```

`inference="none"` remains the default and is required for arbitrary estimated or
cross-fitted propensities. When the score is genuinely known/fixed by design, the
maintained Abadie-Imbens path is explicit:

```python
fixed_score_match = NearestNeighborMatch(
    estimand="att",
    caliper=None,
    common_support=None,
    inference="abadie_imbens",
    variance_neighbors=1,
).fit(
    outcome,
    treatment=treated,
    propensity=known_propensity,
    propensity_score_status="known",
    propensity_provenance="fixed_by_declared_design",
)

print(fixed_score_match.summary_frame())
print(fixed_score_match.conditional_variances)
```

For a regular full-sample unpenalized Logit MLE, use the separate provider-neutral
fitted-result protocol:

```python
estimated_score_match = NearestNeighborMatch(
    estimand="att",
    metric="propensity",
    caliper=None,
    common_support=None,
    inference="abadie_imbens_estimated",
    variance_neighbors=1,
).fit(
    outcome,
    treatment=treated,
    propensity_model=propensity_fit,
    propensity_design=propensity_design,
    propensity_score_status="estimated",
    propensity_provenance="full_sample_unpenalized_logit_mle",
)
```

CauseKit does not fit `propensity_fit`. It requires public `params`, `converged`, `nobs`,
`feature_names`, and `predict_proba`, then validates sample size, feature/parameter order,
fitted Logit probabilities, the likelihood first-order condition, and Fisher-information
conditioning.
It then reports the known-score variance and Abadie–Imbens first-step adjustment separately.
Cross-fitted, penalized, probit, or otherwise unverifiable scores do not satisfy this
contract and must retain `inference="none"`.

The fixed-score path uses same-arm nearest neighbors to estimate conditional outcome variances and
accounts for comparison reuse for ATT, ATC, and ATE. It refuses caliper/support selection,
expanded cross-arm or same-arm boundary ties, non-fixed scores, and inadequate same-arm
samples. Both analytical paths refuse target-changing support/caliper selection. A descriptive provenance string does not activate inference; the machine-readable
score status is separate. Ordinary bootstrap, matching without replacement, arbitrary tie
selection, clustered uncertainty, and unimplemented bias correction also refuse. See the
[matching contract](docs/MATCHING_CONTRACT.md) before publication-facing use.

### Conventional and efficient difference-in-differences

The conventional estimator remains first-class. It reports cohort-time effects using a
never-treated comparison by default, or adds valid not-yet-treated entities when requested.
The efficient estimator is a separate PT-All procedure: it uses all admissible
pre-treatment periods and auxiliary cohorts to estimate the Chen-Sant'Anna-Xie
inverse-covariance weights. It is not a silent default because PT-All is stronger than the
conventional post-treatment parallel-trends contract.

```python
from causekit import DifferenceInDifferences, EfficientDiD

conventional = DifferenceInDifferences(
    control_group="never_treated",
    covariance="robust",
).fit(
    panel,
    outcome="outcome",
    entity="unit",
    time="period",
    treatment_time="first_treated",
)

cross_fitter = CrossFitter(
    propensity_factory=make_multiclass_cohort_model,
    outcome_factory=make_outcome_model,
    second_moment_factory=make_second_moment_model,
    n_splits=5,
    random_state=2026,
)

efficient = EfficientDiD(
    pre_periods="all",
    inference="multiplier_bootstrap",
    bootstrap_iterations=999,
    random_state=2026,
).fit(
    panel,
    outcome="outcome",
    entity="unit",
    time="period",
    treatment_time="first_treated",
    covariates=["baseline_outcome", "age"],
    cross_fitter=cross_fitter,
)

print(conventional.group_time)
print(conventional.event_study)
print(efficient.efficiency_weights)
print(efficient.simultaneous_event_study)
```

The data must be a balanced long panel with one row per entity-period, an absorbing first
treatment time, and an explicit never-treated sentinel (positive infinity by default).
`anticipation=` moves the effective treatment boundary back by an integer number of
periods. Robust inference treats the panel entity as the sampling unit; higher-level
one-way clustering is available through `covariance="clustered"` and `cluster=`.

The covariate-efficient path forms cohort-density ratios from cross-fitted multiclass
probabilities, estimates group-specific conditional outcome changes and residual-product
conditional covariances through `CrossFitter`, and solves the observation-specific
covariance systems without hidden regularization. Probabilities below
`nuisance_probability_floor` and singular systems refuse rather than clip or repair.
The efficiency claim is conditional on PT-All and the paper's nuisance regularity
conditions. Repeated cross-sections and sampling weights remain unsupported. See the
[DiD contract](docs/DID_CONTRACT.md) for formulas, assumptions, target populations, and
promotion gates.

There is no formula API yet. Prepare numeric arrays, `Series`, or `DataFrame` objects
explicitly, including categorical encoding and transformations. `add_constant=True` is
the default; set it to `False` when the supplied exogenous design already contains the
desired intercept or the model should not have one.

See [Package scope](docs/PACKAGE_SCOPE.md),
[Identification and interpretation](docs/IDENTIFICATION.md),
[Validation](docs/VALIDATION.md), [DiD contract](docs/DID_CONTRACT.md), and
[Architecture](docs/ARCHITECTURE.md). The package-wide Python/R/Stata evidence status is
tracked in the [cross-software parity register](docs/PARITY.md).

## Randomized-experiment example

```python
from causekit import RandomizedATE

result = RandomizedATE(adjustment="lin", covariance="robust").fit(
    outcome,
    treatment=assigned,  # exactly 0/1 with both arms present
    covariates=baseline_data,  # pre-treatment covariates only
)
print(result.summary_frame())
print(result.balance)
```

Lin adjustment centers covariates at the analysis-sample mean and interacts every
covariate with treatment. The `ate` coefficient is therefore the covariate-averaged
adjusted treatment contrast. It does not assume a common outcome slope across arms.
Covariates supplied with `adjustment="none"` are used only for balance diagnostics.

The causal interpretation requires genuine random assignment, consistency, no
interference, a pre-specified analysis population, no post-treatment adjustment, and an
inference choice matching the assignment/dependence structure. This release does not yet
handle blocked probabilities, cluster-level assignment estimands, randomization tests,
attrition correction, or multi-arm experiments.

## Installation

From PyPI after publication:

```bash
python -m pip install causekit==0.7.0a3
```

From a source checkout:

```bash
python -m pip install -e .
```

For development and validation dependencies:

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 through 3.13 is supported by the package metadata.

## Real-world workflow

The runnable workflow covers IV, randomized effects, cross-fitted IPW/AIPW, matching,
native partially linear DML, conventional DiD, and efficient DiD on four pinned real
datasets. The project does not redistribute the source files: downloads are opt-in,
HTTPS-only, cached outside the repository, and checked against release-pinned SHA-256
digests.

```bash
python -m pip install -e ".[validation]"
python examples/real_world_causal_workflow.py --download
```

The output keeps identification boundaries visible: numerical IV diagnostics cannot
validate exclusion, observational estimates require exchangeability and positivity,
cross-fitted matching receives no unsupported analytical standard error, and efficient
DiD is reported beside—not instead of—the conventional estimator.

See [real-data validation](docs/REAL_DATA_VALIDATION.md) for source provenance, pinned
digests, cross-language comparator mappings, and reproduction commands.

## Runnable example

The following example creates an endogenous treatment, fits 2SLS with one excluded
instrument, and requests HC1 inference. All pandas objects share the same index.

```python
import numpy as np
import pandas as pd

from causekit import IV2SLS

rng = np.random.default_rng(20260722)
nobs = 800
index = pd.RangeIndex(nobs, name="observation")

instrument = rng.normal(size=nobs)
baseline = rng.normal(size=nobs)
confounder = rng.normal(size=nobs)
treatment = 0.9 * instrument + 0.4 * baseline + 0.7 * confounder + rng.normal(size=nobs)
outcome = 2.0 * treatment + 0.5 * baseline + confounder + rng.normal(size=nobs)

result = IV2SLS(covariance="robust").fit(
    pd.Series(outcome, index=index, name="outcome"),
    endogenous=pd.DataFrame({"treatment": treatment}, index=index),
    instruments=pd.DataFrame({"encouragement": instrument}, index=index),
    exogenous=pd.DataFrame({"baseline": baseline}, index=index),
)

print(result.summary_frame())
diagnostic = result.first_stage["treatment"]
print(diagnostic.partial_r_squared)
print(diagnostic.classical_f_statistic)
print(diagnostic.excluded_instrument_statistic)
print(diagnostic.weak_instrument_warning)
print(result.to_markdown())
```

`instruments` contains only `encouragement`, the excluded instrument. With the default
intercept, the estimator constructs the full instrument set from the constant,
`baseline`, and `encouragement`. Parameter order is constant, supplied exogenous
regressors, then endogenous regressors.

The simulated construction makes the instrument independent of the latent confounder by
design. That property is known because the data-generating process is controlled; it
cannot generally be established from an observed dataset by fitting this model.

## Covariance choices

Choose covariance from the sampling structure, not from the p-value it produces.

| `covariance` | Inference target | Reference distribution | Additional input |
| --- | --- | --- | --- |
| `"unadjusted"` | Homoskedastic 2SLS covariance | Student t with residual degrees of freedom | None |
| `"robust"` | Heteroskedasticity-consistent HC1 covariance | Asymptotic normal | None |
| `"clustered"` | One-way cluster-robust CR1 covariance | Student t with `G - 1` degrees of freedom | `clusters=` with `G` groups |

`"robust"` is the default. The supported labels are exact; there are no undocumented
aliases. One-way clustering allows arbitrary dependence within a cluster and relies on
independence across clusters. It does not provide multiway clustering or a guarantee of
reliable few-cluster inference.

Covariance selection changes uncertainty estimates and the covariance-aware first-stage
test. It does not repair an invalid exclusion restriction or create instrument relevance.

## Diagnostics and their limits

`result.first_stage` is a dictionary keyed by endogenous-regressor name. Each diagnostic
contains the ordinary first-stage
R-squared, partial R-squared for the excluded instruments, a classical excluded-instrument
F statistic, and a joint excluded-instrument statistic built for the selected covariance.
The diagnostic records the restriction degrees of freedom, p-value, reference distribution,
and a weak-instrument warning. Its `classical_f_df_denom` field records the classical F
denominator degrees of freedom; a clustered covariance-aware F uses the `G - 1` reference
defined by the fitted cluster design.

The classical F statistic is a homoskedastic diagnostic even when the structural fit uses
HC1 or clustered inference. In those cases, use the covariance-aware joint statistic for
the corresponding sampling assumption. Neither statistic proves exclusion or
independence, and a warning threshold is a screening aid rather than an identification
certificate. This release does not claim a complete weak-IV-robust inference procedure.

For an overidentified homoskedastic fit, `result.overidentification` contains Sargan's
test. It is absent for exactly identified models and for `"robust"` or `"clustered"`
fits. Absence does not mean the restrictions passed. A non-rejection also does not prove
instrument validity; the test has limited power and evaluates the restrictions jointly.

## Data contract

The default `missing="raise"` policy rejects missing and non-finite numeric values. The
explicit `missing="drop"` alternative builds one complete-case mask across the outcome,
regressors, instruments, and clusters; it records the number removed in
`result.dropped_rows` and preserves the retained index. Both policies reject misaligned
pandas indices rather than silently reordering or intersecting rows. The estimator validates
row counts, unique column names, dimensionality, numeric content, full-rank designs, the
order condition, and the rank condition. Cluster labels follow the same row-alignment and
missing-value rules. Prediction always rejects missing or non-finite values.

This release does not silently:

- drop or impute observations;
- align pandas inputs by an index intersection;
- create dummy variables or interactions;
- standardize variables;
- infer which columns are exogenous instruments; or
- select instruments from the outcome data.

Preserve the preprocessing recipe and pass prediction inputs with the fitted schema and
column order. Strict refusal is deliberate: silent sample or schema drift can change the
estimand.

## Causal interpretation

Linear IV can support a causal interpretation only under a defensible design. At minimum,
discuss instrument relevance, independence from structural disturbances, the exclusion
restriction, the treatment version and interference assumptions, and the population to
which the result applies. With heterogeneous effects, binary instruments and treatments
require additional conditions such as monotonicity for a local average treatment effect
interpretation. The resulting local estimand need not equal the population average
treatment effect.

`causekit` reports numerical evidence relevant to some implications of the design. It
cannot learn exclusion or independence from the observed covariance matrix, and it does
not turn observational association into causation. See
[Identification and interpretation](docs/IDENTIFICATION.md) for the formal contract and a
reporting checklist.

## Migration from `limiteddepkit.TreatmentEffect`

The historical `limiteddepkit.TreatmentEffect` was an ordinary homoskedastic linear 2SLS
estimator, not a limited-dependent-variable model. `limiteddepkit` therefore removed it
from its public namespaces. The migration is complete and its obsolete source snapshot
has also been removed from that repository.

`causekit.IV2SLS` is the supported destination. Its design was informed by the migration
requirements and the public econometric definition of 2SLS; no private implementation
code was copied into this package. `causekit` has its own validation, covariance,
diagnostic, and result contracts, including a numerical test that reconstructs the old
homoskedastic matrix result after explicitly mapping the full instrument matrix.

The migration is not a drop-in rename:

- legacy `Z` was the full instrument matrix and had to span `X_exog`;
- new `instruments=` contains excluded instruments only, while `exogenous=` is included
  internally in the full instrument matrix;
- the new estimator adds a constant by default;
- `covariance="unadjusted"` is the closest inference setting to the legacy
  homoskedastic estimator;
- the new result exposes one labelled `params` vector in constant, exogenous, endogenous
  order rather than separate `params_exog` and `params_endog` vectors; and
- strict pandas alignment, richer diagnostics, multiple covariance choices, and explicit
  metadata are new contracts.

For a faithful migration, separate the excluded columns from the legacy full `Z`, decide
explicitly whether the old design already supplied a constant, and compare estimates on
the same rows and column order. Do not assume numerical equivalence until that comparison
has been run for the actual specification.

## Ecosystem conventions

The result surface follows conventions used across `systemgmmkit` and `limiteddepkit`:
labelled pandas parameters and covariance matrices, aligned standard errors and test
statistics, explicit `nobs` and covariance metadata, confidence intervals, prediction,
linear combinations, Wald tests, tabular summaries, and optional OutputHub adaptation.
Compatibility here means familiar
field meanings and reporting shape; it does not mean class identity or universal
drop-in interchangeability across estimators.

The packages remain separated by estimand:

- `causekit` owns identification-aware causal and cross-sectional IV workflows;
- `limiteddepkit` owns limited-outcome and observation-rule models; and
- `systemgmmkit` owns panel-data and dynamic-panel GMM workflows.

Applicable validation, indexing, covariance, diagnostics, and reporting conventions are
reused conceptually without importing private source or coupling the packages at runtime.
Limited-outcome likelihoods are not duplicated here. The supplied-nuisance IPW/AIPW and
matching APIs accept provider-neutral predictions while keeping causal identification
inside `causekit`; the narrow estimated-score matching path consumes only the public
`FittedPropensityMLEProtocol`. `CrossFitter` owns reusable fold orchestration and
prediction adaptation, not nuisance estimators.

## Roadmap

The matching alpha follows the
[nearest-neighbor matching contract](docs/MATCHING_CONTRACT.md). Known-score analytical
inference, R reference parity, OutputHub adaptation, and 100,000-row fixed- and estimated-
score inference smokes are implemented, with fixed-score parity against pinned R `Matching`
4.10-15 and Stata/MP 17. Full-sample Logit-MLE first-step adjustment is independently
checked against `statsmodels` and a reviewed Stata/IC 17 `teffects psmatch` fixture. DiD
promotion includes cross-fitted covariate nuisances and simultaneous
event-study bands; pre-trend/Hausman diagnostics and repeated cross-sections remain.
The causal-ML alpha includes native partially linear DML and separately contracted public
[honest R-learner](docs/R_LEARNER_CONTRACT.md) and
[honest DR-learner](docs/DR_LEARNER_CONTRACT.md) paths, with immutable
construction/evaluation roles, held-out loss/calibration, group inference, and graph data.
Available aligned Python/R/Stata parity rows are recorded, while unavailable comparator
cells remain explicit. Later releases may add regression discontinuity and panel IV. Each family
must define its estimand, assumptions, failure behavior, diagnostics, and independent
validation evidence before promotion.

DADPLM and BDCPM are outside the current package scope. The roadmap is directional, not a
promise of API shape or release timing. See [Package scope](docs/PACKAGE_SCOPE.md).

## Verification commands

From the repository root, install the development dependencies and run:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
python -m pip_audit --strict .
python -m build
python -m twine check --strict dist/*
```

Optional independent-reference checks use the `validation` extra and the registered
pytest markers described in [Validation](docs/VALIDATION.md). These commands are the
required verification workflow; this document does not assert that a particular checkout
has passed them.

## Contributing, security, citation, and license

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing an estimator or public contract.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Cite the exact package version; machine-readable metadata is in
  [CITATION.cff](CITATION.cff).
- Changes are recorded in [CHANGELOG.md](CHANGELOG.md).
- The active milestone sequence and reuse rules are recorded in [HANDOVER.md](HANDOVER.md).

`causekit` is distributed under the [MIT License](LICENSE).
