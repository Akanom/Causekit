# causalkit

`causalkit` is an identification-aware Python package for causal inference and
instrumental-variable workflows. The `0.1.0a1` surface is intentionally narrow: it
provides a stable linear two-stage least-squares estimator with explicit instrument,
covariance, diagnostic, and data-alignment contracts.

This is alpha research software. A successful fit is not evidence that an instrument is
valid, and an IV coefficient is not automatically an average treatment effect. State the
estimand and defend the identifying assumptions before using causal language.

## Current scope

The stable `0.1.0a1` API provides:

- `IV2SLS` with one or more endogenous regressors and excluded instruments;
- an excluded-instrument API: exogenous regressors are added to the instrument matrix
  internally and must not be repeated in `instruments`;
- homoskedastic (`"unadjusted"`), HC1 (`"robust"`), and one-way cluster-robust CR1
  (`"clustered"`) covariance estimators;
- strict missing-value, non-finite-value, index-alignment, shape, name, and rank checks;
- classical and covariance-aware first-stage excluded-instrument diagnostics; and
- Sargan's overidentification test only for overidentified fits using
  `covariance="unadjusted"`.

There is no formula API yet. Prepare numeric arrays, `Series`, or `DataFrame` objects
explicitly, including categorical encoding and transformations. `add_constant=True` is
the default; set it to `False` when the supplied exogenous design already contains the
desired intercept or the model should not have one.

See [Package scope](docs/PACKAGE_SCOPE.md),
[Identification and interpretation](docs/IDENTIFICATION.md),
[Validation](docs/VALIDATION.md), and [Architecture](docs/ARCHITECTURE.md).

## Installation

From a source checkout:

```bash
python -m pip install -e .
```

For development and validation dependencies:

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 through 3.13 is supported by the package metadata.

## Runnable example

The following example creates an endogenous treatment, fits 2SLS with one excluded
instrument, and requests HC1 inference. All pandas objects share the same index.

```python
import numpy as np
import pandas as pd

from causalkit import IV2SLS

rng = np.random.default_rng(20260722)
nobs = 800
index = pd.RangeIndex(nobs, name="observation")

instrument = rng.normal(size=nobs)
baseline = rng.normal(size=nobs)
confounder = rng.normal(size=nobs)
treatment = (
    0.9 * instrument
    + 0.4 * baseline
    + 0.7 * confounder
    + rng.normal(size=nobs)
)
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

`causalkit` reports numerical evidence relevant to some implications of the design. It
cannot learn exclusion or independence from the observed covariance matrix, and it does
not turn observational association into causation. See
[Identification and interpretation](docs/IDENTIFICATION.md) for the formal contract and a
reporting checklist.

## Migration from `limiteddepkit.TreatmentEffect`

The historical `limiteddepkit.TreatmentEffect` was an ordinary homoskedastic linear 2SLS
estimator, not a limited-dependent-variable model. `limiteddepkit` therefore removed it
from its public namespaces and retained a non-installable snapshot under its
`_out_of_scope/` migration area.

`causalkit.IV2SLS` is the supported destination. Its design was informed by the migration
requirements and the public econometric definition of 2SLS; no private implementation
code was copied into this package. `causalkit` has its own validation, covariance,
diagnostic, and result contracts.

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

- `causalkit` owns identification-aware causal and cross-sectional IV workflows;
- `limiteddepkit` owns limited-outcome and observation-rule models; and
- `systemgmmkit` owns panel-data and dynamic-panel GMM workflows.

Applicable validation, indexing, covariance, diagnostics, and reporting conventions are
reused conceptually without importing private source or coupling the packages at runtime.

## Roadmap

Later releases may add randomized-experiment adjustment, IPW and AIPW, matching,
difference-in-differences and event studies, regression discontinuity, and panel IV. Each
family must define its estimand, assumptions, failure behavior, diagnostics, and independent
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

`causalkit` is distributed under the [MIT License](LICENSE).
