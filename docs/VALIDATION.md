# Validation strategy

`causalkit` separates implementation checks from evidence claims. A command returning
success in one environment supports the tested version, fixture, specification, comparator,
and tolerance. It does not establish universal numerical parity or validate an empirical
instrument.

This document defines the validation gate and how to report it. It does **not**
assert that the commands below have passed for the current checkout. Release notes and
review records must state the actual execution environment and outcome.

For supplied-nuisance IPW/AIPW, maintained tests must include the exact score identity,
influence-function centering and variance identity, deterministic recovery under a known
data-generating process, propensity-bound refusal, clipping disclosure, exact index
alignment, cluster-sum covariance, and a large vectorized smoke path. Nuisance predictions
used in empirical validation must record whether they are oracle, fixed low-complexity,
held-out, or cross-fitted; in-sample adaptive predictions cannot be presented as validated
cross-fitted inference.

## Claim boundary

Validation can provide evidence that:

- the estimator implements the declared 2SLS matrix problem;
- coefficient and covariance calculations agree with analytical identities or an aligned
  independent implementation;
- diagnostics use their stated definitions and reference distributions;
- strict input, alignment, schema, and refusal contracts behave as documented; and
- results remain internally aligned through prediction and reporting.

Validation cannot establish that an empirical instrument is independent, satisfies the
exclusion restriction, has no interference pathway, or identifies the analyst's preferred
causal estimand. Those claims require research-design evidence outside the software.

## Evidence layers

### 1. Contract and refusal tests

Fast tests should cover the complete public surface:

- accepted constructor and fit arguments;
- exact covariance labels and required cluster input;
- parameter order: constant, exogenous, endogenous;
- `params`, `covariance`, `standard_errors`, `test_statistics`, `pvalues`, `residuals`,
  `fitted_values`, `nobs`, `df_resid`, and covariance metadata;
- confidence intervals, summary frames, Markdown output, and prediction schema;
- missing and non-finite values in every numeric role;
- mismatched and permuted pandas indices, plus consistent repeated-index sequences;
- duplicate or changed column names and wrong prediction column order;
- inconsistent lengths and dimensionality;
- exact collinearity in structural and instrument designs;
- underidentification, exact rank failure, and too few residual degrees of freedom; and
- missing, misaligned, degenerate, or insufficient cluster labels; and
- joint complete-case filtering, retained-index preservation, and `dropped_rows` metadata
  under `missing="drop"`.

Refusal behavior is part of numerical correctness. A test that silently drops a bad row is
not equivalent to a test that confirms `missing="raise"`; the explicit `missing="drop"`
path must prove that every input used the same mask.

### 2. Analytical identities

Small deterministic fixtures should compare public results with independently assembled
matrix identities:

```text
theta_hat = (S' P_Q S)^(-1) S' P_Q y
fitted     = S theta_hat
residual   = y - fitted
```

The gate should separately reconstruct:

- homoskedastic covariance from the structural residual variance;
- HC1 sandwich covariance with multiplier `n / (n - k)`;
- one-way CR1 covariance from cluster-summed scores with multiplier
  `[G / (G - 1)] * [(n - 1) / (n - k)]`;
- coefficient statistics and confidence intervals using the recorded t or normal
  reference distribution;
- partial R-squared from residualized endogenous variables and instruments;
- classical first-stage excluded-instrument F tests;
- covariance-aware first-stage joint tests; and
- Sargan's `n R-squared` identity and `q - m` degrees of freedom when its homoskedastic
  conditions apply.

Tests should also verify covariance symmetry, finite diagonals, label alignment, the
identity `fitted_values + residuals == y` within numerical tolerance, and invariance to
consistent row permutation and cluster-label renaming.

### 3. Deterministic simulation

Simulations should exercise known data-generating processes rather than assert exact
sample recovery. Required designs include:

- a strong, valid, just-identified instrument;
- an overidentified model with multiple excluded instruments;
- multiple endogenous regressors with distinct relevant variation;
- heteroskedastic structural errors for HC1 inference;
- within-cluster dependence for CR1 inference;
- weak or nearly irrelevant excluded instruments; and
- deliberately invalid or collinear designs that must be rejected or warned about.

Use fixed random seeds. Recovery thresholds must be chosen before observing a convenient
draw and justified from the sample size, signal strength, and conditioning. A simulation
can show behavior under its known construction; it does not validate instruments in
uncontrolled data.

### 4. Independent implementation comparisons

The optional `validation` dependencies support aligned comparisons with maintained Python
implementations. `linearmodels` is the primary external 2SLS comparator because its IV
interface also distinguishes exogenous regressors from excluded instruments. `statsmodels`
can provide supporting regression and diagnostic identities where definitions align.

Every comparison must record:

- comparator package and version;
- data generator or data provenance and random seed;
- outcome, intercept, exogenous, endogenous, and excluded-instrument mapping;
- covariance definition and finite-sample/debiasing option;
- cluster definition and cluster count, when applicable;
- parameter and matrix label mapping;
- fields compared and numeric tolerances; and
- observed maximum absolute and relative differences.

Comparator defaults often differ. In particular, intercept handling, HC versus debiased
scaling, cluster corrections, degrees of freedom, and p-value reference distributions must
be aligned explicitly. Agreement after an undocumented option change is not reproducible
evidence.

## Required model matrix

The following matrix defines the intended release evidence. A row may be described as
passing only after its tests have executed successfully in the recorded environment.

| Area | Required cases | Required assertions |
| --- | --- | --- |
| Point estimates | Just identified, overidentified, multiple endogenous regressors | Estimates, labels, ordering, fitted values, residuals |
| Unadjusted inference | Homoskedastic fixture | Covariance, t statistics, p-values, intervals, `n-k` degrees of freedom |
| HC1 inference | Heteroskedastic fixture | Sandwich meat, `n/(n-k)` multiplier, normal reference |
| One-way CR1 | Unequal cluster sizes and within-cluster dependence | Cluster scores, CR1 multiplier, `G-1` reference degrees, cluster metadata |
| First stages | Strong, weak, multiple-instrument, multiple-endogenous cases | R-squared, partial R-squared, classical F, covariance-aware joint test and metadata |
| Overidentification | Just identified, overidentified, every covariance label | Sargan only for overidentified unadjusted fit; correct statistic, p-value, and `q-m` df |
| Data contract | NumPy and pandas, missing/non-finite, misaligned indices, duplicate names | Stable success behavior or precise refusal; never silent sample drift |
| Post-estimation | In-sample and new-data prediction, tables, Markdown | Schema enforcement, index preservation, numerical alignment |
| Integrations | Optional dependency present and absent | Stable adapter output or actionable optional-dependency error |

## Numerical tolerances

Do not use one global tolerance. Exact integer metadata and labels require equality.
Closed-form values on well-conditioned synthetic designs should use tight floating-point
tolerances. Reference-package comparisons may need a documented wider tolerance where
linear solvers, finite-sample conventions, or tail-probability implementations differ.

For every tolerance, record:

- whether it is absolute, relative, or both;
- the scale and conditioning of the fixture;
- the expected source of numerical variation; and
- the maximum observed discrepancy in the recorded run.

Never loosen a tolerance merely to absorb an unexplained regression. Diagnose matrix
conditioning, sample alignment, parameter order, and comparator conventions first.

## Reproduction commands

Install the development environment from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Run the complete project gate:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
python -m build
python -m twine check --strict dist/*
```

Run marked evidence subsets explicitly when reviewing them:

```bash
python -m pytest -m validation
python -m pytest -m simulation
```

Run the README example in a clean installation and inspect both wheel and source
distribution before release. Archive the commands, operating system, Python version,
resolved dependency versions, commit identifier, test output, and artifact hashes with a
release decision.

## Reporting validation status

Use bounded statements:

- “matched `linearmodels` for the recorded specification and covariance options within the
  declared tolerances”;
- “recovered the generating coefficient across the recorded deterministic simulations”;
- “the strict index-mismatch refusal test passed”; or
- “not run in this environment; exact command: ...”.

Avoid statements such as “fully validated,” “proven correct,” “identical to all software,”
or “the instruments are valid.” If a required gate is unavailable, distinguish an
environment limitation from a code failure and list the remaining definition-of-done gap.

## Adding evidence

New evidence belongs in maintained tests or a documented validation harness, not only in a
notebook or screenshot. Generated artifacts must record their script, source data or
generator, configuration, comparator versions, mappings, tolerances, date, and reproduction
command. Do not commit restricted data, local paths, or unreviewed binary output.
