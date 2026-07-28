# Contributing to causalkit

Thank you for helping improve `causalkit`. This package treats estimator correctness,
identification language, explicit failure behavior, and reproducible validation as part of
the public API.

## Development setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the editable package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Keep changes focused. Preserve unrelated work, update documentation with behavior, and do
not add generated output, local paths, credentials, private data, caches, or build
artifacts to a change set.

## Required quality gate

Run the following commands before requesting review:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
python -m pip_audit --strict .
python -m build
python -m twine check --strict dist/*
```

Run the README example from a clean environment as an end-to-end smoke check. If a gate
cannot run, report the exact command, environment limitation, and unverified behavior.
Never describe an unexecuted command as passing.

## Econometric contribution standard

An estimator or diagnostic is not ready because it returns plausible numbers. A proposal
must define:

1. the research question and estimand;
2. the statistical model and parameterization;
3. identification assumptions, including the population and treatment version;
4. supported data structures and explicit missing/index/schema behavior;
5. covariance targets, small-sample corrections, and reference distributions;
6. diagnostics and what they can and cannot establish;
7. numerical failure modes and actionable errors;
8. independent reference or analytical validation; and
9. user-facing documentation with appropriately qualified causal language.

For IV changes, test the order and rank conditions separately. Include just-identified and
overidentified designs, weak and irrelevant instruments, collinearity, multiple endogenous
regressors, heteroskedastic errors, one-way clustered errors, few/invalid clusters, and
permuted or mismatched pandas indices. Verify estimates, covariance matrices, inference
distributions, degrees of freedom, diagnostics, fitted values, residuals, prediction
schema, and parameter order.

Do not add an overidentification statistic under a covariance regime for which its null
distribution is not implemented. In particular, Sargan's homoskedastic statistic must not
be presented as robust merely because the coefficient covariance is HC1 or clustered.

## Validation evidence

Use three complementary forms of evidence where applicable:

- analytical identities and small hand-checkable examples;
- deterministic simulations with known data-generating processes; and
- comparison with an independent maintained implementation under a fully aligned
  specification.

Mark independent reference tests with `@pytest.mark.validation` and deterministic
simulation or recovery tests with `@pytest.mark.simulation`. Fix random seeds and justify
tolerances from scale, conditioning, sample size, and comparator conventions. Record
package versions and parameter mappings. Agreement on one fixture is evidence for that
fixture and specification, not universal parity.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the claim boundary and reproduction
commands.

## API and architecture rules

Before changing a public class, argument, field, or label:

- identify existing consumers and migration consequences;
- preserve the excluded-instrument contract;
- retain strict pandas alignment and schema behavior;
- use the established validation, covariance, diagnostics, result, and integration
  layers;
- update exports, tests, README examples, methodology docs, and the changelog; and
- document any unavoidable breaking change before release.

The accepted covariance labels are `"unadjusted"`, `"robust"`, and `"clustered"`.
Avoid aliases that make the inferential target ambiguous. Formula parsing is not part of
the current API; a future formula layer must compile into and preserve the same explicit
design-matrix contract.

Result objects should remain familiar to users of `systemgmmkit` and `limiteddepkit`:
labelled pandas values, stable parameter order, explicit covariance metadata, clear
diagnostic objects, and report adapters. Compatibility does not justify copying private
source, importing sibling internals, or creating a hard runtime dependency.

## Tests and style

- Put fast behavior and regression tests under `tests/`.
- Test refusal paths as carefully as successful fits.
- Use deterministic seeds and avoid fragile assertions on random p-values.
- Test public behavior rather than internal implementation details where possible.
- Preserve Python 3.10 compatibility.
- Follow the configured Ruff rules and 100-character line length.
- Add a regression test for each bug fix unless technically impossible.

Security-relevant failures, including data leakage, unsafe serialization, dependency
compromise, or path handling, should be reported privately under [SECURITY.md](SECURITY.md).
Ordinary numerical or methodological defects belong in the public issue tracker unless
they expose confidential information.

## Pull request checklist

```text
[ ] The estimand, assumptions, and supported population are documented.
[ ] The implementation is connected to the public execution path.
[ ] Success, edge, and refusal paths are tested.
[ ] Numerical evidence is reproducible and appropriately scoped.
[ ] Covariance and diagnostic labels match their implemented definitions.
[ ] README, methodology docs, and CHANGELOG agree with the code.
[ ] Backward-compatibility and migration effects are documented.
[ ] Security, privacy, and performance implications were reviewed.
[ ] Ruff, formatting, mypy, pytest, build, and artifact checks were run.
[ ] The final diff contains no unrelated work, secrets, private data, or generated junk.
```

## Scope proposals

The roadmap includes built-in cross-fitting, matching, DiD/event studies, RDD,
and panel IV. Roadmap placement is not automatic approval. Start with the estimand and
validation design, then show how the feature fits [docs/PACKAGE_SCOPE.md](docs/PACKAGE_SCOPE.md)
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). DADPLM and BDCPM are not current
`causalkit` scope.
