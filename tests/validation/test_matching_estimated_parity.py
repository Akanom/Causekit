"""Independent Python first-step and Stata-harness matching checks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from causekit import FittedPropensityMLEProtocol, NearestNeighborMatch


class _StatsmodelsLogitResult:
    """Expose statsmodels' fitted Logit through CauseKit's public protocol."""

    def __init__(self, result, design: pd.DataFrame) -> None:
        self._result = result
        self.params = pd.Series(result.params, index=design.columns, name="estimate")
        self.converged = bool(result.mle_retvals["converged"])
        self.nobs = len(design)
        self.feature_names = tuple(design.columns)

    def predict_proba(self, X):
        probability = np.asarray(self._result.predict(X), dtype=float)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


def _fixture():
    index = pd.Index([f"unit_{position}" for position in range(12)])
    design = pd.DataFrame(
        {
            "const": 1.0,
            "x": [-2.60, -2.00, -1.35, -0.95, -0.62, -0.18, 0.13, 0.49, 0.91, 1.38, 1.92, 2.57],
        },
        index=index,
    )
    treatment = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], index=index, dtype=int)
    outcome = pd.Series(
        [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 11.0, 10.0, 14.0],
        index=index,
    )
    return outcome, treatment, design


@pytest.mark.validation
@pytest.mark.parametrize(
    ("estimand", "expected_estimate", "expected_standard_error"),
    [
        ("att", 2.5, 0.49507487173864956),
        ("atc", 13.0 / 6.0, 0.4566897818895616),
        ("ate", 7.0 / 3.0, 0.6975256843595473),
    ],
)
def test_estimated_score_matching_uses_statsmodels_logit_mle_without_reestimating(
    estimand: str,
    expected_estimate: float,
    expected_standard_error: float,
) -> None:
    outcome, treatment, design = _fixture()
    fitted = sm.Logit(treatment, design).fit(method="newton", maxiter=200, tol=1e-12, disp=False)
    model = _StatsmodelsLogitResult(fitted, design)
    assert isinstance(model, FittedPropensityMLEProtocol)

    result = NearestNeighborMatch(
        estimand=estimand,
        metric="propensity",
        caliper=None,
        common_support=None,
        inference="abadie_imbens_estimated",
        variance_neighbors=1,
    ).fit(
        outcome,
        treatment=treatment,
        propensity_model=model,
        propensity_design=design,
        propensity_score_status="estimated",
        propensity_provenance="statsmodels.Logit full-sample MLE",
    )

    np.testing.assert_allclose(
        result.propensity_scores,
        fitted.predict(design),
        rtol=0,
        atol=2e-14,
    )
    assert result.estimate == pytest.approx(expected_estimate, abs=2e-14)
    assert result.standard_error == pytest.approx(expected_standard_error, abs=2e-12)


def test_estimated_stata_harness_maps_the_separate_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_matching_estimated_stata.do").read_text(
        encoding="utf-8"
    )

    assert script.count("teffects psmatch") == 6
    assert script.count("nneighbor(1) vce(robust, nn(2))") == 6
    assert script.count("vce(robust, nn(2)) nocorrection") == 3
    assert 'display as result "first_step_covariance_neighbors=2"' in script
    assert 'display as result "parity_status=`parity_status\'"' in script
    assert 'display as result "variance_neighbors=1"' in script
    assert 'display as result "stata_robust_neighbors=2"' in script
    assert 'display as result "first_step_regression_neighbors=2"' in script
    assert "teffects nnmatch" not in script
    assert "att_ps_uncorrected_variance" in script
    assert "atc_ps_uncorrected_variance" in script
    assert "ate_ps_uncorrected_variance" in script


def test_estimated_stata_harness_persists_diagnostics_before_asserting() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_matching_estimated_stata.do").read_text(
        encoding="utf-8"
    )

    artifact = "benchmarks/validate_matching_estimated_stata_output.txt"
    open_position = script.index("file open `results_file' using \"`results_path'\"")
    close_position = script.index("file close `results_file'")
    first_assertion_position = script.index('if "`parity_status\'" != "pass"')

    assert f'local results_path "{artifact}"' in script
    assert open_position < close_position < first_assertion_position
    assert "file write `results_file' \"stata_version=`c(stata_version)'\"" in script
    assert 'file write `results_file\' "att_standard_error="' in script
    assert 'file write `results_file\' "expected_att_standard_error="' in script
    assert 'file write `results_file\' "att_standard_error_absolute_difference="' in script
    assert 'file write `results_file\' "att_ps_uncorrected_variance="' in script
    assert 'file write `results_file\' "expected_att_known_score_variance="' in script
    assert 'file write `results_file\' "att_ps_first_step_adjustment="' in script
    assert "file write `results_file' \"parity_status=`parity_status'\"" in script


def test_estimated_stata_harness_respects_identifier_length_limit() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (repository_root / "benchmarks" / "validate_matching_estimated_stata.do").read_text(
        encoding="utf-8"
    )

    declarations = re.findall(
        r"(?m)^(?:generate(?:\s+\w+)?|local|matrix|scalar|tempname)\s+([A-Za-z_]\w*)",
        script,
    )
    overlength = [name for name in declarations if len(name) > 32]
    assert overlength == []


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_saved_stata_17_result_matches_estimated_score_contract(estimand: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script_path = repository_root / "benchmarks" / "validate_matching_estimated_stata.do"
    output_path = repository_root / "benchmarks" / "validate_matching_estimated_stata_output.txt"
    reference = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )

    # This reviewed artifact predates the distribution rename and is immutable evidence.
    assert reference["artifact"] == "causalkit_estimated_score_matching_stata_parity"
    assert reference["execution_date"] == "2026-07-29"
    assert reference["host_os"] == "Windows"
    assert reference["stata_version"] == "17"
    assert reference["stata_flavor"] == "IC"
    assert reference["causalkit_version"] == "0.6.0a4"
    assert reference["causalkit_estimator_commit"] == "3a7a33f"
    assert reference["generator"] == "benchmarks/validate_matching_estimated_stata.do"
    assert (
        reference["generator_sha256"]
        == hashlib.sha256(script_path.read_bytes()).hexdigest().upper()
    )
    assert reference["contract"] == "full_sample_unpenalized_logit_mle_no_ties"
    assert reference["parity_status"] == "pass"
    assert all(value == "pass" for name, value in reference.items() if name.endswith("_status"))
    assert reference["variance_neighbors"] == "1"
    assert reference["stata_robust_neighbors"] == "2"
    assert reference["first_step_covariance_neighbors"] == "2"
    assert reference["first_step_regression_neighbors"] == "2"
    assert reference["first_step_covariate_neighbors"] == "1"

    outcome, treatment, design = _fixture()
    fitted = sm.Logit(treatment, design).fit(method="newton", maxiter=200, tol=1e-12, disp=False)
    native = NearestNeighborMatch(
        estimand=estimand,
        metric="propensity",
        caliper=None,
        common_support=None,
        inference="abadie_imbens_estimated",
        variance_neighbors=int(reference["variance_neighbors"]),
        first_step_covariance_neighbors=int(reference["first_step_covariance_neighbors"]),
        first_step_regression_neighbors=int(reference["first_step_regression_neighbors"]),
        first_step_covariate_neighbors=int(reference["first_step_covariate_neighbors"]),
    ).fit(
        outcome,
        treatment=treatment,
        propensity_model=_StatsmodelsLogitResult(fitted, design),
        propensity_design=design,
        propensity_score_status="estimated",
        propensity_provenance="statsmodels.Logit full-sample MLE",
    )

    tolerance = float(reference["tolerance"])
    assert native.estimate == pytest.approx(float(reference[f"{estimand}_estimate"]), abs=tolerance)
    assert native.known_score_variance == pytest.approx(
        float(reference[f"{estimand}_ps_uncorrected_variance"]), abs=tolerance
    )
    assert native.first_step_variance_adjustment == pytest.approx(
        float(reference[f"{estimand}_ps_first_step_adjustment"]), abs=tolerance
    )
    assert native.standard_error == pytest.approx(
        float(reference[f"{estimand}_standard_error"]), abs=tolerance
    )


@pytest.mark.simulation
def test_estimated_score_ate_has_seeded_recovery_and_coverage_smoke() -> None:
    rng = np.random.default_rng(20_260_728)
    true_effect = 1.25
    estimates: list[float] = []
    standard_errors: list[float] = []
    covered: list[bool] = []
    for _ in range(40):
        covariate = rng.normal(size=500)
        design = pd.DataFrame({"const": 1.0, "x": covariate})
        probability = 1.0 / (1.0 + np.exp(-(-0.1 + 0.65 * covariate)))
        treatment = pd.Series(rng.binomial(1, probability), dtype=int)
        outcome = pd.Series(0.4 + 0.7 * covariate + true_effect * treatment + rng.normal(size=500))
        fitted = sm.Logit(treatment, design).fit(
            method="newton", maxiter=200, tol=1e-11, disp=False
        )
        result = NearestNeighborMatch(
            estimand="ate",
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=1,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=_StatsmodelsLogitResult(fitted, design),
            propensity_design=design,
            propensity_score_status="estimated",
            propensity_provenance="seeded_statsmodels_logit_mle",
        )
        interval = result.conf_int()
        estimates.append(result.estimate)
        standard_errors.append(result.standard_error)
        covered.append(interval["lower"] <= true_effect <= interval["upper"])

    empirical_standard_deviation = float(np.std(estimates, ddof=1))
    mean_standard_error = float(np.mean(standard_errors))
    assert np.mean(estimates) == pytest.approx(true_effect, abs=0.10)
    assert 0.82 <= float(np.mean(covered)) <= 1.0
    assert mean_standard_error / empirical_standard_deviation == pytest.approx(1.0, abs=0.35)
