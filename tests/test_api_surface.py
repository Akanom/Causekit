"""Regression tests for the initial stable causekit namespace."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import causekit
from causekit import IV2SLS, IV2SLSResult

PUBLIC_EXPORTS = {
    "CATEEstimatorProtocol",
    "CATEResultProtocol",
    "IV2SLS",
    "IV2SLSResult",
    "RandomizedATE",
    "RandomizedATEResult",
    "IPWATE",
    "AIPWATE",
    "ObservationalATEResult",
    "OverlapDiagnostic",
    "CrossFitter",
    "CrossFitResult",
    "CrossFitTask",
    "CrossFitTaskResult",
    "ClassProbabilityCrossFitResult",
    "ClassProbabilityCrossFitTask",
    "ClassProbabilityTaskCrossFitResult",
    "NuisanceEstimatorProtocol",
    "NuisanceDiagnosticsProtocol",
    "PropensityResultProtocol",
    "WeightedCATEEstimatorProtocol",
    "OutcomeResultProtocol",
    "NearestNeighborMatch",
    "NearestNeighborMatchResult",
    "PropensityScoreStatus",
    "FittedPropensityMLEProtocol",
    "DifferenceInDifferences",
    "DRLearner",
    "DRLearnerResult",
    "EfficientDiD",
    "DiDHausmanDiagnostic",
    "DiDPretrendDiagnostic",
    "DiDResult",
    "did_hausman_test",
    "PartiallyLinearDML",
    "PartiallyLinearDMLResult",
    "NativeSplineRidgeCATE",
    "NativeSplineRidgeCATEResult",
    "RLearner",
    "RLearnerResult",
    "RepeatedCrossSectionDiD",
    "RepeatedCrossSectionCompositionDiagnostic",
    "RepeatedCrossSectionDiDResult",
    "RepeatedCrossSectionPretrendDiagnostic",
    "did_rcs_composition_test",
    "confint",
    "fitted_values",
    "predict",
    "residuals",
    "summary_frame",
    "vcov",
}


def test_initial_stable_namespace_exports_iv_and_postestimation_contract() -> None:
    assert set(causekit.__all__) >= PUBLIC_EXPORTS
    assert all(hasattr(causekit, name) for name in PUBLIC_EXPORTS)
    assert "TreatmentEffect" not in causekit.__all__
    assert not hasattr(causekit, "TreatmentEffect")
    assert not hasattr(causekit, "NativeOrthogonalStackedCATE")


def test_causekit_has_no_limiteddepkit_dependency_or_import() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    metadata = (repository_root / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "limiteddepkit" not in metadata

    import_roots = [
        repository_root / "src",
        repository_root / "tests",
        repository_root / "benchmarks",
    ]
    imported = []
    for root in import_roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if any(
                line.lstrip().startswith(("import limiteddepkit", "from limiteddepkit"))
                for line in source.splitlines()
            ):
                imported.append(path.relative_to(repository_root).as_posix())
    assert imported == []


def test_distribution_and_import_namespace_are_causekit_only() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    metadata = (repository_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "causekit"' in metadata
    assert 'version = "0.7.0a6"' in metadata
    assert causekit.__version__ == "0.7.0a6"
    assert (repository_root / "src" / "causekit" / "__init__.py").is_file()
    assert not (repository_root / "src" / "causalkit").exists()


def test_estimator_signatures_keep_identification_inputs_explicit() -> None:
    constructor = inspect.signature(IV2SLS)
    assert constructor.parameters["covariance"].default == "robust"
    assert constructor.parameters["add_constant"].default is True
    assert constructor.parameters["missing"].default == "raise"

    fit = inspect.signature(IV2SLS.fit)
    assert list(fit.parameters) == [
        "self",
        "y",
        "endogenous",
        "instruments",
        "exogenous",
        "clusters",
    ]
    assert fit.parameters["endogenous"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit.parameters["instruments"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit.parameters["exogenous"].default is None
    assert fit.parameters["clusters"].default is None

    matching = inspect.signature(causekit.NearestNeighborMatch)
    assert matching.parameters["estimand"].default == "att"
    assert matching.parameters["replacement"].default is True
    assert matching.parameters["caliper"].default == "auto"
    assert matching.parameters["ties"].default == "all"
    assert matching.parameters["inference"].default == "none"
    assert matching.parameters["variance_neighbors"].default == 1
    assert matching.parameters["first_step_covariance_neighbors"].default == 2
    assert matching.parameters["first_step_regression_neighbors"].default == 2
    assert matching.parameters["first_step_covariate_neighbors"].default == 1

    matching_fit = inspect.signature(causekit.NearestNeighborMatch.fit)
    assert matching_fit.parameters["propensity_score_status"].default == "estimated"
    assert matching_fit.parameters["propensity"].default is None
    assert matching_fit.parameters["propensity_model"].default is None
    assert matching_fit.parameters["propensity_design"].default is None

    conventional_did = inspect.signature(causekit.DifferenceInDifferences)
    assert conventional_did.parameters["control_group"].default == "never_treated"
    assert conventional_did.parameters["anticipation"].default == 0
    assert conventional_did.parameters["covariance"].default == "robust"
    assert conventional_did.parameters["inference"].default == "analytic"
    assert conventional_did.parameters["bootstrap_iterations"].default == 999
    assert conventional_did.parameters["simultaneous_level"].default == 0.95

    efficient_did = inspect.signature(causekit.EfficientDiD)
    assert efficient_did.parameters["pre_periods"].default == "all"
    assert efficient_did.parameters["anticipation"].default == 0
    assert efficient_did.parameters["covariance"].default == "robust"
    assert efficient_did.parameters["inference"].default == "analytic"
    assert efficient_did.parameters["nuisance_probability_floor"].default == 1e-6

    repeated_cross_section_did = inspect.signature(causekit.RepeatedCrossSectionDiD)
    assert repeated_cross_section_did.parameters["control_group"].default == "never_treated"
    assert repeated_cross_section_did.parameters["composition"].default == "stationary"
    assert repeated_cross_section_did.parameters["anticipation"].default == 0
    assert repeated_cross_section_did.parameters["covariance"].default == "robust"
    assert repeated_cross_section_did.parameters["inference"].default == "analytic"
    assert repeated_cross_section_did.parameters["bootstrap_iterations"].default == 999
    assert repeated_cross_section_did.parameters["random_state"].default is None
    assert repeated_cross_section_did.parameters["simultaneous_level"].default == 0.95
    assert repeated_cross_section_did.parameters["nuisance_probability_floor"].default == 1e-6
    repeated_cross_section_fit = inspect.signature(causekit.RepeatedCrossSectionDiD.fit)
    assert "entity" not in repeated_cross_section_fit.parameters
    assert "panel" not in repeated_cross_section_fit.parameters
    assert repeated_cross_section_fit.parameters["cross_fitter"].default is None

    partially_linear_dml = inspect.signature(causekit.PartiallyLinearDML)
    assert partially_linear_dml.parameters["outcome_factory"].default is None
    assert partially_linear_dml.parameters["treatment_factory"].default is None
    assert partially_linear_dml.parameters["n_splits"].default == 5
    assert partially_linear_dml.parameters["covariance"].default == "robust"

    rlearner = inspect.signature(causekit.RLearner)
    assert rlearner.parameters["outcome_factory"].default is None
    assert rlearner.parameters["propensity_factory"].default is None
    assert rlearner.parameters["cate_factory"].default is None
    assert rlearner.parameters["n_splits"].default == 5
    assert rlearner.parameters["evaluation_fraction"].default == 0.5
    assert rlearner.parameters["overlap_floor"].default == 0.01
    assert rlearner.parameters["covariance"].default == "robust"
    assert rlearner.parameters["calibration_groups"].default == 5
    assert rlearner.parameters["bootstrap_iterations"].default == 999

    drlearner = inspect.signature(causekit.DRLearner)
    assert drlearner.parameters["outcome_factory"].default is None
    assert drlearner.parameters["propensity_factory"].default is None
    assert drlearner.parameters["cate_factory"].default is None
    assert drlearner.parameters["n_splits"].default == 5
    assert drlearner.parameters["evaluation_fraction"].default == 0.5
    assert drlearner.parameters["overlap_floor"].default == 0.01
    assert drlearner.parameters["covariance"].default == "robust"
    assert drlearner.parameters["calibration_groups"].default == 5
    assert drlearner.parameters["bootstrap_iterations"].default == 999

    nonlinear_cate = inspect.signature(causekit.NativeSplineRidgeCATE)
    assert nonlinear_cate.parameters["knot_counts"].default == (0, 1, 3)
    assert nonlinear_cate.parameters["include_pairwise_interactions"].default is False
    assert nonlinear_cate.parameters["max_basis_features"].default == 512


def test_fit_returns_the_public_result_type() -> None:
    rng = np.random.default_rng(92)
    nobs = 120
    index = pd.RangeIndex(nobs)
    instrument = rng.normal(size=nobs)
    treatment = instrument + rng.normal(size=nobs)
    outcome = 1.0 + 2.0 * treatment + rng.normal(size=nobs)

    result = IV2SLS().fit(
        pd.Series(outcome, index=index),
        endogenous=pd.DataFrame({"treatment": treatment}, index=index),
        instruments=pd.DataFrame({"instrument": instrument}, index=index),
    )

    assert isinstance(result, IV2SLSResult)
