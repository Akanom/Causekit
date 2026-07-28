"""Identification-aware causal inference and instrumental-variable workflows."""

from .crossfit import (
    ClassProbabilityCrossFitResult,
    CrossFitResult,
    CrossFitTask,
    CrossFitTaskResult,
    CrossFitter,
    NuisanceEstimatorProtocol,
    OutcomeResultProtocol,
    PropensityResultProtocol,
)
from .diagnostics import FirstStageDiagnostic, SarganTest
from .did import DiDResult, DifferenceInDifferences, EfficientDiD
from .integrations import add_to_outputhub, to_outputhub_model
from .iv import IV2SLS, IV2SLSResult
from .matching import NearestNeighborMatch, NearestNeighborMatchResult, PropensityScoreStatus
from .observational import AIPWATE, IPWATE, ObservationalATEResult, OverlapDiagnostic
from .postestimation import (
    confint,
    fitted_values,
    lincom,
    predict,
    residuals,
    summary_frame,
    vcov,
    wald_test,
)
from .randomized import CovariateBalance, RandomizedATE, RandomizedATEResult

__all__ = [
    "FirstStageDiagnostic",
    "CovariateBalance",
    "ClassProbabilityCrossFitResult",
    "CrossFitResult",
    "CrossFitTask",
    "CrossFitTaskResult",
    "CrossFitter",
    "DiDResult",
    "DifferenceInDifferences",
    "EfficientDiD",
    "AIPWATE",
    "IV2SLS",
    "IV2SLSResult",
    "IPWATE",
    "ObservationalATEResult",
    "OverlapDiagnostic",
    "NuisanceEstimatorProtocol",
    "NearestNeighborMatch",
    "NearestNeighborMatchResult",
    "OutcomeResultProtocol",
    "PropensityResultProtocol",
    "PropensityScoreStatus",
    "SarganTest",
    "RandomizedATE",
    "RandomizedATEResult",
    "add_to_outputhub",
    "confint",
    "fitted_values",
    "lincom",
    "predict",
    "residuals",
    "summary_frame",
    "to_outputhub_model",
    "vcov",
    "wald_test",
]

__version__ = "0.6.0a3"
