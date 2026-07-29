"""Identification-aware causal inference and instrumental-variable workflows."""

from .crossfit import (
    CATEEstimatorProtocol,
    CATEResultProtocol,
    ClassProbabilityCrossFitResult,
    CrossFitResult,
    CrossFitTask,
    CrossFitTaskResult,
    CrossFitter,
    NuisanceDiagnosticsProtocol,
    NuisanceEstimatorProtocol,
    OutcomeResultProtocol,
    PropensityResultProtocol,
    WeightedCATEEstimatorProtocol,
)
from .diagnostics import FirstStageDiagnostic, SarganTest
from .did import DiDResult, DifferenceInDifferences, EfficientDiD
from .integrations import add_to_outputhub, to_outputhub_model
from .iv import IV2SLS, IV2SLSResult
from .matching import (
    FittedPropensityMLEProtocol,
    NearestNeighborMatch,
    NearestNeighborMatchResult,
    PropensityScoreStatus,
)
from .ml import (
    DRLearner,
    DRLearnerResult,
    NativeSplineRidgeCATE,
    NativeSplineRidgeCATEResult,
    PartiallyLinearDML,
    PartiallyLinearDMLResult,
    RLearner,
    RLearnerResult,
)
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
    "CATEEstimatorProtocol",
    "CATEResultProtocol",
    "FirstStageDiagnostic",
    "CovariateBalance",
    "ClassProbabilityCrossFitResult",
    "CrossFitResult",
    "CrossFitTask",
    "CrossFitTaskResult",
    "CrossFitter",
    "DiDResult",
    "DifferenceInDifferences",
    "DRLearner",
    "DRLearnerResult",
    "EfficientDiD",
    "FittedPropensityMLEProtocol",
    "AIPWATE",
    "IV2SLS",
    "IV2SLSResult",
    "IPWATE",
    "ObservationalATEResult",
    "OverlapDiagnostic",
    "NuisanceEstimatorProtocol",
    "NuisanceDiagnosticsProtocol",
    "NearestNeighborMatch",
    "NearestNeighborMatchResult",
    "NativeSplineRidgeCATE",
    "NativeSplineRidgeCATEResult",
    "OutcomeResultProtocol",
    "PartiallyLinearDML",
    "PartiallyLinearDMLResult",
    "RLearner",
    "RLearnerResult",
    "PropensityResultProtocol",
    "PropensityScoreStatus",
    "WeightedCATEEstimatorProtocol",
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

__version__ = "0.7.0a3"
