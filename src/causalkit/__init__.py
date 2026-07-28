"""Identification-aware causal inference and instrumental-variable workflows."""

from .crossfit import (
    CrossFitResult,
    CrossFitter,
    NuisanceEstimatorProtocol,
    OutcomeResultProtocol,
    PropensityResultProtocol,
)
from .diagnostics import FirstStageDiagnostic, SarganTest
from .integrations import add_to_outputhub, to_outputhub_model
from .iv import IV2SLS, IV2SLSResult
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
    "CrossFitResult",
    "CrossFitter",
    "AIPWATE",
    "IV2SLS",
    "IV2SLSResult",
    "IPWATE",
    "ObservationalATEResult",
    "OverlapDiagnostic",
    "NuisanceEstimatorProtocol",
    "OutcomeResultProtocol",
    "PropensityResultProtocol",
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

__version__ = "0.4.0a1"
