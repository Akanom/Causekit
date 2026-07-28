"""Optional Universal Output Hub adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..iv import IV2SLSResult
from ..observational import ObservationalATEResult
from ..randomized import RandomizedATEResult


def _regression_model_class() -> Any:
    try:
        from universal_output_hub import RegressionModel
    except ImportError as error:
        raise ImportError(
            "Universal Output Hub is required for this integration. "
            "Install causalkit with the 'outputhub' extra."
        ) from error
    return RegressionModel


def _first_stage_table(result: IV2SLSResult) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [diagnostic.to_dict() for diagnostic in result.first_stage.values()]
    ).set_index("endogenous")


def to_outputhub_model(
    result: IV2SLSResult | RandomizedATEResult | ObservationalATEResult,
    *,
    name: str | None = None,
) -> Any:
    """Convert a fitted causal result into Output Hub's canonical regression model."""

    if not isinstance(result, (IV2SLSResult, RandomizedATEResult, ObservationalATEResult)):
        raise TypeError(
            "result must be an IV2SLSResult, RandomizedATEResult, or ObservationalATEResult."
        )
    RegressionModel = _regression_model_class()
    if isinstance(result, ObservationalATEResult):
        return RegressionModel(
            name=name or result.estimator.upper(),
            depvar="outcome",
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "N": result.nobs,
                "Treated": result.n_treated,
                "Control": result.n_control,
                "Converged": result.converged,
            },
            diagnostics={
                "Propensity minimum": result.overlap.propensity_min,
                "Propensity maximum": result.overlap.propensity_max,
                "Treated effective N": result.overlap.treated_effective_sample_size,
                "Control effective N": result.overlap.control_effective_sample_size,
                "Clipped observations": result.clipped_observations,
            },
            metadata={
                "estimator": result.estimator,
                "backend": result.backend,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "n_clusters": result.n_clusters,
                "nuisance_predictions_supplied": True,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causalkit",
        )
    if isinstance(result, RandomizedATEResult):
        return RegressionModel(
            name=name or "Randomized ATE",
            depvar=result.y_name,
            params=pd.Series({"ate": result.estimate}, name="coef"),
            std_errors=pd.Series({"ate": result.standard_error}, name="se"),
            pvalues=pd.Series({"ate": result.pvalue}, name="pvalue"),
            statistics={
                "N": result.nobs,
                "Treated": result.n_treated,
                "Control": result.n_control,
                "Residual df": result.df_resid,
                "Converged": result.converged,
            },
            diagnostics={
                f"Absolute standardized difference ({item.covariate})": abs(
                    item.standardized_difference
                )
                for item in result.balance
            },
            metadata={
                "estimator": "randomized_ate",
                "backend": result.backend,
                "adjustment": result.adjustment,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "n_clusters": result.n_clusters,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causalkit",
        )
    diagnostics: dict[str, Any] = {}
    for endogenous, first_stage in result.first_stage.items():
        diagnostics[f"First-stage F ({endogenous})"] = first_stage.classical_f_statistic
        diagnostics[f"Partial R2 ({endogenous})"] = first_stage.partial_r_squared
    if result.overidentification is not None:
        diagnostics.update(
            {
                "Sargan statistic": result.overidentification.statistic,
                "Sargan df": result.overidentification.df,
                "Sargan p": result.overidentification.p_value,
            }
        )
    return RegressionModel(
        name=name or "IV/2SLS",
        depvar=result.y_name,
        params=result.params.rename("coef"),
        std_errors=result.standard_errors.rename("se"),
        pvalues=result.pvalues.rename("pvalue"),
        statistics={
            "N": result.nobs,
            "Residual df": result.df_resid,
            "Converged": result.converged,
        },
        diagnostics=diagnostics,
        metadata={
            "estimator": "iv_2sls",
            "backend": result.backend,
            "covariance_type": result.covariance_type,
            "inference_distribution": result.inference_distribution,
            "n_clusters": result.n_clusters,
            "endogenous": list(result.endogenous_names),
            "excluded_instruments": list(result.instrument_names),
            "causal_interpretation_requires_assumptions": True,
            "assumptions": list(result.assumptions),
        },
        source="causalkit",
    )


def add_to_outputhub(
    hub: Any,
    result: IV2SLSResult | RandomizedATEResult | ObservationalATEResult,
    *,
    name: str | None = None,
) -> Any:
    """Add an IV model and its first-stage diagnostics to an OutputHub."""

    if not hasattr(hub, "add_model"):
        raise TypeError("hub must provide an OutputHub-compatible add_model method.")
    model_name = name or (
        result.estimator.upper()
        if isinstance(result, ObservationalATEResult)
        else "Randomized ATE"
        if isinstance(result, RandomizedATEResult)
        else "IV/2SLS"
    )
    model = to_outputhub_model(result, name=model_name)
    hub.add_model(model)
    if isinstance(result, IV2SLSResult) and hasattr(hub, "add_table"):
        hub.add_table(
            f"{model_name} first-stage diagnostics",
            _first_stage_table(result).reset_index(),
            caption=(
                "First-stage fit and excluded-instrument relevance diagnostics. "
                "The classical F < 10 warning is a heuristic, not a universal "
                "weak-identification test."
            ),
            metadata={"source": "causalkit", "estimator": "iv_2sls"},
        )
    elif isinstance(result, RandomizedATEResult) and result.balance and hasattr(hub, "add_table"):
        hub.add_table(
            f"{model_name} covariate balance",
            pd.DataFrame([item.__dict__ for item in result.balance]),
            caption="Unadjusted pre-treatment covariate balance by randomized arm.",
            metadata={"source": "causalkit", "estimator": "randomized_ate"},
        )
    return model


__all__ = ["add_to_outputhub", "to_outputhub_model"]
