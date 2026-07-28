"""Optional Universal Output Hub adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..did import DiDResult
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
    result: IV2SLSResult | RandomizedATEResult | ObservationalATEResult | DiDResult,
    *,
    name: str | None = None,
) -> Any:
    """Convert a fitted causal result into Output Hub's canonical regression model."""

    if not isinstance(
        result, (IV2SLSResult, RandomizedATEResult, ObservationalATEResult, DiDResult)
    ):
        raise TypeError(
            "result must be an IV2SLSResult, RandomizedATEResult, "
            "ObservationalATEResult, or DiDResult."
        )
    RegressionModel = _regression_model_class()
    if isinstance(result, DiDResult):
        negative_weights = (
            int((result.efficiency_weights["weight"] < 0).sum())
            if not result.efficiency_weights.empty
            else 0
        )
        return RegressionModel(
            name=name
            or (
                "Efficient DiD"
                if result.method.startswith("chen_santanna_xie_efficient")
                else "Difference-in-Differences"
            ),
            depvar=result.outcome_name,
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "Entities": result.n_entities,
                "Periods": result.n_periods,
                "Treated cohorts": len(result.cohort_sizes),
                "Converged": result.converged,
            },
            diagnostics={
                "Group-time effects": len(result.group_time),
                "Event-time effects": len(result.event_study),
                "Negative efficiency weights": negative_weights,
            },
            metadata={
                "estimator": result.method,
                "backend": result.backend,
                "parallel_trends": result.parallel_trends,
                "control_group": result.control_group,
                "anticipation": result.anticipation,
                "pre_periods": result.pre_periods,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "inference_method": result.inference_method,
                "simultaneous_level": result.simultaneous_level,
                "simultaneous_critical_value": result.simultaneous_critical_value,
                "n_clusters": result.n_clusters,
                "covariates": list(result.covariates),
                "nuisance_cross_fitted": result.cross_fitted,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causalkit",
        )
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
                "estimand": result.estimand,
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
    result: IV2SLSResult | RandomizedATEResult | ObservationalATEResult | DiDResult,
    *,
    name: str | None = None,
) -> Any:
    """Add an IV model and its first-stage diagnostics to an OutputHub."""

    if not hasattr(hub, "add_model"):
        raise TypeError("hub must provide an OutputHub-compatible add_model method.")
    model_name = name or (
        result.estimator.upper()
        if isinstance(result, ObservationalATEResult)
        else "Efficient DiD"
        if isinstance(result, DiDResult) and result.method.startswith("chen_santanna_xie_efficient")
        else "Difference-in-Differences"
        if isinstance(result, DiDResult)
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
    elif isinstance(result, DiDResult) and hasattr(hub, "add_table"):
        table_metadata = {"source": "causalkit", "estimator": result.method}
        hub.add_table(
            f"{model_name} group-time effects",
            result.group_time.reset_index(),
            caption="Cohort-time average treatment effects and pointwise inference.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} event study",
            result.event_study.reset_index(),
            caption="Cohort-share-weighted event-time effects with pointwise inference.",
            metadata=table_metadata,
        )
        if not result.simultaneous_event_study.empty:
            hub.add_table(
                f"{model_name} simultaneous event-study bands",
                result.simultaneous_event_study.reset_index(),
                caption=(
                    "Studentized multiplier-bootstrap max-t bands over the reported "
                    "event-time path."
                ),
                metadata=table_metadata,
            )
        hub.add_table(
            f"{model_name} calendar-time effects",
            result.calendar_time.reset_index(),
            caption="Cohort-share-weighted post-adoption calendar-time effects.",
            metadata=table_metadata,
        )
        if not result.efficiency_weights.empty:
            hub.add_table(
                f"{model_name} efficiency weights",
                result.efficiency_weights.copy(),
                caption=(
                    "Realized PT-All generated-outcome weights; negative values are "
                    "permitted by the homogeneous-moment contract."
                ),
                metadata=table_metadata,
            )
    return model


__all__ = ["add_to_outputhub", "to_outputhub_model"]
