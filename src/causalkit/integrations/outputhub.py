"""Optional Universal Output Hub adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..iv import IV2SLSResult


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
    result: IV2SLSResult,
    *,
    name: str | None = None,
) -> Any:
    """Convert a fitted IV result into Output Hub's canonical regression model."""

    if not isinstance(result, IV2SLSResult):
        raise TypeError("result must be an IV2SLSResult.")
    RegressionModel = _regression_model_class()
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
    result: IV2SLSResult,
    *,
    name: str | None = None,
) -> Any:
    """Add an IV model and its first-stage diagnostics to an OutputHub."""

    if not hasattr(hub, "add_model"):
        raise TypeError("hub must provide an OutputHub-compatible add_model method.")
    model_name = name or "IV/2SLS"
    model = to_outputhub_model(result, name=model_name)
    hub.add_model(model)
    if hasattr(hub, "add_table"):
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
    return model


__all__ = ["add_to_outputhub", "to_outputhub_model"]
