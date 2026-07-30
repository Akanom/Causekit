"""Optional Universal Output Hub adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..did import DiDResult
from ..did_rcs import RepeatedCrossSectionDiDResult
from ..iv import IV2SLSResult
from ..matching import NearestNeighborMatchResult
from ..ml import DRLearnerResult, PartiallyLinearDMLResult, RLearnerResult
from ..observational import ObservationalATEResult
from ..randomized import RandomizedATEResult


def _regression_model_class() -> Any:
    try:
        from universal_output_hub import RegressionModel
    except ImportError as error:
        raise ImportError(
            "Universal Output Hub is required for this integration. "
            "Install causekit with the 'outputhub' extra."
        ) from error
    return RegressionModel


def _first_stage_table(result: IV2SLSResult) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [diagnostic.to_dict() for diagnostic in result.first_stage.values()]
    ).set_index("endogenous")


def to_outputhub_model(
    result: (
        IV2SLSResult
        | RandomizedATEResult
        | ObservationalATEResult
        | DiDResult
        | RepeatedCrossSectionDiDResult
        | NearestNeighborMatchResult
        | PartiallyLinearDMLResult
        | DRLearnerResult
        | RLearnerResult
    ),
    *,
    name: str | None = None,
) -> Any:
    """Convert a fitted causal result into Output Hub's canonical regression model."""

    if not isinstance(
        result,
        (
            IV2SLSResult,
            RandomizedATEResult,
            ObservationalATEResult,
            DiDResult,
            RepeatedCrossSectionDiDResult,
            NearestNeighborMatchResult,
            PartiallyLinearDMLResult,
            DRLearnerResult,
            RLearnerResult,
        ),
    ):
        raise TypeError(
            "result must be an IV2SLSResult, RandomizedATEResult, "
            "ObservationalATEResult, DiDResult, RepeatedCrossSectionDiDResult, "
            "NearestNeighborMatchResult, or "
            "PartiallyLinearDMLResult, DRLearnerResult, or RLearnerResult."
        )
    RegressionModel = _regression_model_class()
    if isinstance(result, NearestNeighborMatchResult):
        return RegressionModel(
            name=name or "Nearest-neighbor matching",
            depvar="outcome",
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "N": result.nobs,
                "Treated": result.n_treated,
                "Control": result.n_control,
                "Matched focal observations": result.n_matched_focal,
                "Converged": result.converged,
            },
            diagnostics={
                "Matched focal fraction": result.matched_focal_fraction,
                "Maximum comparison reuse": result.maximum_reuse_count,
                "Boundary tie events": result.boundary_tie_events,
                "Known-score variance": result.known_score_variance,
                "First-step variance adjustment": result.first_step_variance_adjustment,
                "Propensity likelihood score norm": result.propensity_model_score_norm,
                **{
                    f"{name.replace('_', ' ').title()}": value
                    for name, value in result.balance_summary.items()
                },
            },
            metadata={
                "estimator": "nearest_neighbor_match",
                "backend": result.backend,
                "requested_estimand": result.requested_estimand,
                "realized_estimand": result.realized_estimand,
                "target_population": result.target_population,
                "metric": result.metric,
                "neighbors": result.neighbors,
                "replacement": result.replacement,
                "ties": result.ties,
                "caliper": result.requested_caliper,
                "common_support": result.common_support,
                "propensity_provenance": result.propensity_provenance,
                "propensity_score_status": result.propensity_score_status,
                "inference": result.inference,
                "variance_neighbors": result.variance_neighbors,
                "first_step_covariance_neighbors": result.first_step_covariance_neighbors,
                "first_step_regression_neighbors": result.first_step_regression_neighbors,
                "first_step_covariate_neighbors": result.first_step_covariate_neighbors,
                "propensity_model": result.propensity_model_name,
                "propensity_link": result.propensity_link,
                "inference_distribution": result.inference_distribution,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causekit",
        )
    if isinstance(result, RepeatedCrossSectionDiDResult):
        return RegressionModel(
            name=name or "Repeated-cross-section DiD",
            depvar=result.outcome_name,
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "Observations": result.nobs,
                "Periods": result.n_periods,
                "Treated cohorts": len(result.cohort_sizes),
                "Converged": result.converged,
            },
            diagnostics={
                "Group-time effects": len(result.group_time),
                "Event-time effects": len(result.event_study),
                "Pre-trend restrictions": result.pretrend.n_restrictions,
                **(
                    {
                        "Pre-trend joint statistic": result.pretrend.statistic,
                        "Pre-trend joint p-value": result.pretrend.pvalue,
                    }
                    if result.pretrend.available
                    else {}
                ),
            },
            metadata={
                "estimator": result.method,
                "backend": result.backend,
                "sampling_unit": result.sampling_unit,
                "parallel_trends": result.parallel_trends,
                "control_group": result.control_group,
                "composition": result.composition,
                "composition_verified": False,
                "anticipation": result.anticipation,
                "pre_periods": result.pre_periods,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "inference_method": result.inference_method,
                "n_clusters": result.n_clusters,
                "covariates": list(result.covariates),
                "nuisance_cross_fitted": result.cross_fitted,
                "pretrend_available": result.pretrend.available,
                "pretrend_unavailable_reason": result.pretrend.reason,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causekit",
        )
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
                "Pre-trend restrictions": result.pretrend.n_restrictions,
                **(
                    {
                        "Pre-trend joint statistic": result.pretrend.statistic,
                        "Pre-trend joint p-value": result.pretrend.pvalue,
                    }
                    if result.pretrend.available
                    else {}
                ),
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
                "pretrend_available": result.pretrend.available,
                "pretrend_unavailable_reason": result.pretrend.reason,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causekit",
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
            source="causekit",
        )
    if isinstance(result, PartiallyLinearDMLResult):
        boundary_fits = (
            int(result.nuisance_diagnostics["alpha_at_boundary"].sum())
            if "alpha_at_boundary" in result.nuisance_diagnostics
            else None
        )
        return RegressionModel(
            name=name or "Partially linear DML",
            depvar="outcome",
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "N": result.nobs,
                "Outer folds": result.n_splits,
                "Converged": result.converged,
            },
            diagnostics={
                "Residual treatment second moment": (result.residual_treatment_second_moment),
                "Residual treatment tolerance": result.residual_treatment_tolerance,
                "Orthogonal score mean": result.orthogonal_score_mean,
                "Nuisance alpha boundary fits": boundary_fits,
            },
            metadata={
                "estimator": result.estimator,
                "estimand": result.estimand,
                "backend": result.backend,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "n_clusters": result.n_clusters,
                "n_splits": result.n_splits,
                "random_state": result.random_state,
                "treatment_kind": result.treatment_kind,
                "native_nuisance": result.native_nuisance,
                "outcome_model": result.outcome_model_name,
                "treatment_model": result.treatment_model_name,
                "nuisance_cross_fitted": True,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causekit",
        )
    if isinstance(result, RLearnerResult):
        return RegressionModel(
            name=name or "Honest R-learner",
            depvar="outcome",
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "N": result.nobs,
                "Construction N": result.construction_nobs,
                "Evaluation N": result.evaluation_nobs,
                "Outer folds": result.n_splits,
                "Calibration groups": result.calibration_groups,
                "Converged": result.converged,
            },
            diagnostics={
                "Honest R-loss": result.honest_r_loss,
                "Honest constant R-loss": result.honest_constant_r_loss,
                "R-loss gain": result.r_loss_gain,
                "Construction constant effect": result.construction_constant_effect,
                "Calibration center": result.calibration_center,
                "Residual treatment second moment": result.residual_treatment_second_moment,
            },
            metadata={
                "estimator": result.estimator,
                "estimand": result.estimand,
                "backend": result.backend,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "n_clusters": result.n_clusters,
                "n_splits": result.n_splits,
                "split_seed": result.split_seed,
                "requested_evaluation_fraction": result.requested_evaluation_fraction,
                "realized_evaluation_fraction": result.realized_evaluation_fraction,
                "split_conditional": result.split_conditional,
                "overlap_floor": result.overlap_floor,
                "simultaneous_level": result.simultaneous_level,
                "simultaneous_critical_value": result.simultaneous_critical_value,
                "bootstrap_iterations": result.bootstrap_iterations,
                "bootstrap_random_state": result.bootstrap_random_state,
                "native_outcome": result.native_outcome,
                "native_propensity": result.native_propensity,
                "native_cate": result.native_cate,
                "outcome_model": result.outcome_model_name,
                "propensity_model": result.propensity_model_name,
                "cate_model": result.cate_model_name,
                "nuisance_cross_fitted": True,
                "evaluation_used_for_fitting": False,
                "unit_level_intervals": False,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causekit",
        )
    if isinstance(result, DRLearnerResult):
        return RegressionModel(
            name=name or "Honest DR learner",
            depvar="outcome",
            params=result.params.rename("coef"),
            std_errors=result.standard_errors.rename("se"),
            pvalues=result.pvalues.rename("pvalue"),
            statistics={
                "N": result.nobs,
                "Construction N": result.construction_nobs,
                "Evaluation N": result.evaluation_nobs,
                "Outer folds": result.n_splits,
                "Calibration groups": result.calibration_groups,
                "Converged": result.converged,
            },
            diagnostics={
                "Honest DR loss": result.honest_dr_loss,
                "Honest constant DR loss": result.honest_constant_dr_loss,
                "DR loss gain": result.dr_loss_gain,
                "Construction constant effect": result.construction_constant_effect,
                "Calibration center": result.calibration_center,
                "Minimum propensity": result.minimum_propensity,
                "Maximum propensity": result.maximum_propensity,
            },
            metadata={
                "estimator": result.estimator,
                "estimand": result.estimand,
                "backend": result.backend,
                "covariance_type": result.covariance_type,
                "inference_distribution": result.inference_distribution,
                "n_clusters": result.n_clusters,
                "n_splits": result.n_splits,
                "split_seed": result.split_seed,
                "requested_evaluation_fraction": result.requested_evaluation_fraction,
                "realized_evaluation_fraction": result.realized_evaluation_fraction,
                "split_conditional": result.split_conditional,
                "overlap_floor": result.overlap_floor,
                "simultaneous_level": result.simultaneous_level,
                "simultaneous_critical_value": result.simultaneous_critical_value,
                "bootstrap_iterations": result.bootstrap_iterations,
                "bootstrap_random_state": result.bootstrap_random_state,
                "native_outcome": result.native_outcome,
                "native_propensity": result.native_propensity,
                "native_cate": result.native_cate,
                "outcome_control_model": result.outcome_control_model_name,
                "outcome_treated_model": result.outcome_treated_model_name,
                "propensity_model": result.propensity_model_name,
                "cate_model": result.cate_model_name,
                "nuisance_cross_fitted": True,
                "evaluation_used_for_fitting": False,
                "unit_level_intervals": False,
                "causal_interpretation_requires_assumptions": True,
                "assumptions": list(result.assumptions),
            },
            source="causekit",
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
            source="causekit",
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
        source="causekit",
    )


def add_to_outputhub(
    hub: Any,
    result: (
        IV2SLSResult
        | RandomizedATEResult
        | ObservationalATEResult
        | DiDResult
        | RepeatedCrossSectionDiDResult
        | NearestNeighborMatchResult
        | PartiallyLinearDMLResult
        | DRLearnerResult
        | RLearnerResult
    ),
    *,
    name: str | None = None,
) -> Any:
    """Add a fitted causal model and any supported diagnostic tables to OutputHub."""

    if not hasattr(hub, "add_model"):
        raise TypeError("hub must provide an OutputHub-compatible add_model method.")
    model_name = name or (
        result.estimator.upper()
        if isinstance(result, ObservationalATEResult)
        else "Partially linear DML"
        if isinstance(result, PartiallyLinearDMLResult)
        else "Honest R-learner"
        if isinstance(result, RLearnerResult)
        else "Honest DR learner"
        if isinstance(result, DRLearnerResult)
        else "Nearest-neighbor matching"
        if isinstance(result, NearestNeighborMatchResult)
        else "Repeated-cross-section DiD"
        if isinstance(result, RepeatedCrossSectionDiDResult)
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
            metadata={"source": "causekit", "estimator": "iv_2sls"},
        )
    elif isinstance(result, NearestNeighborMatchResult) and hasattr(hub, "add_table"):
        table_metadata = {"source": "causekit", "estimator": "nearest_neighbor_match"}
        hub.add_table(
            f"{model_name} matches",
            result.match_table.copy(),
            caption=(
                "Focal-to-comparison design with distance, rank, tie group, and fractional "
                "match weight; outcome values are intentionally excluded."
            ),
            metadata=table_metadata,
        )
        if not result.balance.empty:
            hub.add_table(
                f"{model_name} balance",
                result.balance.reset_index(),
                caption="Pre-treatment covariate balance before and after matching.",
                metadata=table_metadata,
            )
    elif isinstance(result, PartiallyLinearDMLResult) and hasattr(hub, "add_table"):
        hub.add_table(
            f"{model_name} nuisance tuning",
            result.nuisance_diagnostics.copy(),
            caption=(
                "Fold-local nuisance tuning and training diagnostics; every row was fitted "
                "without its corresponding holdout fold."
            ),
            metadata={"source": "causekit", "estimator": result.estimator},
        )
    elif isinstance(result, RLearnerResult) and hasattr(hub, "add_table"):
        table_metadata = {"source": "causekit", "estimator": result.estimator}
        hub.add_table(
            f"{model_name} honest loss",
            pd.DataFrame(
                {
                    "value": [
                        result.honest_r_loss,
                        result.honest_constant_r_loss,
                        result.r_loss_gain,
                    ]
                },
                index=["r_loss", "constant_r_loss", "r_loss_gain"],
            ).reset_index(names="metric"),
            caption=(
                "Held-out R-loss against a constant effect fitted only on construction; "
                "the gain is not ordinary predictive R-squared."
            ),
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} calibration tests",
            result.calibration_tests.reset_index(names="hypothesis"),
            caption="Split-conditional differential-calibration tests on honest evaluation data.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} calibration groups",
            result.group_effects.reset_index(),
            caption=(
                "Tie-preserving overlap-weighted group effects with pointwise intervals and "
                "studentized multiplier-bootstrap simultaneous bands."
            ),
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} nuisance tuning",
            result.nuisance_diagnostics.copy(),
            caption="Construction-only outer-fold and full-construction nuisance diagnostics.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} CATE tuning",
            result.cate_diagnostics.copy(),
            caption="Construction-only weighted CATE learner diagnostics.",
            metadata=table_metadata,
        )
    elif isinstance(result, DRLearnerResult) and hasattr(hub, "add_table"):
        table_metadata = {"source": "causekit", "estimator": result.estimator}
        hub.add_table(
            f"{model_name} honest loss",
            pd.DataFrame(
                {
                    "value": [
                        result.honest_dr_loss,
                        result.honest_constant_dr_loss,
                        result.dr_loss_gain,
                    ]
                },
                index=["dr_loss", "constant_dr_loss", "dr_loss_gain"],
            ).reset_index(names="metric"),
            caption=(
                "Held-out DR-score loss against a constant fitted only on construction; "
                "the pseudo-outcome is not observed unit-level effect truth."
            ),
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} calibration tests",
            result.calibration_tests.reset_index(names="hypothesis"),
            caption="Split-conditional DR-score calibration tests on honest evaluation data.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} calibration groups",
            result.group_effects.reset_index(),
            caption=(
                "Tie-preserving mean DR group scores with pointwise intervals and "
                "studentized multiplier-bootstrap simultaneous bands."
            ),
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} nuisance tuning",
            result.nuisance_diagnostics.copy(),
            caption="Construction-only arm outcome and propensity nuisance diagnostics.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} CATE tuning",
            result.cate_diagnostics.copy(),
            caption="Construction-only unweighted DR pseudo-outcome regression diagnostics.",
            metadata=table_metadata,
        )
    elif isinstance(result, RandomizedATEResult) and result.balance and hasattr(hub, "add_table"):
        hub.add_table(
            f"{model_name} covariate balance",
            pd.DataFrame([item.__dict__ for item in result.balance]),
            caption="Unadjusted pre-treatment covariate balance by randomized arm.",
            metadata={"source": "causekit", "estimator": "randomized_ate"},
        )
    elif isinstance(result, RepeatedCrossSectionDiDResult) and hasattr(hub, "add_table"):
        table_metadata = {"source": "causekit", "estimator": result.method}
        if not result.pretrend.placebo_effects.empty:
            hub.add_table(
                f"{model_name} pre-trend placebos",
                result.pretrend.placebo_effects.reset_index(),
                caption=(
                    "Independent-cell adjacent pre-period placebos. Failure to reject "
                    "does not prove parallel trends or stationary composition."
                ),
                metadata={
                    **table_metadata,
                    "joint_test_available": result.pretrend.available,
                    "joint_test_statistic": result.pretrend.statistic,
                    "joint_test_pvalue": result.pretrend.pvalue,
                },
            )
        hub.add_table(
            f"{model_name} cell counts",
            result.cell_counts.reset_index(),
            caption="Observed cohort-period support used to audit repeated samples.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} group-time effects",
            result.group_time.reset_index(),
            caption="Four-cell cohort-time effects with pointwise observation/PSU inference.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} event study",
            result.event_study.reset_index(),
            caption="Pooled-cohort-share-weighted event-time effects with pointwise inference.",
            metadata=table_metadata,
        )
        hub.add_table(
            f"{model_name} calendar-time effects",
            result.calendar_time.reset_index(),
            caption="Pooled-cohort-share-weighted post-adoption calendar-time effects.",
            metadata=table_metadata,
        )
    elif isinstance(result, DiDResult) and hasattr(hub, "add_table"):
        table_metadata = {"source": "causekit", "estimator": result.method}
        if not result.pretrend.placebo_effects.empty:
            hub.add_table(
                f"{model_name} pre-trend placebos",
                result.pretrend.placebo_effects.reset_index(),
                caption=(
                    "Uncontaminated adjacent pre-period placebo effects. The joint test "
                    "does not prove parallel trends when it fails to reject."
                ),
                metadata={
                    **table_metadata,
                    "joint_test_available": result.pretrend.available,
                    "joint_test_statistic": result.pretrend.statistic,
                    "joint_test_pvalue": result.pretrend.pvalue,
                },
            )
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
