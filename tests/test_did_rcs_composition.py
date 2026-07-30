"""Failing-first contracts for composition-robust repeated-section DiD."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from causekit import (
    CrossFitter,
    RepeatedCrossSectionDiD,
    add_to_outputhub,
    to_outputhub_model,
)


class _KnownGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X["x"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                0: 0.40 - 0.03 * x,
                1: 0.20 + 0.02 * x,
                2: 0.25 - 0.01 * x,
                3: 0.15 + 0.02 * x,
            },
            index=X.index,
        )


class _KnownGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownGeneralizedPropensityResult:
        del X, y
        return _KnownGeneralizedPropensityResult()


class _KnownCellOutcomeResult:
    def __init__(self, cell: str) -> None:
        self.cell = cell

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X["x"].to_numpy(dtype=float)
        if self.cell == "d0-s0":
            return 1.0 + 0.50 * x
        if self.cell == "d0-s1":
            return 2.0 + x
        if self.cell == "d1-s0":
            return 3.0 + 0.25 * x
        raise AssertionError(f"Unexpected outcome cell {self.cell!r}.")


class _KnownCellOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownCellOutcomeResult:
        del y
        cells = {str(label).rsplit("-r", 1)[0] for label in X.index}
        if len(cells) != 1:
            raise AssertionError("Each robust outcome nuisance must train within one cell.")
        return _KnownCellOutcomeResult(cells.pop())


def _cross_fitter(
    *,
    propensity_factory: Callable[[], object] = _KnownGeneralizedPropensity,
    outcome_factory: Callable[[], object] = _KnownCellOutcome,
    random_state: int = 19,
) -> CrossFitter:
    return CrossFitter(
        propensity_factory=propensity_factory,
        outcome_factory=outcome_factory,
        n_splits=2,
        random_state=random_state,
    )


def _composition_hand_sample() -> pd.DataFrame:
    x = np.tile(np.arange(4.0), 4)
    treated = np.repeat([0.0, 0.0, 1.0, 1.0], 4)
    post = np.repeat([0.0, 1.0, 0.0, 1.0], 4)
    m00 = 1.0 + 0.50 * x
    m01 = 2.0 + x
    m10 = 3.0 + 0.25 * x
    residual = np.array(
        [
            -0.4,
            0.2,
            0.1,
            0.3,
            0.3,
            -0.2,
            0.5,
            -0.1,
            -0.2,
            0.4,
            -0.3,
            0.1,
            0.1,
            -0.2,
            0.3,
            -0.1,
        ]
    )
    treatment_effect = 2.0 + 0.20 * x
    outcome = np.where(
        (treated == 0.0) & (post == 0.0),
        m00 + residual,
        np.where(
            (treated == 0.0) & (post == 1.0),
            m01 + residual,
            np.where(
                (treated == 1.0) & (post == 0.0),
                m10 + residual,
                m10 + m01 - m00 + treatment_effect + residual,
            ),
        ),
    )
    labels = [
        f"d{int(treated[position])}-s{int(post[position])}-r{position % 4}"
        for position in range(len(x))
    ]
    return pd.DataFrame(
        {
            "outcome": outcome,
            "time": post + 1.0,
            "treatment_time": np.where(treated == 1.0, 2.0, np.inf),
            "x": x,
        },
        index=pd.Index(labels, name="row"),
    )


def _fit_robust(
    data: pd.DataFrame,
    *,
    cross_fitter: CrossFitter | None = None,
    **constructor: object,
):
    return RepeatedCrossSectionDiD(composition="robust", **constructor).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=cross_fitter or _cross_fitter(),
    )


def test_hand_computed_composition_robust_score_eif_hc1_and_weights() -> None:
    data = _composition_hand_sample()

    result = _fit_robust(data)

    assert result.estimate == pytest.approx(2.295038744545108, abs=2e-14)
    expected_influence = pd.Series(
        [
            -1.15902703606356,
            0.710034580651553,
            0.431794385984466,
            1.5702946940216,
            -1.15307512832478,
            0.792011199253384,
            -2.02855809612692,
            0.413924405039663,
            0.621567268423242,
            -1.4675893837771,
            1.28367153261322,
            -0.494428508973033,
            -0.780154978180434,
            -1.18015497818043,
            1.61984502181957,
            0.819845021819567,
        ],
        index=data.index,
        name="esavg_influence",
    )
    pd.testing.assert_series_equal(result.overall_influence, expected_influence, atol=2e-14)
    expected_se = np.sqrt(
        float(expected_influence @ expected_influence) / (len(data) * (len(data) - 1))
    )
    assert result.standard_error == pytest.approx(expected_se, abs=2e-14)

    expected_weights = pd.DataFrame(
        {
            "w_00": [
                2.89756759015891,
                3.55017290325776,
                4.31794385984465,
                5.23431564673868,
                *([0.0] * 12),
            ],
            "w_01": [
                *([0.0] * 4),
                3.84358376108259,
                3.96005599626692,
                4.05711619225385,
                4.13924405039664,
                *([0.0] * 8),
            ],
            "w_10": [
                *([0.0] * 8),
                3.10783634211621,
                3.66897345944274,
                4.27890510871072,
                4.94428508973033,
                *([0.0] * 4),
            ],
            "w_11": [*([0.0] * 12), 4.0, 4.0, 4.0, 4.0],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result.composition_weights, expected_weights, atol=2e-14)
    assert result.composition_weights.mean().eq(1.0).all()
    assert result.composition == "robust"
    assert result.target_population == "treated_target_period"
    assert result.method == "repeated_cross_section_composition_robust_group_time"
    assert result.parallel_trends == (
        "conditional_repeated_cross_section_post_composition_robust_target_period_treated"
    )
    assert result.pretrend.available is False
    assert result.pretrend.conditional is True
    assert result.nuisance_predictions.columns.get_level_values("nuisance").tolist() == [
        "generalized_propensity_00",
        "generalized_propensity_01",
        "generalized_propensity_10",
        "generalized_propensity_11",
        "outcome_control_pre",
        "outcome_control_post",
        "outcome_treated_pre",
    ]
    assert "outcome_treated_post" not in result.nuisance_predictions.columns.get_level_values(
        "nuisance"
    )
    assert len(result.nuisance_diagnostics) == 8
    assert result.nuisance_fold.index.equals(data.index)


def test_robust_composition_refuses_missing_covariates_and_nonpairwise_designs() -> None:
    data = _composition_hand_sample()
    with pytest.raises(ValueError, match="composition='robust'.*covariates"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
        )
    with pytest.raises(ValueError, match="at least one"):
        RepeatedCrossSectionDiD(composition="robust").fit(
            data,
            outcome="outcome",
            time="time",
            treatment_time="treatment_time",
            covariates=[],
            cross_fitter=_cross_fitter(),
        )

    later = data.copy()
    later.index = pd.Index([f"later-{label}" for label in later.index], name="row")
    later["time"] = 3.0
    with pytest.raises(NotImplementedError, match="exactly two observed periods"):
        _fit_robust(pd.concat([data, later]))

    rows: list[dict[str, float | str]] = []
    for cohort in (2.0, 3.0, np.inf):
        for period in (1.0, 2.0, 3.0):
            for replicate in range(2):
                rows.append(
                    {
                        "row": f"g{cohort}-t{period}-r{replicate}",
                        "outcome": cohort if np.isfinite(cohort) else 0.0,
                        "time": period,
                        "treatment_time": cohort,
                        "x": float(replicate),
                    }
                )
    staggered = pd.DataFrame(rows).set_index("row")
    with pytest.raises(NotImplementedError, match="exactly one treated cohort"):
        _fit_robust(staggered)


class _NearBoundaryGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                0: np.full(len(X), 0.55),
                1: np.full(len(X), 0.20),
                2: np.full(len(X), 0.24999999),
                3: np.full(len(X), 1e-8),
            },
            index=X.index,
        )


class _NearBoundaryGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _NearBoundaryGeneralizedPropensityResult:
        del X, y
        return _NearBoundaryGeneralizedPropensityResult()


def test_generalized_propensity_overlap_refuses_without_clipping() -> None:
    with pytest.raises(ValueError, match="generalized propensities.*no clipping"):
        _fit_robust(
            _composition_hand_sample(),
            cross_fitter=_cross_fitter(
                propensity_factory=_NearBoundaryGeneralizedPropensity,
            ),
        )


class _AuditResult:
    def __init__(
        self,
        train_index: pd.Index,
        registry: list[tuple[pd.Index, pd.Index]],
    ) -> None:
        self.train_index = train_index.copy()
        self.registry = registry

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self.registry.append((self.train_index, X.index.copy()))
        return pd.DataFrame(
            np.full((len(X), 4), 0.25),
            index=X.index,
            columns=[0, 1, 2, 3],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.registry.append((self.train_index, X.index.copy()))
        return np.zeros(len(X))


class _AuditFactory:
    def __init__(self, registry: list[tuple[pd.Index, pd.Index]]) -> None:
        self.registry = registry

    def __call__(self):
        registry = self.registry

        class _AuditModel:
            def fit(self, X: pd.DataFrame, y: pd.Series) -> _AuditResult:
                del y
                return _AuditResult(X.index, registry)

        return _AuditModel()


def test_robust_cross_fitting_preserves_rows_and_indivisible_psus() -> None:
    data = _composition_hand_sample()
    data["psu"] = [f"psu-{position % 4}" for position in range(len(data))]
    registry: list[tuple[pd.Index, pd.Index]] = []
    factory = _AuditFactory(registry)
    result = RepeatedCrossSectionDiD(
        composition="robust",
        covariance="clustered",
    ).fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        cluster="psu",
        covariates=["x"],
        cross_fitter=_cross_fitter(
            propensity_factory=factory,
            outcome_factory=factory,
        ),
    )

    assert registry
    for train_index, holdout_index in registry:
        assert set(train_index).isdisjoint(holdout_index)
        assert set(data.loc[train_index, "psu"]).isdisjoint(data.loc[holdout_index, "psu"])
    assert result.nuisance_fold.groupby(data["psu"]).nunique().eq(1).all()


def test_outputhub_preserves_composition_target_nuisances_and_weights() -> None:
    outputhub = pytest.importorskip("universal_output_hub")
    result = _fit_robust(_composition_hand_sample())

    model = to_outputhub_model(result)
    hub = outputhub.OutputHub("Composition-robust repeated sections")
    add_to_outputhub(hub, result)

    assert model.metadata["composition"] == "robust"
    assert model.metadata["target_population"] == "treated_target_period"
    assert model.diagnostics["Nuisance fold fits"] == 8
    assert hub.tables[0].name == "Repeated-cross-section DiD nuisance fitting diagnostics"
    assert hub.tables[1].name == "Repeated-cross-section DiD composition weights"
