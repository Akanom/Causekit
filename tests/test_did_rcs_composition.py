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


class _CompositionShiftGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X["x"].to_numpy(dtype=float)
        low_to_high = 0.125 + 0.25 * x
        high_to_low = 0.375 - 0.25 * x
        return pd.DataFrame(
            {
                0: high_to_low,
                1: low_to_high,
                2: high_to_low,
                3: low_to_high,
            },
            index=X.index,
        )


class _CompositionShiftGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _CompositionShiftGeneralizedPropensityResult:
        del X, y
        return _CompositionShiftGeneralizedPropensityResult()


class _CompositionShiftBinaryPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probability = np.full(len(X), 0.5)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=X.index)


class _CompositionShiftBinaryPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _CompositionShiftBinaryPropensityResult:
        del X, y
        return _CompositionShiftBinaryPropensityResult()


class _CompositionShiftOutcomeResult:
    def __init__(self, cell: str) -> None:
        self.cell = cell

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X["x"].to_numpy(dtype=float)
        if self.cell == "d0-s0":
            return 1.0 + x
        if self.cell == "d0-s1":
            return 2.0 + 3.0 * x
        if self.cell == "d1-s0":
            return 4.0 + 2.0 * x
        if self.cell == "d1-s1":
            return 7.0 + 8.0 * x
        raise AssertionError(f"Unexpected outcome cell {self.cell!r}.")


class _CompositionShiftOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _CompositionShiftOutcomeResult:
        del y
        cells = {"-".join(str(label).split("-")[:2]) for label in X.index}
        if len(cells) != 1:
            raise AssertionError("Each outcome nuisance must train within one group-period cell.")
        return _CompositionShiftOutcomeResult(cells.pop())


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


def _composition_shift_sample() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    cell_x_counts = {
        (0, 0): (6, 2),
        (0, 1): (2, 6),
        (1, 0): (6, 2),
        (1, 1): (2, 6),
    }
    for (treated, post), counts in cell_x_counts.items():
        x_values = [0.0] * counts[0] + [1.0] * counts[1]
        for replicate, x in enumerate(x_values):
            m00 = 1.0 + x
            m01 = 2.0 + 3.0 * x
            m10 = 4.0 + 2.0 * x
            treatment_effect = 2.0 + 4.0 * x
            untreated_treated_post = m10 + m01 - m00
            if treated == 0 and post == 0:
                outcome = m00
            elif treated == 0 and post == 1:
                outcome = m01
            elif treated == 1 and post == 0:
                outcome = m10
            else:
                outcome = untreated_treated_post + treatment_effect
            rows.append(
                {
                    "row": f"d{treated}-s{post}-x{int(x)}-r{replicate}",
                    "outcome": outcome,
                    "time": float(post + 1),
                    "treatment_time": 2.0 if treated else np.inf,
                    "x": x,
                    "true_effect": treatment_effect,
                }
            )
    return pd.DataFrame(rows).set_index("row")


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


def test_composition_shift_recovers_target_att_and_exposes_stationary_bias() -> None:
    data = _composition_shift_sample()
    robust = _fit_robust(
        data,
        cross_fitter=_cross_fitter(
            propensity_factory=_CompositionShiftGeneralizedPropensity,
            outcome_factory=_CompositionShiftOutcome,
        ),
    )
    stationary = RepeatedCrossSectionDiD(composition="stationary").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_cross_fitter(
            propensity_factory=_CompositionShiftBinaryPropensity,
            outcome_factory=_CompositionShiftOutcome,
        ),
    )

    treated = data["treatment_time"].eq(2.0)
    target = treated & data["time"].eq(2.0)
    baseline = treated & data["time"].eq(1.0)
    target_truth = float(data.loc[target, "true_effect"].mean())
    pooled_treated_truth = float(data.loc[treated, "true_effect"].mean())

    assert data.loc[target, "x"].mean() == pytest.approx(0.75)
    assert data.loc[baseline, "x"].mean() == pytest.approx(0.25)
    assert target_truth == pytest.approx(5.0)
    assert pooled_treated_truth == pytest.approx(4.0)
    assert robust.estimate == pytest.approx(target_truth, abs=1e-14)
    assert stationary.estimate == pytest.approx(pooled_treated_truth, abs=1e-14)
    assert stationary.estimate - target_truth == pytest.approx(-1.0, abs=1e-14)
    assert robust.target_population == "treated_target_period"
    assert stationary.target_population == "pooled_treated_stationary_composition"
    assert robust.overall_influence.mean() == pytest.approx(0.0, abs=1e-14)


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


class _ExtraClassGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1)),
            index=X.index,
            columns=[0, 1, 2, 3, 4],
        )


class _ExtraClassGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ExtraClassGeneralizedPropensityResult:
        del X, y
        return _ExtraClassGeneralizedPropensityResult()


class _MissingClassGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.40, 0.30, 0.30], (len(X), 1)),
            index=X.index,
            columns=[0, 1, 2],
        )


class _MissingClassGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _MissingClassGeneralizedPropensityResult:
        del X, y
        return _MissingClassGeneralizedPropensityResult()


class _DuplicateClassGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1)),
            index=X.index,
            columns=[0, 1, 2, 3, 3],
        )


class _DuplicateClassGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _DuplicateClassGeneralizedPropensityResult:
        del X, y
        return _DuplicateClassGeneralizedPropensityResult()


class _ExtraArrayClassGeneralizedPropensityResult:
    classes_ = np.array([0, 1, 2, 3, 4])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1))


class _ExtraArrayClassGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _ExtraArrayClassGeneralizedPropensityResult:
        del X, y
        return _ExtraArrayClassGeneralizedPropensityResult()


class _DuplicateArrayClassGeneralizedPropensityResult:
    classes_ = np.array([0, 1, 2, 3, 3])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (len(X), 1))


class _DuplicateArrayClassGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _DuplicateArrayClassGeneralizedPropensityResult:
        del X, y
        return _DuplicateArrayClassGeneralizedPropensityResult()


class _UnlabeledGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.40, 0.20, 0.25, 0.15], (len(X), 1))


class _UnlabeledGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _UnlabeledGeneralizedPropensityResult:
        del X, y
        return _UnlabeledGeneralizedPropensityResult()


class _PermutedGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = _KnownGeneralizedPropensityResult().predict_proba(X)
        return probabilities.loc[:, [3, 1, 0, 2]]


class _PermutedGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _PermutedGeneralizedPropensityResult:
        del X, y
        return _PermutedGeneralizedPropensityResult()


class _PermutedArrayGeneralizedPropensityResult:
    classes_ = np.array([3, 1, 0, 2])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = _KnownGeneralizedPropensityResult().predict_proba(X)
        return probabilities.loc[:, self.classes_].to_numpy(dtype=float)


class _PermutedArrayGeneralizedPropensity:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> _PermutedArrayGeneralizedPropensityResult:
        del X, y
        return _PermutedArrayGeneralizedPropensityResult()


@pytest.mark.parametrize(
    ("propensity_factory", "message"),
    [
        (_MissingClassGeneralizedPropensity, "missing class column"),
        (_ExtraClassGeneralizedPropensity, "exactly the observed classes"),
        (_DuplicateClassGeneralizedPropensity, "unique class columns"),
        (_ExtraArrayClassGeneralizedPropensity, "exactly the observed classes"),
        (_DuplicateArrayClassGeneralizedPropensity, "unique class labels"),
        (_UnlabeledGeneralizedPropensity, "requires fitted result.classes_"),
    ],
)
def test_generalized_propensity_refuses_invalid_or_ambiguous_class_schema(
    propensity_factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _fit_robust(
            _composition_hand_sample(),
            cross_fitter=_cross_fitter(propensity_factory=propensity_factory),
        )


@pytest.mark.parametrize(
    "propensity_factory",
    [_PermutedGeneralizedPropensity, _PermutedArrayGeneralizedPropensity],
)
def test_labeled_generalized_propensity_order_is_realigned_without_changing_result(
    propensity_factory: Callable[[], object],
) -> None:
    data = _composition_hand_sample()
    expected = _fit_robust(data)
    permuted = _fit_robust(
        data,
        cross_fitter=_cross_fitter(propensity_factory=propensity_factory),
    )

    assert permuted.estimate == pytest.approx(expected.estimate, abs=1e-14)
    pd.testing.assert_frame_equal(
        permuted.composition_weights,
        expected.composition_weights,
        atol=1e-14,
    )
    pd.testing.assert_series_equal(
        permuted.overall_influence,
        expected.overall_influence,
        atol=1e-14,
    )


def test_composition_robust_result_is_invariant_to_row_permutation() -> None:
    data = _composition_hand_sample()
    permuted_data = data.sample(frac=1.0, random_state=2_026_073)
    expected = _fit_robust(data)
    permuted = _fit_robust(permuted_data)

    assert permuted.estimate == pytest.approx(expected.estimate, abs=1e-14)
    assert permuted.standard_error == pytest.approx(expected.standard_error, abs=1e-14)
    assert permuted.design_fingerprint == expected.design_fingerprint
    pd.testing.assert_frame_equal(permuted.group_time, expected.group_time, atol=1e-14)
    pd.testing.assert_frame_equal(
        permuted.composition_weights.sort_index(),
        expected.composition_weights.sort_index(),
        atol=1e-14,
    )
    pd.testing.assert_series_equal(
        permuted.overall_influence.sort_index(),
        expected.overall_influence.sort_index(),
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(
        permuted.nuisance_predictions.sort_index(),
        expected.nuisance_predictions.sort_index(),
        atol=1e-14,
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
