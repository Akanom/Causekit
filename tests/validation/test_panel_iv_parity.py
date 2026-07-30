"""Independent fixed-effects Panel IV parity against linearmodels IV2SLS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causekit import PanelIV2SLS

linearmodels = pytest.importorskip("linearmodels.iv")

pytestmark = pytest.mark.validation


def _panel(*, seed: int = 717, unbalanced: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for entity in range(24):
        entity_effect = rng.normal()
        for period in range(5):
            instrument = rng.normal()
            control = rng.normal()
            first_error = rng.normal(scale=0.7)
            endogenous = 0.8 * instrument + 0.3 * control + 0.4 * entity_effect + first_error
            outcome = (
                1.7 * endogenous
                - 0.6 * control
                + entity_effect
                + 0.2 * period
                + 0.55 * first_error
                + rng.normal(scale=0.4)
            )
            rows.append(
                {
                    "entity": entity,
                    "time": period,
                    "outcome": outcome,
                    "endogenous": endogenous,
                    "instrument": instrument,
                    "control": control,
                }
            )
    data = pd.DataFrame(rows)
    if unbalanced:
        data = data.loc[((3 * data["entity"] + data["time"]) % 13) != 0].copy()
    return data


def _dummy_reference(data: pd.DataFrame, *, covariance: str, time_effects: bool):
    entity_dummies = pd.get_dummies(
        data["entity"].astype("category"), prefix="entity", drop_first=True, dtype=float
    )
    pieces = [
        pd.Series(1.0, index=data.index, name="const"),
        data["control"],
        entity_dummies,
    ]
    if time_effects:
        pieces.append(
            pd.get_dummies(
                data["time"].astype("category"), prefix="time", drop_first=True, dtype=float
            )
        )
    exogenous = pd.concat(pieces, axis=1)
    kwargs: dict[str, object] = {"cov_type": covariance, "debiased": True}
    if covariance == "clustered":
        kwargs["clusters"] = data["entity"]
    return linearmodels.IV2SLS(
        data["outcome"], exogenous, data[["endogenous"]], data[["instrument"]]
    ).fit(**kwargs)


@pytest.mark.parametrize("covariance", ["unadjusted", "robust", "clustered"])
@pytest.mark.parametrize("time_effects", [False, True])
@pytest.mark.parametrize("unbalanced", [False, True])
def test_compact_within_panel_iv_matches_explicit_dummy_2sls(
    covariance: str, time_effects: bool, unbalanced: bool
) -> None:
    data = _panel(unbalanced=unbalanced)
    native = PanelIV2SLS(covariance=covariance, time_effects=time_effects).fit(
        data,
        outcome="outcome",
        endogenous="endogenous",
        instruments="instrument",
        exogenous="control",
        entity="entity",
        time="time",
    )
    reference = _dummy_reference(data, covariance=covariance, time_effects=time_effects)
    terms = ["control", "endogenous"]

    np.testing.assert_allclose(
        native.params.loc[terms], reference.params.loc[terms], rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        native.covariance.loc[terms, terms],
        reference.cov.loc[terms, terms],
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        native.residuals.to_numpy(), reference.resids.to_numpy(), rtol=1e-10, atol=1e-11
    )
    assert native.df_resid == reference.df_resid
    assert native.nobs == reference.nobs
