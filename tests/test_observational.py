"""Tests for supplied-nuisance IPW and AIPW estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from causekit import AIPWATE, IPWATE, ObservationalATEResult


def _sample(nobs: int = 3000, seed: int = 20260729):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(nobs, 2))
    propensity = expit(-0.15 + x @ np.array([0.55, -0.4]))
    treatment = rng.binomial(1, propensity)
    mu0 = 0.5 + x @ np.array([0.9, -0.6])
    mu1 = mu0 + 1.4 + 0.2 * x[:, 0]
    outcome = np.where(treatment == 1, mu1, mu0) + rng.normal(size=nobs)
    index = pd.Index([f"person-{i}" for i in range(nobs)])
    return tuple(
        pd.Series(value, index=index) for value in (outcome, treatment, propensity, mu1, mu0)
    )


def test_ipw_matches_direct_horvitz_thompson_identity() -> None:
    y, treatment, propensity, _, _ = _sample(nobs=600)
    result = IPWATE().fit(y, treatment=treatment, propensity=propensity)
    expected_scores = treatment * y / propensity - (1 - treatment) * y / (1 - propensity)
    assert isinstance(result, ObservationalATEResult)
    assert result.estimate == pytest.approx(expected_scores.mean(), abs=1e-14)
    assert result.influence_function.mean() == pytest.approx(0, abs=1e-14)


def test_aipw_matches_efficient_score_identity() -> None:
    y, treatment, propensity, mu1, mu0 = _sample(nobs=700)
    result = AIPWATE().fit(
        y,
        treatment=treatment,
        propensity=propensity,
        outcome_treated=mu1,
        outcome_control=mu0,
    )
    scores = (
        mu1
        - mu0
        + treatment * (y - mu1) / propensity
        - (1 - treatment) * (y - mu0) / (1 - propensity)
    )
    assert result.estimate == pytest.approx(scores.mean(), abs=1e-14)


@pytest.mark.parametrize("estimand", ["att", "atc"])
def test_ipw_target_population_estimands_match_normalized_weight_identities(
    estimand: str,
) -> None:
    y, treatment, propensity, _, _ = _sample(nobs=900)
    result = IPWATE(estimand=estimand).fit(y, treatment=treatment, propensity=propensity)
    if estimand == "att":
        comparison_weights = (1 - treatment) * propensity / (1 - propensity)
        expected = y[treatment == 1].mean() - np.average(y, weights=comparison_weights)
    else:
        comparison_weights = treatment * (1 - propensity) / propensity
        expected = np.average(y, weights=comparison_weights) - y[treatment == 0].mean()
    assert result.estimate == pytest.approx(expected, abs=1e-14)
    assert result.params.index.tolist() == [estimand]
    assert result.influence_function.mean() == pytest.approx(0, abs=1e-14)


@pytest.mark.simulation
@pytest.mark.parametrize("estimand", ["att", "atc"])
def test_oracle_aipw_recovers_target_population_effect(estimand: str) -> None:
    y, treatment, propensity, mu1, mu0 = _sample(nobs=16000, seed=704)
    result = AIPWATE(estimand=estimand).fit(
        y,
        treatment=treatment,
        propensity=propensity,
        outcome_treated=mu1,
        outcome_control=mu0,
    )
    target = (mu1 - mu0)[treatment == (1 if estimand == "att" else 0)].mean()
    assert result.estimate == pytest.approx(target, abs=0.04)
    assert result.estimand == estimand


@pytest.mark.simulation
def test_oracle_aipw_recovers_population_ate_and_is_more_precise() -> None:
    y, treatment, propensity, mu1, mu0 = _sample(nobs=12000)
    ipw = IPWATE().fit(y, treatment=treatment, propensity=propensity)
    aipw = AIPWATE().fit(
        y,
        treatment=treatment,
        propensity=propensity,
        outcome_treated=mu1,
        outcome_control=mu0,
    )
    assert aipw.estimate == pytest.approx(1.4, abs=0.04)
    assert aipw.standard_error < ipw.standard_error


def test_clustered_influence_aggregation_matches_manual_formula() -> None:
    y, treatment, propensity, mu1, mu0 = _sample(nobs=600)
    clusters = pd.Series(np.repeat(np.arange(60), 10), index=y.index)
    result = AIPWATE(covariance="clustered").fit(
        y,
        treatment=treatment,
        propensity=propensity,
        outcome_treated=mu1,
        outcome_control=mu0,
        clusters=clusters,
    )
    grouped = result.influence_function.groupby(clusters).sum().to_numpy()
    expected_variance = 60 / 59 * (grouped @ grouped) / len(y) ** 2
    assert result.standard_error**2 == pytest.approx(expected_variance, rel=1e-12)
    assert result.n_clusters == 60
    assert result.inference_df == 59


def test_clipping_is_explicit_and_diagnosed() -> None:
    y, treatment, propensity, _, _ = _sample(nobs=300)
    propensity.iloc[0] = 0.001
    propensity.iloc[1] = 0.999
    result = IPWATE(clip=(0.01, 0.99)).fit(y, treatment=treatment, propensity=propensity)
    assert result.clipped_observations == 2
    assert result.overlap.clipped_observations == 2
    assert result.notes and "sensitivity" in result.notes[0]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
def test_invalid_propensity_values_are_rejected(bad: float) -> None:
    y, treatment, propensity, _, _ = _sample(nobs=100)
    propensity.iloc[0] = bad
    with pytest.raises(ValueError, match="strictly between"):
        IPWATE().fit(y, treatment=treatment, propensity=propensity)


def test_binary_treatment_alignment_and_cluster_contract_are_strict() -> None:
    y, treatment, propensity, _, _ = _sample(nobs=100)
    with pytest.raises(ValueError, match="coded exactly"):
        IPWATE().fit(y, treatment=treatment.replace({1: 2}), propensity=propensity)
    with pytest.raises(ValueError, match="indices must match"):
        IPWATE().fit(y.reset_index(drop=True), treatment=treatment, propensity=propensity)
    with pytest.raises(ValueError, match="clusters are required"):
        IPWATE(covariance="clustered").fit(y, treatment=treatment, propensity=propensity)


def test_large_vectorized_fit_smoke() -> None:
    y, treatment, propensity, mu1, mu0 = _sample(nobs=100_000, seed=13)
    result = AIPWATE().fit(
        y,
        treatment=treatment,
        propensity=propensity,
        outcome_treated=mu1,
        outcome_control=mu0,
    )
    assert result.nobs == 100_000
    assert np.isfinite(result.estimate)
