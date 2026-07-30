"""Sharp and fuzzy regression-discontinuity workflows with native bandwidth selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causekit import RegressionDiscontinuity


def main() -> None:
    rng = np.random.default_rng(20_260_730)
    nobs = 1_500
    index = pd.Index(np.arange(50_000, 50_000 + nobs), name="unit")
    score = pd.Series(rng.uniform(-2.5, 2.5, nobs), index=index, name="eligibility_score")

    sharp_outcome = pd.Series(
        0.6
        + 0.5 * score
        + 0.20 * score**2
        + 1.75 * (score >= 0.0)
        + rng.normal(scale=0.65, size=nobs),
        index=index,
        name="outcome",
    )
    sharp = RegressionDiscontinuity(
        design="sharp",
        cutoff=0.0,
        bandwidth="native_mse",
        bandwidth_candidates=11,
    ).fit(sharp_outcome, running=score)

    fuzzy_rng = np.random.default_rng(20_260_731)
    fuzzy_score = pd.Series(
        fuzzy_rng.uniform(-2.5, 2.5, nobs), index=index, name="eligibility_score"
    )
    take_up_probability = 1.0 / (
        1.0 + np.exp(-(-1.0 + 2.5 * (fuzzy_score >= 0.0) + 0.15 * fuzzy_score))
    )
    take_up = pd.Series(
        fuzzy_rng.binomial(1, take_up_probability),
        index=index,
        name="program_take_up",
    )
    fuzzy_outcome = pd.Series(
        0.4
        + 0.45 * fuzzy_score
        + 0.15 * fuzzy_score**2
        + 2.25 * take_up
        + fuzzy_rng.normal(scale=0.5, size=nobs),
        index=index,
        name="outcome",
    )
    fuzzy = RegressionDiscontinuity(
        design="fuzzy",
        cutoff=0.0,
        bandwidth="native_mse",
        bandwidth_candidates=11,
    ).fit(fuzzy_outcome, running=fuzzy_score, treatment=take_up)

    print("Sharp cutoff effect")
    print(sharp.summary_frame().to_string())
    print(
        "bandwidths=",
        (sharp.bandwidth_left, sharp.bandwidth_right),
        "density_p=",
        sharp.manipulation.pvalue,
    )
    print()
    print("Fuzzy cutoff-complier effect")
    print(fuzzy.summary_frame().to_string())
    print("corrected_first_stage=", fuzzy.treatment_jump)
    print("weak_first_stage_warning=", fuzzy.first_stage["weak_first_stage_warning"])
    print()
    print(
        "Interpretation requires smooth potential outcomes, no precise sorting, no other "
        "discontinuous intervention, and - for fuzzy RD - exclusion and monotonicity."
    )

    # Optional diagnostic graph:
    # axes = sharp.plot(bins=20)
    # axes.figure.show()


if __name__ == "__main__":
    main()
