"""Hash-verified real-data fixed-effects Panel IV example."""

from __future__ import annotations

import argparse

from causekit import PanelIV2SLS
from causekit.datasets import load_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly authorize the pinned HTTPS wage-panel download.",
    )
    arguments = parser.parse_args()
    data = load_real_dataset("wage_panel", download=arguments.download).sort_values(["nr", "year"])
    data = data.copy()
    data["union_lag"] = data.groupby("nr", sort=False)["union"].shift()
    data["hours_1000"] = data["hours"] / 1_000.0

    result = PanelIV2SLS(missing="drop").fit(
        data,
        outcome="lwage",
        endogenous="union",
        instruments="union_lag",
        exogenous=["hours_1000", "married"],
        entity="nr",
        time="year",
    )
    print(result.to_markdown())
    print("\nPanel design")
    print(result.panel_summary().to_string())
    print("\nInstrument variation")
    print(result.instrument_variation.to_string())
    print("\nFirst stage")
    print(result.first_stage["union"].to_dict())
    print(
        "\nInterpretation boundary: this reproduces a numerical Panel IV specification. "
        "A one-period union lag is not automatically exogenous or excluded, so the output "
        "does not establish a causal union-wage effect."
    )


if __name__ == "__main__":
    main()
