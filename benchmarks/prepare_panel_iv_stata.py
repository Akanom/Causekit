"""Prepare the hash-verified wage-panel fixture for manual Stata parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from causekit.datasets import REAL_DATASETS, load_real_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_directory() -> Path:
    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return local_root / "causekit" / "parity"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=None)
    arguments = parser.parse_args()

    data = load_real_dataset("wage_panel", download=arguments.download).sort_values(["nr", "year"])
    data = data.copy()
    data["union_lag"] = data.groupby("nr", sort=False)["union"].shift()
    data["hours_1000"] = data["hours"] / 1_000.0
    roles = ["nr", "year", "lwage", "hours_1000", "married", "union", "union_lag"]
    fixture = data.loc[:, roles].dropna().copy()
    fixture["nr"] = fixture["nr"].astype(int)
    fixture["year"] = fixture["year"].astype(int)
    output_directory = arguments.output_directory or _default_directory()
    output_directory.mkdir(parents=True, exist_ok=True)
    fixture_path = output_directory / "panel_iv_wage_v1.csv"
    metadata_path = output_directory / "panel_iv_wage_v1.json"
    fixture.to_csv(fixture_path, index=False, lineterminator="\n", float_format="%.17g")
    metadata = {
        "contract": "wage_panel_two_way_fe_entity_clustered_cr1",
        "source_url": REAL_DATASETS["wage_panel"].url,
        "source_sha256": REAL_DATASETS["wage_panel"].sha256,
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": _sha256(fixture_path),
        "rows": int(len(fixture)),
        "entities": int(fixture["nr"].nunique()),
        "periods": int(fixture["year"].nunique()),
        "generator": "benchmarks/prepare_panel_iv_stata.py",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"fixture={fixture_path.resolve()}")
    print(f"metadata={metadata_path.resolve()}")
    print(f"fixture_sha256={metadata['fixture_sha256']}")
    print('stata_command=do "benchmarks/validate_panel_iv_stata.do"')


if __name__ == "__main__":
    main()
