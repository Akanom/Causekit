"""Materialize verified real-data CSVs shared by Python, R, and Stata parity.

Raw source files remain in the external CauseKit cache. The generated CSV directory is
also external by default and contains a manifest with source and output SHA-256 digests.

Usage from the repository root::

    python benchmarks/prepare_real_data.py --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from causekit.datasets import (
    REAL_DATASETS,
    default_real_data_directory,
    load_real_dataset,
    verified_real_data_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_parity_data_directory() -> Path:
    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return local_root / "causekit" / "parity" / "real_data_v1"


def _hsng(data: pd.DataFrame) -> pd.DataFrame:
    return data[["rent", "pcturban", "hsngval", "faminc", "region"]].copy()


def _nsw(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        ["treat", "age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75", "re78"]
    ].copy()


def _cattaneo(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[["bweight", "mbsmoke", "mmarried", "mage", "medu", "fbaby"]].copy()
    frame.insert(0, "source_row", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def _hospdd(data: pd.DataFrame) -> pd.DataFrame:
    panel = (
        data.groupby(["hospital", "month"], as_index=False)
        .agg(outcome=("satis", "mean"), treated=("procedure", "max"))
        .sort_values(["hospital", "month"], kind="stable")
        .reset_index(drop=True)
    )
    first_treated = panel.loc[panel["treated"].eq(1)].groupby("hospital", sort=False)["month"].min()
    panel["treatment_time"] = panel["hospital"].map(first_treated).fillna(0).astype(int)
    return panel[["hospital", "month", "outcome", "treated", "treatment_time"]]


_TRANSFORMS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "hsng": ("hsng.csv", _hsng),
    "nsw_mixtape": ("nsw_mixtape.csv", _nsw),
    "cattaneo2": ("cattaneo2.csv", _cattaneo),
    "hospdd": ("hospdd_panel.csv", _hospdd),
}


def prepare_real_data(
    *,
    source_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    download: bool = False,
) -> Path:
    """Write deterministic parity CSVs and return the generated manifest path."""

    source_root = (
        default_real_data_directory() if source_directory is None else Path(source_directory)
    )
    output_root = (
        default_parity_data_directory() if output_directory is None else Path(output_directory)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "causekit_real_data_parity_v1",
        "float_format": "%.17g",
        "datasets": {},
    }
    entries: dict[str, object] = {}
    for name, (filename, transform) in _TRANSFORMS.items():
        source_path = verified_real_data_path(
            name,
            data_directory=source_root,
            download=download,
        )
        source = load_real_dataset(name, data_directory=source_root)
        frame = transform(source)
        destination = output_root / filename
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        specification = REAL_DATASETS[name]
        entries[name] = {
            "source_filename": specification.filename,
            "source_url": specification.url,
            "source_sha256": specification.sha256,
            "source_verified_sha256": _sha256(source_path),
            "csv_filename": filename,
            "csv_sha256": _sha256(destination),
            "rows": len(frame),
            "columns": list(frame.columns),
        }
    manifest["datasets"] = entries
    manifest_path = output_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.part")
    temporary_manifest.unlink(missing_ok=True)
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = prepare_real_data(
        source_directory=args.source_directory,
        output_directory=args.output_directory,
        download=args.download,
    )
    print(f"manifest={manifest.resolve()}")


if __name__ == "__main__":
    main()
