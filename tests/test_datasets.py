"""Tests for opt-in, hash-verified real-data loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from causekit.datasets import (
    REAL_DATASETS,
    RealDataset,
    default_real_data_directory,
    verified_real_data_path,
)


def test_real_dataset_registry_is_https_and_sha256_pinned() -> None:
    assert set(REAL_DATASETS) == {
        "cattaneo2",
        "hospdd",
        "hsng",
        "nsw_mixtape",
        "wage_panel",
        "yrbs_beverage_tax",
    }
    for name, specification in REAL_DATASETS.items():
        assert name == specification.name
        assert specification.url.startswith("https://")
        assert specification.provenance_url.startswith("https://")
        assert len(specification.sha256) == 64
        int(specification.sha256, 16)


def test_verified_path_accepts_only_the_registered_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"fixed non-sensitive fixture"
    specification = RealDataset(
        name="fixture",
        filename="fixture.dta",
        url="https://example.test/fixture.dta",
        sha256=hashlib.sha256(content).hexdigest(),
        description="Test fixture.",
        provenance_url="https://example.test/provenance",
    )
    monkeypatch.setitem(REAL_DATASETS, "fixture", specification)
    path = tmp_path / specification.filename
    path.write_bytes(content)

    assert verified_real_data_path("fixture", data_directory=tmp_path) == path
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Refusing unverified dataset"):
        verified_real_data_path("fixture", data_directory=tmp_path)


def test_missing_data_requires_explicit_download_authorization(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="explicit download authorization"):
        verified_real_data_path("hsng", data_directory=tmp_path)


def test_default_cache_uses_local_application_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_real_data_directory() == tmp_path / "causekit" / "datasets"


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown real dataset"):
        verified_real_data_path("unknown")
