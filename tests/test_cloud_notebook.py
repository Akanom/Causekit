from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle" / "causekit_quickstart.ipynb"
KAGGLE_METADATA = ROOT / "notebooks" / "kaggle" / "kernel-metadata.json"
EXPECTED_VERSION = "0.7.0a6"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(index: int) -> str:
    return "".join(_notebook()["cells"][index]["source"])


def test_cloud_notebook_is_clean_valid_and_portable() -> None:
    notebook = _notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert "colab.research.google.com/github/Akanom/Causekit" in _source(0)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_install_cell_is_bounded_private_release_safe_and_evicts_stale_modules() -> None:
    install = _source(2)

    assert EXPECTED_VERSION in install
    assert "WHEEL_NAME" in install
    assert "causekit[validation,plot,outputhub]" in install
    assert 'Path("/kaggle/input")' in install
    assert "from google.colab import files" in install
    assert "files.upload()" in install
    assert "timeout=180" in install
    assert "git+https://" not in install
    assert "github" not in install.lower()
    assert "token" not in install.lower()
    assert "tuple(sys.modules)" in install
    assert 'module_name == "causekit"' in install
    assert 'module_name.startswith("causekit.")' in install
    assert "sys.modules.pop(module_name, None)" in install
    assert "importlib.invalidate_caches()" in install


def test_notebook_uses_verified_real_data_and_honest_ml_contracts() -> None:
    source = "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])

    assert 'load_real_dataset("nsw_mixtape", download=True)' in source
    assert 'load_real_dataset("cattaneo2", download=True)' in source
    assert 'load_real_dataset("hospdd", download=True)' in source
    assert "REAL_DATASETS[name].sha256" in source
    assert "PartiallyLinearDML" in source
    assert "RLearner" in source
    assert "construction_index.intersection(rlearner.evaluation_index).empty" in source
    assert "construction_nobs + rlearner.evaluation_nobs" in source
    assert 'NearestNeighborMatch(estimand="att", inference="none")' in source
    assert "diagnostics—not proofs of identification" in source


def test_kaggle_metadata_points_to_shared_colab_notebook_without_secrets() -> None:
    metadata = json.loads(KAGGLE_METADATA.read_text(encoding="utf-8"))

    assert metadata == {
        "id": "akanom/causekit-quickstart",
        "title": "CauseKit quickstart",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    combined = NOTEBOOK.read_text(encoding="utf-8") + KAGGLE_METADATA.read_text(encoding="utf-8")
    assert "kaggle.json" not in combined
    assert "KAGGLE_KEY" not in combined
    assert "KAGGLE_USERNAME" not in combined
