"""The workspace layout is a public development and packaging contract."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = [('storage_common','pipeline_storage'), ('01_data_collection','wp_tracker'),
           ('02_data_canonicalization','canonical_pipeline'), ('03_data_chunking','chunking_pipeline'),
           ('04_embedding','embedding'), ('05_retrieval','retrieval'), ('assistant_app','assistant_app'),
           ('orchestration','orchestration')]


@pytest.mark.parametrize('folder,package', MODULES)
def test_module_structure(folder,package):
    module = ROOT / folder
    assert (module/'pyproject.toml').is_file()
    assert (module/'README.md').is_file()
    assert (module/'src'/package/'__init__.py').is_file()
    assert (module/'tests').is_dir()


@pytest.mark.parametrize('folder,package', MODULES[1:7])
def test_checkout_config_and_wheel_defaults_exist(folder,package):
    filename = 'settings.yaml' if package=='wp_tracker' else 'config.yaml'
    assert (ROOT/folder/'config'/filename).is_file()
    assert (ROOT/folder/'src'/package/'config'/filename).is_file()
