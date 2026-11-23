from pathlib import Path
from pipeline_storage import default_config, configuration_root, workspace_root


def test_external_checkout_config_precedes_packaged_defaults(tmp_path):
    module = tmp_path/'embedding'
    file = module/'src/embedding/config.py'
    file.parent.mkdir(parents=True)
    file.touch()
    (module/'pyproject.toml').touch()
    external = module/'config/config.yaml'
    external.parent.mkdir()
    external.write_text('external')
    bundled = file.parent/'config/config.yaml'
    bundled.parent.mkdir()
    bundled.write_text('bundled')
    assert default_config(str(file),'config/config.yaml') == external
    assert configuration_root(external,str(file),'embedding') == module


def test_wheel_defaults_target_workspace_instead_of_site_packages(tmp_path,monkeypatch):
    file = tmp_path/'site-packages/04_embedding/config.py'
    config = file.parent/'config/config.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('default')
    file.touch()
    workspace = tmp_path/'workspace'
    monkeypatch.setenv('CHATBOT_PROJECT_ROOT',str(workspace))
    assert default_config(str(file),'config/config.yaml') == config
    assert workspace_root(str(file)) == workspace
    assert configuration_root(config,str(file),'embedding') == workspace/'embedding'
