from baler.modules.config.config_service import load_config


def test_read_config():
    config = load_config("config.yaml")
    assert config is not None
