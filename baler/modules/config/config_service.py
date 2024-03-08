from typing import Any
import yaml


def load_config(file_path) -> dict[str, Any]:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file).get("config")
    return config
