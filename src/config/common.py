import importlib
from typing import Any


def load_config(config_name: str) -> Any:
    """config_nameに対応するConfigクラスを返す"""
    config = importlib.import_module(f"src.config.{config_name}").Config()
    return config
