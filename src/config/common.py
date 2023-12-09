import importlib
from typing import TypeAlias

from src.config import exp000

Configs: TypeAlias = exp000.Config

def load_config(config_name: str) -> Configs:
    """config_nameに対応するConfigクラスを返す"""
    config = importlib.import_module(f"src.config.{config_name}").Config()
    return config
