import dataclasses
import pathlib

from src import constants


@dataclasses.dataclass
class Config:
    name: str = "exp000"
    """実験名"""
    description: str = """
    baseline.
    共起行列,人気ランキングから候補を生成してrerankして上位10件を予測にする
    """
    """実験の説明"""

    input_dir: pathlib.Path = constants.INPUT_DIR
    """入力データのディレクトリ"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / name
    """出力用のディレクトリ"""
