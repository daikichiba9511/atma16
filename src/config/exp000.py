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

    seed: int = 42

    input_dir: pathlib.Path = constants.INPUT_DIR
    """入力データのディレクトリ"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / name
    """出力用のディレクトリ"""

    n_splits: int = 5
    """fold数."""

    xgb_params: dict[str, str | int | float] = dataclasses.field(
        default_factory=lambda: {
            "objective": "rank:pairwise",
            "tree_method": "gpu_hist",
            "random_state": 42,
            "gpu_id": 0,
            "learning_rate": 0.05,
            "verbosity": 1,
        }
    )