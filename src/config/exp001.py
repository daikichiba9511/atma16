import dataclasses
import pathlib

from src import constants


@dataclasses.dataclass
class Config:
    name: str = "exp001"
    """実験名"""
    description: str = """
    original: exp000

    共起行列,人気ランキングから候補を生成してrerankして上位10件を予測にする

    特徴量追加
        * diff_raw_mean_total_room_cnt_with_onsen_flg
        * diff_raw_median_total_room_cnt_with_onsen_flg
        * diff_raw_mean_total_room_cnt_with_yad_type
        * diff_raw_median_total_room_cnt_with_yad_type
        * diff_raw_mean_total_room_cnt_with_kd_stn_5min
        * diff_raw_median_total_room_cnt_with_kd_stn_5min
        * diff_raw_mean_total_room_cnt_with_kd_bch_5min
        * diff_raw_median_total_room_cnt_with_kd_bch_5min
        * diff_raw_mean_total_room_cnt_with_kd_conv_walk_5min
        * diff_raw_median_total_room_cnt_with_kd_conv_walk_5min
    """
    """実験の説明"""

    seed: int = 42

    input_dir: pathlib.Path = constants.INPUT_DIR
    """入力データのディレクトリ"""
    output_dir: pathlib.Path = constants.OUTPUT_DIR / name
    """出力用のディレクトリ"""

    n_splits: int = 5
    """fold数."""

    negative_sampling_rate: float = 0.5

    xgb_params: dict[str, str | int | float] = dataclasses.field(
        default_factory=lambda: {
            # "objective": "rank:pairwise",
            "objective": "binary:logistic",
            "tree_method": "hist",
            "random_state": 42,
            "max_depth": 8,
            "learning_rate": 0.1,
            "verbosity": 1,
            "device": "cuda",  # gpuでの学習に必要
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )
