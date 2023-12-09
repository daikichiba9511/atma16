import pathlib
from typing import Any, Protocol


class XGBTrainCFG(Protocol):
    input_dir: pathlib.Path
    """入力データのディレクトリ""" ""
    output_dir: pathlib.Path
    """出力用のディレクトリ""" ""

    xgb_params: dict[str, Any]
    """xgbのパラメータ""" ""


def train_one_fold(cfg: XGBTrainCFG, fold: int) -> None:
    """1fold training

    1. split train/valid data by session_id
    2. create candidates
    3. train xgb model to rank candidates
    4. retrieve top-k candidates, where k is 10
    5. evaluate top-k candidates by map@10

    Args:
        cfg: XGBTrainCFG
        fold: fold number
    """
    pass
