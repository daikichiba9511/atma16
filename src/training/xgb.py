import pathlib
from logging import getLogger
from typing import Any, Protocol

import polars as pl
import xgboost as xgb

logger = getLogger(__name__)


class XGBTrainCFG(Protocol):
    input_dir: pathlib.Path
    """入力データのディレクトリ""" ""
    output_dir: pathlib.Path
    """出力用のディレクトリ""" ""

    xgb_params: dict[str, Any]
    """xgbのパラメータ""" ""


def train_one_fold(cfg: XGBTrainCFG, fold: int, train_df: pl.DataFrame, valid_df: pl.DataFrame) -> None:
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
    logger.info(f"Training start fold={fold}")
    logger.info(f"train_df shape: {train_df.shape}, valid_df shape: {valid_df.shape}")

    dtrain = xgb.DMatrix(train_df.drop(["target"]), label=train_df["target"])
    dvalid = xgb.DMatrix(valid_df.drop(["target"]), label=valid_df["target"])

    num_boost_round = 1000
    model = xgb.train(
        params=cfg.xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        varbose_eval=num_boost_round // 50,
    )
    model.save_model(str(cfg.output_dir / f"model_fold{fold}.xgb"))
