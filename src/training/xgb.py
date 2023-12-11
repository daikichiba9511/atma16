import pathlib
import pprint
from logging import getLogger
from typing import Any, Protocol

import matplotlib.pyplot as plt
import polars as pl
import xgboost as xgb

from src import constants
from src.preprocess.dataset import negative_sampling
from src.utils import common as utils_common

logger = getLogger(__name__)


class XGBTrainCFG(Protocol):
    input_dir: pathlib.Path
    """入力データのディレクトリ""" ""
    output_dir: pathlib.Path
    """出力用のディレクトリ""" ""

    xgb_params: dict[str, Any]
    """xgbのパラメータ""" ""


def train_one_fold(cfg: XGBTrainCFG, fold: int, train_df: pl.DataFrame, valid_df: pl.DataFrame, negative_sampling_rate: float = 0.05) -> None:
    """1fold training, and then save model

    1. split train/valid data by session_id
    2. create candidates
    3. train xgb model to rank candidates
    4. retrieve top-k candidates, where k is 10
    5. evaluate top-k candidates by map@10

    Args:
        cfg: XGBTrainCFG
        fold: fold number
        train_df: train dataframe
        valid_df: valid dataframe

    """
    logger.info(f"Training start fold={fold}")

    # negative sampling
    # クラス不均衡なのでnegative samplingを行う
    logger.info(f"Before negative_sampling: {train_df.shape}")
    train_df = negative_sampling(train_df, sampling_rate=negative_sampling_rate)
    logger.info(f"After negative_sampling: {train_df.shape}")

    logger.info(f"train_df shape: {train_df.shape}, valid_df shape: {valid_df.shape}")
    logger.info(f"train_label_cnts: {train_df['target'].value_counts()}")
    logger.info(f"valid_label_cnts: {valid_df['target'].value_counts()}")

    train_df_ = train_df.drop(constants.NOT_USED_COLUMNS)
    valid_df_ = valid_df.drop(constants.NOT_USED_COLUMNS)

    logger.info(f"Used Columns: \n{pprint.pformat(train_df_.columns)}")

    dtrain = xgb.DMatrix(train_df_, label=train_df["target"], feature_names=train_df_.columns)
    dvalid = xgb.DMatrix(valid_df_, label=valid_df["target"], feature_names=valid_df_.columns)

    num_boost_round = 10000
    with utils_common.trace("training..."):
        model = xgb.train(
            params=cfg.xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

    fig, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    fig.savefig(str(cfg.output_dir / f"importance_fold{fold}.png"))

    model.save_model(str(cfg.output_dir / f"xgb_model_fold{fold}.ubj"))
