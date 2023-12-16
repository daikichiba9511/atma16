import dataclasses
import pathlib
import pprint
from logging import INFO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from matplotlib import axes, figure
from sklearn import model_selection
from tqdm.auto import tqdm

from src import constants
from src.training.common import load_dataframes
from src.utils.logger import attach_file_handler, get_root_logger

LOGGER = get_root_logger(INFO)


@dataclasses.dataclass
class Config:
    name = "exp005"
    seed: int = 42
    n_splits: int = 5

    output_dir: pathlib.Path = constants.OUTPUT_DIR / name
    category_features: list[str] = dataclasses.field(
        default_factory=lambda: [
            # "yad_no",
            "latest_yad_no",
            "wid_cd",
            "ken_cd",
            "lrg_cd",
            "sml_cd",
            "session_interested_in_sml_cd_0",
            "session_interested_in_lrg_cd_0",
            "session_interested_in_ken_cd_0",
            "session_interested_in_wid_cd_0",
            "latest_ken_cd",
            "latest_lrg_cd",
            "latest_sml_cd",
        ]
    )

    train_candidates_path: pathlib.Path = constants.OUTPUT_DIR / "train_candidates.parquet"
    test_candidates_path: pathlib.Path = constants.OUTPUT_DIR / "test_candidates.parquet"


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k for a single actual value.

    Parameters:
    actual : int
        The actual value that is to be predicted
    predicted : list
        A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns:
    float
        The average precision at k
    """
    if actual in predicted[:k]:
        return 1.0 / (predicted[:k].index(actual) + 1)
    return 0.0


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k for lists of actual values and predicted values.

    Parameters:
    actual : list
        A list of actual values that are to be predicted
    predicted : list
        A list of lists of predicted elements (order does matter in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns:
    float
        The mean average precision at k
    """
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)


def create_top_10_yad_predict(_df: pd.DataFrame) -> pd.DataFrame:
    # セッションごとに予測確率の高い順に yad_no の配列を作成
    _agg = _df.sort_values("predict", ascending=False).groupby("session_id")["yad_no"].apply(list)

    out_df = pd.DataFrame(index=_agg.index, data=_agg.values.tolist()).iloc[:, :10]

    return out_df


def _describe(df: pl.DataFrame, phase: str):
    LOGGER.info(f"{phase} shape: {df.shape}")
    LOGGER.info(f"Use Columns in {phase}: \n{pprint.pformat(df.columns)}")
    LOGGER.info(f"Num of each session_id {phase}: \n{df.group_by('session_id').count()}")



def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    attach_file_handler(LOGGER, str(cfg.output_dir / f"train_{cfg.name}.log"), INFO)

    train = pl.read_parquet(cfg.train_candidates_path)
    test = pl.read_parquet(cfg.test_candidates_path)

    _describe(train, "train")
    _describe(test, "test")

    use_session_ids = (
        train.group_by("session_id").agg(pl.col("target").sum()).filter(pl.col("target") == 1)["session_id"].to_list()
    )

    xgb_params = {
        # "objective": "rank:pairwise",
        "objective": "binary:logistic",
        "tree_method": "hist",
        "random_state": 42,
        "max_depth": 5,
        "learning_rate": 0.1,
        "verbosity": 1,
        "device": "cuda",  # gpuでの学習に必要
        "subsample": 0.95,
        "colsample_bytree": 0.95,
        "eval_metric": "auc",
        # "scale_pos_weight": 30,
    }
    LOGGER.info(f"xgb_params: \n{pprint.pformat(xgb_params)}")
    xgb_model_list = []
    for fold in range(cfg.n_splits):
        LOGGER.info(f"fold: {fold}")
        x_train = train.filter((pl.col("session_id").is_in(use_session_ids)) & (pl.col("fold") != fold)).drop([
            "fold",
            "target",
            "session_id",
            "yad_no",
        ])
        y_train = train.filter((pl.col("session_id").is_in(use_session_ids)) & (pl.col("fold") != fold))[
            "target"
        ].to_numpy()
        x_valid = train.filter((pl.col("session_id").is_in(use_session_ids)) & (pl.col("fold") == fold)).drop([
            "fold",
            "target",
            "session_id",
            "yad_no",
        ])
        y_valid = train.filter((pl.col("session_id").is_in(use_session_ids)) & (pl.col("fold") == fold))[
            "target"
        ].to_numpy()

        LOGGER.info(f"y_train value_counts: \n{pd.Series(y_train).value_counts()}")
        LOGGER.info(f"y_valid value_counts: \n{pd.Series(y_valid).value_counts()}")

        x_train = x_train.to_pandas(use_pyarrow_extension_array=False)
        x_valid = x_valid.to_pandas(use_pyarrow_extension_array=False)
        for col in cfg.category_features:
            x_train[col] = x_train[col].astype("category")
            x_valid[col] = x_valid[col].astype("category")

        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            num_boost_round=10000,
            evals_result={},
            early_stopping_rounds=200,
            verbose_eval=100,
        )

        fig, ax = plt.subplots(figsize=(30, 20))
        xgb.plot_importance(xgb_model, ax=ax)
        fig.savefig(str(cfg.output_dir / f"importance_fold{fold}.png"))

        xgb_model_list.append(xgb_model)
        del x_train, y_train, x_valid, y_valid, dtrain, dvalid

    # 推論
    oof = pd.DataFrame()
    test = test.with_columns(pl.lit(0.0).alias("predict"))
    x_test = test.drop(["sessoin_id", "fold", "target"]).to_pandas(use_pyarrow_extension_array=False)
    for col in cfg.category_features:
        x_test[col] = x_test[col].astype("category")

    LOGGER.info(f"test shape: {x_test.shape}")

    for fold in range(cfg.n_splits):
        x_valid = train.filter((pl.col("fold") == fold)).drop(["fold", "target", "session_id"])
        y_valid = train.filter((pl.col("fold") == fold))["target"].to_numpy()

        x_valid = x_valid.to_pandas(use_pyarrow_extension_array=False)
        for col in cfg.category_features:
            x_valid[col] = x_valid[col].astype("category")

        x_valid["predict"] = xgb_model_list[fold].predict(xgb.DMatrix(x_valid.drop(["yad_no"], axis=1), enable_categorical=True))
        x_valid["session_id"] = train.filter((pl.col("fold") == fold))["session_id"].to_numpy()
        x_test["predict"] += (
            xgb_model_list[fold].predict(xgb.DMatrix(x_test.drop(["predict", "session_id", "yad_no"], axis=1), enable_categorical=True))
            / cfg.n_splits
        )
        oof = pd.concat([oof, x_valid[["session_id", "predict", "yad_no"]]], axis=0)

    oof = oof.sort_values(["session_id", "predict"], ascending=False)

    oof.to_csv(str(cfg.output_dir / "oof.csv"), index=False)

    oof_ = create_top_10_yad_predict(oof)
    label = pd.read_csv(constants.INPUT_DIR / "train_label.csv")
    score = mapk(
        actual=label[label["session_id"].isin(oof_.reset_index()["session_id"])]
        .sort_values("session_id", ascending=True)["yad_no"]
        .to_list(),
        predicted=oof_.values.tolist(),
        k=10,
    )
    LOGGER.info(f"map@10: {score}")

    test_session = pl.read_csv(constants.INPUT_DIR / "test_session.csv")
    x_test["session_id"] = test["session_id"].to_numpy()
    LOGGER.info(f"x_test shape: {x_test.shape}")
    x_test = x_test.query(f"session_id in {test_session['session_id'].to_list()}").reset_index(drop=True)
    LOGGER.info(f"x_test shape: {x_test.shape}")

    sub = create_top_10_yad_predict(x_test)
    sub.columns = [f"predict_{i}" for i in sub.columns]
    sub = sub.reset_index(drop=True)
    print(sub)
    sub.to_csv(f"{cfg.name}_submission.csv", index=False)


if __name__ == "__main__":
    main()
