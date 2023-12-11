from logging import getLogger

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src import constants
from src.preprocess.dataset import make_dataset
from src.training.common import DataFrames

logger = getLogger(__name__)


def predict(
    models,
    session_ids: list[str],
    dfs: DataFrames,
    covisit_matrix: np.ndarray,
    encoders: dict[str, LabelEncoder],
    phase: str,
) -> pl.DataFrame:
    """予測を行う

    Args:
        models: list of xgb.Booster
        session_ids: session_ids for train or test
        dfs: DataFrames
        covisit_matrix: co-visitation matrix
        encoders: dict of LabelEncoder
        phase: train or test

    Returns:
        pl.DataFrame: top10の予測結果
    """

    # 候補生成と特徴量付け
    dataset = make_dataset(
        phase=phase,
        session_ids=session_ids,
        train_label_df=dfs.train_label_df,
        yad_df=dfs.yad_df,
        train_log_df=dfs.train_log_df,
        test_log_df=dfs.test_log_df,
        covisit_matrix=covisit_matrix,
        encoders=encoders,
    )

    logger.info(f"dataset: {dataset}")
    if phase == "train":
        dataset.write_csv(constants.OUTPUT_DIR / "dataset.csv")

    # 予測
    y_preds = np.zeros((len(dataset), 1))
    for model in models:
        if isinstance(model, xgb.Booster):
            _dataset = dataset.drop(constants.NOT_USED_COLUMNS)
            data = xgb.DMatrix(_dataset, feature_names=_dataset.columns)
            y_pred = model.predict(data).reshape(-1, 1)
            y_preds += y_pred
        else:
            raise NotImplementedError

    # top10の予測結果を返す
    ranking = pl.concat(
        [dataset[["session_id", "yad_no"]], pl.DataFrame({"y_preds": y_preds.reshape(-1)})], how="horizontal"
    )
    ranking = ranking.sort(by="y_preds", descending=True)
    preds = ranking.group_by("session_id").head(10)
    return preds


def make_submission(preds: pl.DataFrame) -> pl.DataFrame:
    """予測結果からsubmission用のDataFrameを作成する

    Args:
        preds (pl.DataFrame): 予測結果

    Returns:
        pl.DataFrame: submission用のDataFrame
            * session_id
            * predict_0
            * predict_1
            * predict_2
            * predict_3
            * predict_4
            * predict_5
            * predict_6
            * predict_7
            * predict_8
            * predict_9
    """
    df = preds.select(["session_id", "yad_no"])
    # | session_id | yad_no  |
    # | ---------- | ------  |
    # | sample     | [1,2,3] |
    df = df.group_by("session_id").agg(pl.col("yad_no"))
    df = df.lazy().with_columns(*[pl.col("yad_no").list.get(i).alias(f"predict_{i}") for i in range(10)]).collect()
    return df


def _test_predict():
    import xgboost as xgb

    from src import constants
    from src.training import metrics
    from src.training.common import load_dataframes

    dfs = load_dataframes()
    model = xgb.Booster()
    wpath = constants.OUTPUT_DIR / "exp000" / "xgb_model_fold0.ubj"
    print(wpath)
    if wpath.exists():
        print("LOAD MODEL FROM: ", wpath)
        model.load_model(str(wpath))
    models = [model]
    session_ids = dfs.train_label_df["session_id"].unique().to_list()[:10]
    covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    encoders = {
        "wid_cd": joblib.load(constants.OUTPUT_DIR / "exp000" / "wid_cd_encoder.pkl"),
        "ken_cd": joblib.load(constants.OUTPUT_DIR / "exp000" / "ken_cd_encoder.pkl"),
        "lrg_cd": joblib.load(constants.OUTPUT_DIR / "exp000" / "lrg_cd_encoder.pkl"),
        "sml_cd": joblib.load(constants.OUTPUT_DIR / "exp000" / "sml_cd_encoder.pkl"),
    }
    preds = predict(models, session_ids, dfs, covisit_matrix=covisit_matrix, encoders=encoders, phase="train")
    sub = make_submission(preds)

    label = pl.DataFrame({"session_id": session_ids}).join(
        dfs.train_label_df.filter(pl.col("session_id").is_in(session_ids)), how="left", on="session_id"
    )
    sub = pl.DataFrame({"session_id": session_ids}).join(sub, how="left", on="session_id")
    print("LABEL: ", label)
    print("SUB: ", sub)

    map10 = metrics.mean_average_precision_at_k(label["yad_no"].to_list(), sub["yad_no"].to_list(), k=10)
    print("MAP@10: ", map10)


if __name__ == "__main__":
    _test_predict()
