import numpy as np
import polars as pl
import xgboost as xgb

from src.preprocess.dataset import make_dataset
from src.training.common import DataFrames


def predict(models, session_ids: list[str], dfs: DataFrames) -> pl.DataFrame:
    """

    Returns:
        pl.DataFrame: top10の予測結果
    """

    # 候補生成と特徴量付け
    dataset = make_dataset(
        phase="test",
        session_ids=session_ids,
        train_label_df=dfs.train_label_df,
        yad_df=dfs.yad_df,
        train_log_df=dfs.train_log_df,
        test_log_df=dfs.test_log_df,
    )
    print(dataset)

    # 予測
    y_preds = np.zeros((len(dataset), 1))
    for model in models:
        if isinstance(model, xgb.Booster):
            data = xgb.DMatrix(dataset.drop(["session_id", "yad_no"]))
            y_pred = model.predict(data).reshape(-1, 1)
            y_pred = 1 / (1 + np.exp(-y_pred))
            print(y_pred.shape)
            y_preds += y_pred
        else:
            raise NotImplementedError

    # top10の予測結果を返す
    ranking = pl.concat(
        [dataset[["session_id", "yad_no"]], pl.DataFrame({"y_preds": y_preds.reshape(-1)})], how="horizontal"
    )
    print(ranking)
    ranking = ranking.sort(by="y_preds", descending=True)
    print(ranking)
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
    preds = predict(models, session_ids, dfs)
    sub = make_submission(preds)
    print("SUB: ", sub)

    label = pl.DataFrame({"session_id": session_ids}).join(
        dfs.train_label_df.filter(pl.col("session_id").is_in(session_ids)), how="left", on="session_id"
    )["yad_no"].to_list()
    sub = pl.DataFrame({"session_id": session_ids}).join(
        sub, how="left", on="session_id"
    ).select(["session_id", "yad_no"])
    print("LABEL: ", label)
    print("SUB: ", sub)

    map10 = metrics.mean_average_precision_at_k(label, sub["yad_no"].to_list(), k=10)
    print("MAP@10: ", map10)


if __name__ == "__main__":
    _test_predict()