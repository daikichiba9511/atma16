import pandas as pd
import polars as pl
from sklearn import model_selection


def make_fold(df: pl.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """make fold for train

    Args:
        df: dataframe with session_id and yad_no

    Returns:
        dataframe with session_id, yad_no and fold
    """
    # session_idごとに分割する
    # 推論時にはsession_idごとに精度を評価するので
    kfold = model_selection.GroupKFold(n_splits=n_splits)

    # 冪等性を担保するためにcloneしてる。また特定のインデックスに挿入するのにpd.DataFrameを使う
    _df = df.clone().with_columns(pl.lit(0).alias("fold")).to_pandas(use_pyarrow_extension_array=True)

    for fold, (_, valid_index) in enumerate(kfold.split(_df, y=_df["target"], groups=_df["session_id"])):
        _df.loc[valid_index, "fold"] = fold
    _df.loc[:, "fold"] = _df["fold"].astype(int)
    return _df


def _test_make_fold():
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    from src.preprocess import dataset
    from src.training import common

    pl.Config.set_tbl_cols(100)

    # 選ぶsession_idのユニーク数
    sample_size = 5000
    dfs = common.load_dataframes().sample(n=sample_size)
    encoders = {
        "wid_cd": LabelEncoder(),
        "ken_cd": LabelEncoder(),
        "lrg_cd": LabelEncoder(),
        "sml_cd": LabelEncoder(),
    }
    covisit_matrix = np.load(common.constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    df = dataset.make_dataset(
        phase="train",
        yad_df=dfs.yad_df,
        train_log_df=dfs.train_log_df,
        test_log_df=dfs.test_log_df,
        train_label_df=dfs.train_label_df,
        session_ids=dfs.train_label_df["session_id"].unique().to_list(),
        encoders=encoders,
        covisit_matrix=covisit_matrix,
    )
    folded_df = make_fold(df, n_splits=5)

    print(df)
    print(folded_df)
    print(folded_df["fold"].value_counts())
    print(folded_df.groupby("fold")["target"].value_counts())


if __name__ == "__main__":
    _test_make_fold()
