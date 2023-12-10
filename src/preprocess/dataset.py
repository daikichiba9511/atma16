from logging import getLogger

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

from src.preprocess import candidates, session_features, yad_features

logger = getLogger(__name__)


def make_dataset(
    phase: str,
    yad_df: pl.DataFrame,
    train_log_df: pl.DataFrame,
    test_log_df: pl.DataFrame,
    train_label_df: pl.DataFrame,
    session_ids: list[str],
    covisit_matrix: np.ndarray,
    encoders: dict[str, LabelEncoder],
) -> pl.DataFrame:
    """make dataframe for session_ids

    make pairs of (session_id, yad_no) for each session_id.

    Args:
        phase: train or test
        train_log_df: train_log.csv
        test_log_df: test_log.csv
        train_label_df: train_label.csv
        session_ids: session_ids for train or test
    """

    log_df = pl.concat([train_log_df, test_log_df], how="vertical").drop_nulls()

    # session_id関係ない候補生成
    def _make_candidates():
        """session_idごとじゃない候補生成の集約"""
        popular_candidates = candidates.make_popular_candidates(log_df, train_label_df, k=20)
        popular_candidates_at_seq_0 = candidates.make_popular_candidates_at_seq_0(log_df, k=15)
        return np.concatenate([popular_candidates, popular_candidates_at_seq_0], axis=0)

    # make pairs of (session_id, yad_no)
    candidates_ = _make_candidates()
    df = (
        pl.DataFrame({"session_id": session_ids})
        .with_columns([
            pl.col("session_id"),
            pl.Series("yad_no", [candidates_ for _ in range(len(session_ids))], dtype=pl.List),
        ])
        .explode("yad_no")
    )

    # 過去に見たことのあるyad_noを候補として追加する
    candidates_for_a_session_id = candidates.make_seen_candidates(log_df, session_ids)
    df = df.join(candidates_for_a_session_id, on="session_id", how="left")


    # 同じ候補は消す
    df = df.unique(subset=["session_id", "yad_no"]).select(["session_id", "yad_no"])
    df = candidates.make_covisit_candidates(df, covisit_matrix, k=10)

    # 実際に選ばれたのも候補に入れる
    # バイアスが入る。実際に選ばれたものが入るので
    if phase == "train":
        label_df = train_label_df.select(["session_id", "yad_no"])
        df = pl.concat([df, label_df], how="vertical").drop_nulls()

    # make features_df
    session_features_df = session_features.make_session_featuers(phase, log_df, session_ids)
    yad_features_df = yad_features.make_yad_features(yad_df)
    # attach features
    df = df.join(session_features_df, on="session_id", how="left").join(yad_features_df, on="yad_no", how="left")

    # 同じ候補は消す
    df = df.unique(subset=["session_id", "yad_no"])

    # TODO: 後でちゃんと処理を考える
    # nullの情報を落とす
    if phase == "test":
        encoded_wid_cd = encoders["wid_cd"].transform(df["wid_cd"].to_list())
        encoded_ken_cd = encoders["ken_cd"].transform(df["ken_cd"].to_list())
        encoded_lrg_cd = encoders["lrg_cd"].transform(df["lrg_cd"].to_list())
        encoded_sml_cd = encoders["sml_cd"].transform(df["sml_cd"].to_list())
    else:
        encoded_wid_cd = encoders["wid_cd"].fit_transform(df["wid_cd"].to_list())
        encoded_ken_cd = encoders["ken_cd"].fit_transform(df["ken_cd"].to_list())
        encoded_lrg_cd = encoders["lrg_cd"].fit_transform(df["lrg_cd"].to_list())
        encoded_sml_cd = encoders["sml_cd"].fit_transform(df["sml_cd"].to_list())

    df = df.with_columns(
        pl.Series("encoded_wid_cd", encoded_wid_cd),
        pl.Series("encoded_ken_cd", encoded_ken_cd),
        pl.Series("encoded_lrg_cd", encoded_lrg_cd),
        pl.Series("encoded_sml_cd", encoded_sml_cd),
    ).drop(["wid_cd", "ken_cd", "lrg_cd", "sml_cd"])

    return df


def make_target(featured_pair_df: pl.DataFrame, train_label_df: pl.DataFrame) -> pl.DataFrame:
    """make target dataframe

    Args:
        featured_pair_df: dataframe with session_id, yad_no and features
        train_label_df: train_label.csv

    Returns:
        dataframe with session_id, yad_no and target
    """
    target_df = train_label_df.select(["session_id", "yad_no"]).with_columns(pl.lit(1).alias("target"))
    df = featured_pair_df.join(target_df, on=["session_id", "yad_no"], how="left").with_columns(
        pl.col("target").fill_null(0).alias("target")
    )
    return df


def negative_sampling(df: pl.DataFrame, sampling_rate: float) -> pl.DataFrame:
    """negative sampling

    Args:
        df: dataframe with session_id, yad_no and target
        sampling_rate: sampling rate
    """

    positive_df = df.filter(pl.col("target") == 1)
    negative_df = df.filter(pl.col("target") == 0).sample(fraction=sampling_rate)
    df_ = pl.concat([positive_df, negative_df], how="vertical")
    return df_


def _test_make_dataframe():
    from tqdm.auto import tqdm

    from src import constants
    from src.training.common import load_dataframes
    from src.utils.common import trace
    from src.utils.logger import get_root_logger

    _ = get_root_logger()

    pl.Config.set_tbl_cols(100)

    # dfs = load_dataframes().sample(n=50000)
    dfs = load_dataframes().sample(n=10000)
    # dfs = load_dataframes().sample(n=5000)
    # dfs = load_dataframes()
    covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    encoders = {
        "wid_cd": LabelEncoder(),
        "ken_cd": LabelEncoder(),
        "lrg_cd": LabelEncoder(),
        "sml_cd": LabelEncoder(),
    }
    with trace("making dataframe..."):
        train_df = make_dataset(
            phase="train",
            yad_df=dfs.yad_df,
            train_log_df=dfs.train_log_df,
            test_log_df=dfs.test_log_df,
            train_label_df=dfs.train_label_df,
            session_ids=dfs.train_label_df["session_id"].unique().to_list(),
            covisit_matrix=covisit_matrix,
            encoders=encoders,
        )
        train_df = make_target(train_df, dfs.train_label_df)
        print("TRAIN_DF: ", train_df)
        print("TRAIN_DF Describe: ", train_df.describe())
        print("target_cnt: ", train_df["target"].value_counts())

        unique_session_ids = train_df["session_id"].unique().to_list()
        cnt_label_in_candidates = 0
        for session_id in tqdm(unique_session_ids, total=len(unique_session_ids)):
            train_df_this_session = train_df.filter(pl.col("session_id") == session_id)
            train_label_this_session = dfs.train_label_df.filter(pl.col("session_id") == session_id)
            intersection = set(train_df_this_session["yad_no"].to_list()).intersection(
                set(train_label_this_session["yad_no"].to_list())
            )
            cnt_label_in_candidates += len(intersection)
        print(
            f"rate of label in candidates: {cnt_label_in_candidates / len(unique_session_ids)} (all: {len(unique_session_ids)}, num: {cnt_label_in_candidates})"
        )

    # see_target_session_ids = [
    #     "75b912d7b31205bbdfe8eba26055312d",
    # ]
    # for session_id in see_target_session_ids:
    #     print(f"session_id: {session_id}")
    #     print(train_df.filter(pl.col("session_id") == session_id))


if __name__ == "__main__":
    _test_make_dataframe()
