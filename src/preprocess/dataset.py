from logging import getLogger

import numpy as np
import polars as pl

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

    def _make_candidates():
        """session_idごとじゃない候補生成の集約"""
        popular_candidates = candidates.make_popular_candidates(log_df, train_label_df)
        return popular_candidates

    candidates_ = _make_candidates()
    # 過去に見たことのあるyad_noを候補として追加する
    candidates_for_a_session_id = candidates.make_seen_candidates(log_df, session_ids)

    # make pairs of (session_id, yad_no)
    df = (
        pl.DataFrame({"session_id": session_ids})
        .with_columns([
            pl.col("session_id"),
            pl.Series("yad_no", [candidates_ for _ in range(len(session_ids))], dtype=pl.List),
        ])
        .explode("yad_no")
    )
    df = df.join(candidates_for_a_session_id, on="session_id", how="left")

    # 同じ候補は消す
    df = df.unique(subset=["session_id", "yad_no"]).select(["session_id", "yad_no"])
    df = candidates.make_covisit_candidates(df, covisit_matrix, k=10)

    # make features_df
    session_features_df = session_features.make_session_featuers(phase, log_df, session_ids)
    yad_features_df = yad_features.make_yad_features(yad_df)
    # attach features
    df = df.join(session_features_df, on="session_id", how="left").join(yad_features_df, on="yad_no", how="left")

    # 同じ候補は消す
    df = df.unique(subset=["session_id", "yad_no"])

    # TODO: 後でちゃんと処理を考える
    # nullの情報を落とす
    logger.info(f"before drop_nulls: {df.shape}")
    need_null_processing_cols = "cd"
    df = df.select([pl.col(col) for col in df.columns if not col.endswith(need_null_processing_cols)])
    logger.info(f"after drop_nulls: {df.shape}")

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
    from src import constants
    from src.utils.common import trace
    from src.utils.logger import get_root_logger

    _ = get_root_logger()

    pl.Config.set_tbl_cols(100)

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    train_label_df = pl.read_csv(constants.INPUT_DIR / "train_label.csv")
    yad_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    with trace("making dataframe..."):
        train_df = make_dataset(
            phase="train",
            yad_df=yad_df,
            train_log_df=train_log_df,
            test_log_df=test_log_df,
            train_label_df=train_label_df,
            session_ids=train_label_df["session_id"].unique().to_list(),
            covisit_matrix=covisit_matrix,
        )
        train_df = make_target(train_df, train_label_df)
        print("TRAIN_DF: ", train_df)
        print("TRAIN_DF Describe: ", train_df.describe())
        print("target_cnt: ", train_df["target"].value_counts())

    see_target_session_ids = [
        "75b912d7b31205bbdfe8eba26055312d",
    ]
    for session_id in see_target_session_ids:
        print(f"session_id: {session_id}")
        print(train_df.filter(pl.col("session_id") == session_id))


if __name__ == "__main__":
    _test_make_dataframe()
