import collections
import functools
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
    # TODO: 後でちゃんと処理を考える
    # nullの情報を落とす
    if phase == "test":
        encoded_wid_cd = encoders["wid_cd"].transform(yad_df["wid_cd"].to_list())
        encoded_ken_cd = encoders["ken_cd"].transform(yad_df["ken_cd"].to_list())
        encoded_lrg_cd = encoders["lrg_cd"].transform(yad_df["lrg_cd"].to_list())
        encoded_sml_cd = encoders["sml_cd"].transform(yad_df["sml_cd"].to_list())
    else:
        encoded_wid_cd = encoders["wid_cd"].fit_transform(yad_df["wid_cd"].to_list())
        encoded_ken_cd = encoders["ken_cd"].fit_transform(yad_df["ken_cd"].to_list())
        encoded_lrg_cd = encoders["lrg_cd"].fit_transform(yad_df["lrg_cd"].to_list())
        encoded_sml_cd = encoders["sml_cd"].fit_transform(yad_df["sml_cd"].to_list())

    yad_df = yad_df.with_columns(
        pl.Series("wid_cd", encoded_wid_cd, dtype=pl.UInt32),
        pl.Series("ken_cd", encoded_ken_cd, dtype=pl.UInt32),
        pl.Series("lrg_cd", encoded_lrg_cd, dtype=pl.UInt32),
        pl.Series("sml_cd", encoded_sml_cd, dtype=pl.UInt32),
    )

    # session_id関係ない候補生成
    def _make_candidates():
        """session_idごとじゃない候補生成の集約"""
        popular_candidates = candidates.make_popular_candidates(log_df, train_label_df, k=20)
        # popular_candidates_at_seq_0 = candidates.make_popular_candidates_at_seq_0(log_df, k=15)
        # return np.concatenate([popular_candidates, popular_candidates_at_seq_0], axis=0).astype(np.int32)

        return popular_candidates.astype(np.int32)

    # make pairs of (session_id, yad_no)
    candidates_ = _make_candidates()
    df = pl.DataFrame({"session_id": session_ids}).with_columns([
        pl.Series("yad_no", [candidates_ for _ in range(len(session_ids))], dtype=pl.List),
    ])
    df = df.explode("yad_no")

    # df = pl.DataFrame()

    # 過去に見たことのあるyad_noを候補として追加する
    candidates_for_a_session_id = candidates.make_seen_candidates(log_df, session_ids).select([
        "session_id",
        pl.col("yad_no").cast(pl.Int32).alias("yad_no"),
    ])
    df = pl.concat([df, candidates_for_a_session_id], how="vertical")
    df = df.filter(pl.col("session_id").is_in(session_ids)).unique(subset=["session_id", "yad_no"])

    # 興味のあるエリアのyad_noを候補として追加する
    area_candidats_df = candidates.make_popular_candidates_in_interested_area(
        df=log_df, session_ids=session_ids, yad_df=yad_df, k=10
    ).select(["session_id", pl.col("yad_no").cast(pl.Int32).alias("yad_no")])

    # 同じ候補は消す
    df = (
        pl.concat([df, area_candidats_df], how="vertical")
        .unique(subset=["session_id", "yad_no"])
        .select([
            "session_id",
            pl.col("yad_no").cast(pl.Int32).alias("yad_no"),
        ])
    )
    # df = candidates.make_covisit_candidates(df, covisit_matrix, k=10)
    # df = df.filter(pl.col("session_id").is_in(session_ids)).select([
    #     "session_id",
    #     pl.col("yad_no").cast(pl.Int32).alias("yad_no"),
    # ])

    # 実際に選ばれたのも候補に入れる
    # バイアスが入る。実際に選ばれたものが入るので
    # if phase == "train":
    #     df = pl.concat(
    #         [df, train_label_df.select(["session_id", pl.col("yad_no").cast(pl.Int32).alias("yad_no")])],
    #         how="vertical",
    #     ).drop_nulls()

    # dataframeのサイズを落とす
    # ナイーブにやると簡単に大きくなってOOMになる
    df = (
        df.filter(pl.col("session_id").is_in(session_ids))
        .drop_nulls()
        .select(pl.all().shrink_dtype())
        .with_columns(pl.col("yad_no").cast(pl.Int64).alias("yad_no"), "session_id")
    )

    # make features_df
    session_features_df = session_features.make_session_featuers(phase, log_df, session_ids, yad_df=yad_df).select(
        pl.all().shrink_dtype(),
    )
    yad_features_df = (
        yad_features.make_yad_features(yad_df)
        .select(pl.all().shrink_dtype())
        .with_columns(pl.col("yad_no").cast(pl.Int64).alias("yad_no"))
    )

    # attach features
    df = (
        df.with_columns(pl.col("yad_no").cast(pl.Int64))
        .join(session_features_df, on="session_id", how="left")
        .join(yad_features_df, on="yad_no", how="left")
    )
    df = df.filter(pl.col("session_id").is_in(session_ids))

    label_with_yado_info = train_label_df.join(yad_df, on="yad_no", how="left")

    def _take_k_most_freq(x: list, k: int):
        cnt = collections.Counter(x)
        sorted_cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_cnt[:k]]

    def _make_topk_yad_per_area_cd(df: pl.DataFrame, area_cd_col: str, k: int):
        topk_yad_no_per_area_cd = (
            df.group_by(area_cd_col)
            .agg(pl.col("yad_no").alias(f"topk_{area_cd_col}_yad_no"))
            .map_rows(lambda x: (x[0], _take_k_most_freq(x[1], k)))
            .with_columns(
                pl.col("column_0").alias(area_cd_col).cast(pl.UInt16),
                pl.col("column_1").alias(f"top{k}_{area_cd_col}_yad_no"),
            )
            .with_columns(*[
                pl.col(f"top{k}_{area_cd_col}_yad_no")
                .map_elements(functools.partial(lambda x, i: x[i] if len(x) > i else None, i=i))
                .alias(f"top{i}_{area_cd_col}_yad_no")
                for i in range(k)
            ])
        ).drop(["column_0", "column_1", f"top{k}_{area_cd_col}_yad_no"])
        return topk_yad_no_per_area_cd

    top10_yad_no_sml_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "sml_cd", 10)
    top10_yad_no_lrg_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "lrg_cd", 10)
    top10_yad_no_ken_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "ken_cd", 10)
    top10_yad_no_wid_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "wid_cd", 10)

    # 特徴量として興味のあるエリアのtop10を追加する
    df = (
        df.with_columns(
            pl.col("sml_cd").cast(pl.UInt16),
            pl.col("lrg_cd").cast(pl.UInt16),
            pl.col("ken_cd").cast(pl.UInt16),
            pl.col("wid_cd").cast(pl.UInt16),
        )
        .join(top10_yad_no_sml_cd, on="sml_cd", how="left")
        .join(top10_yad_no_lrg_cd, on="lrg_cd", how="left")
        .join(top10_yad_no_ken_cd, on="ken_cd", how="left")
        .join(top10_yad_no_wid_cd, on="wid_cd", how="left")
    )

    # 同じ候補は消す
    df = df.unique(subset=["session_id", "yad_no"])
    df = df.filter(pl.col("session_id").is_in(session_ids))

    df = df.select(pl.all().shrink_dtype())

    df = df.fill_null(0)

    return df


def make_target(featured_pair_df: pl.DataFrame, train_label_df: pl.DataFrame) -> pl.DataFrame:
    """make target dataframe

    Args:
        featured_pair_df: dataframe with session_id, yad_no and features
        train_label_df: train_label.csv

    Returns:
        dataframe with session_id, yad_no and target
    """
    target_df = train_label_df.select(["session_id", pl.col("yad_no").cast(pl.Int64)]).with_columns(
        pl.lit(1).alias("target")
    )
    df = (
        featured_pair_df.with_columns(pl.col("yad_no").cast(pl.Int64))
        .join(target_df, on=["session_id", "yad_no"], how="left")
        .with_columns(pl.col("target").fill_null(0).alias("target"))
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

    # pl.Config.set_tbl_cols(100)

    dfs = load_dataframes().sample(n=10000)
    # dfs = load_dataframes().sample(n=50000)
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

        cols = train_df.columns
        list_cols = [col for col in cols if train_df[col].dtype == pl.List]
        print("columns", cols)
        print("list cols", list_cols)
        print("list cols value_counts", train_df[list_cols])

        print("Num of Candidates: ", train_df.group_by("session_id").count())

        train_df.write_csv(constants.OUTPUT_DIR / "debug_train_df.csv")

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
