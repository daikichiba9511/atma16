import collections
import functools
from logging import getLogger
from typing import TypedDict

import numpy as np
import polars as pl
from gensim.models import word2vec
from sklearn.preprocessing import LabelEncoder

from src.preprocess import candidates, session_features, yad_features

logger = getLogger(__name__)


class Encoders(TypedDict):
    wid_cd: LabelEncoder
    ken_cd: LabelEncoder
    lrg_cd: LabelEncoder
    sml_cd: LabelEncoder
    word2vec: word2vec.Word2Vec | None


def make_dataset(
    phase: str,
    yad_df: pl.DataFrame,
    train_log_df: pl.DataFrame,
    test_log_df: pl.DataFrame,
    train_label_df: pl.DataFrame,
    session_ids: list[str],
    covisit_matrix: np.ndarray,
    encoders: Encoders,
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

    # nullの情報を落とす, strなのでnaiveにlabel encoding
    if phase == "train":
        encoded_wid_cd = encoders["wid_cd"].fit_transform(yad_df["wid_cd"].to_list())
        encoded_ken_cd = encoders["ken_cd"].fit_transform(yad_df["ken_cd"].to_list())
        encoded_lrg_cd = encoders["lrg_cd"].fit_transform(yad_df["lrg_cd"].to_list())
        encoded_sml_cd = encoders["sml_cd"].fit_transform(yad_df["sml_cd"].to_list())
    else:
        encoded_wid_cd = encoders["wid_cd"].transform(yad_df["wid_cd"].to_list())
        encoded_ken_cd = encoders["ken_cd"].transform(yad_df["ken_cd"].to_list())
        encoded_lrg_cd = encoders["lrg_cd"].transform(yad_df["lrg_cd"].to_list())
        encoded_sml_cd = encoders["sml_cd"].transform(yad_df["sml_cd"].to_list())

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
    df = (
        pl.concat([df, candidates_for_a_session_id], how="vertical")
        .filter(pl.col("session_id").is_in(session_ids))
        .unique(subset=["session_id", "yad_no"])
    )

    log_latest_df = log_df.group_by("session_id").tail(1).rename({"yad_no": "latest_yad_no"})
    df = df.join(log_latest_df, on="session_id", how="left")

    log_latest_df = (
        log_latest_df.join(train_label_df, on="session_id", how="left")
        .group_by(["latest_yad_no", "yad_no"])
        .count()
        .sort(by=["latest_yad_no", "count"], descending=[False, True])
    )
    latest_candidates = (
        log_latest_df.group_by("latest_yad_no").head(10).select(["latest_yad_no", "yad_no"]).drop_nulls()
    )

    latest_candidates = latest_candidates.join(df, on="latest_yad_no", how="inner").select(["session_id", "yad_no"])
    df = pl.concat(
        [
            df.select(["session_id", "yad_no"]),
            latest_candidates.select(["session_id", pl.col("yad_no").cast(pl.Int32)]),
        ],
        how="vertical",
    )

    # 興味のあるエリアのyad_noを候補として追加する
    area_candidats_df = candidates.make_popular_candidates_in_interested_area(
        df=log_df, session_ids=session_ids, yad_df=yad_df, k=10
    ).select(["session_id", pl.col("yad_no").cast(pl.Int32).alias("yad_no")])
    df = (
        pl.concat([df, area_candidats_df], how="vertical")
        .unique(subset=["session_id", "yad_no"])
        .select([
            "session_id",
            pl.col("yad_no").cast(pl.Int32).alias("yad_no"),
        ])
    )

    # dataframeのサイズを落とす
    # ナイーブにやると簡単に大きくなってOOMになる
    df = (
        df.filter(pl.col("session_id").is_in(session_ids))
        .drop_nulls()
        .select(pl.all().shrink_dtype())
        .with_columns(pl.col("yad_no").cast(pl.Int64).alias("yad_no"), "session_id")
    )

    # make features_df
    def attach_session_feats(df, phase, log_df, session_ids):
        session_features_df = session_features.make_session_featuers(phase, log_df, session_ids, yad_df=yad_df)
        session_features_df = session_features_df.select(pl.all().shrink_dtype())
        df = df.join(session_features_df, on="session_id", how="left")
        return df

    def attach_yad_feats(df, yad_df, log_df):
        yad_features_df = (
            yad_features.make_yad_features(log_df, yad_df)
            .select(pl.all().shrink_dtype())
            .with_columns(pl.col("yad_no").cast(pl.Int64).alias("yad_no"))
        )
        df = df.join(yad_features_df, on="yad_no", how="left")
        return df

    # attach features
    df = attach_session_feats(df, phase, log_df, session_ids)
    df = df.filter(pl.col("session_id").is_in(session_ids))
    df = attach_yad_feats(df, yad_df, log_df)
    df = df.filter(pl.col("session_id").is_in(session_ids))

    def _take_k_most_freq(x: list, k: int):
        """xの中で最も多いものを多い順にk個取る"""
        return [x[0] for x in collections.Counter(x).most_common(k)]

    def _make_topk_yad_per_area_cd(df: pl.DataFrame, area_cd_col: str, k: int):
        """areaごとにtopkのyad_noを返す

        Args:
            df: dataframe with yad_no and area_cd
            area_cd_col: area_cd column name
            k: topk

        Returns:
            | area_cd | topk_yad_no |
            | ------- | ----------- |
            | 1       | [1,2,3]     |
            | 2       | [2,3,1]     |
        """
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

    def attach_topk_ranking_in_the_area_as_feats(df):
        """特徴量として興味のあるエリアのtop10を追加する"""
        label_with_yado_info = train_label_df.join(yad_df, on="yad_no", how="left")
        top10_yad_no_sml_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "sml_cd", 10)
        top10_yad_no_lrg_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "lrg_cd", 10)
        top10_yad_no_ken_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "ken_cd", 10)
        top10_yad_no_wid_cd = _make_topk_yad_per_area_cd(label_with_yado_info, "wid_cd", 10)

        # # 特徴量として興味のあるエリアのtop10を追加する
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
        return df

    df = attach_topk_ranking_in_the_area_as_feats(df)

    def _make_area_popular_ranking(label_with_yado_info: pl.DataFrame, area_cd: str):
        """指定したarea_cdごとに人気の宿ランキングを返す

        もしかしたら上と同じもの返してるかも
        """
        popular_yado_ranking = (
            label_with_yado_info.group_by([area_cd, "yad_no"])
            .count()
            .sort(by=[area_cd, "count"], descending=[False, True])
        )
        popular_yado_area_feature = popular_yado_ranking.group_by(area_cd).map_groups(
            lambda group: group.with_columns(
                pl.col("count").rank(method="dense", descending=True).over(area_cd).alias(f"popular_{area_cd}_rank"),
            )
        )
        return popular_yado_area_feature.drop("count")

    def _attach_area_popular_ranking(df: pl.DataFrame):
        """特徴量として興味のあるエリアの人気の宿ランキングを追加する"""
        label_with_yado_info = train_label_df.join(yad_df, on="yad_no", how="left")
        sml_cd_popular_ranking = _make_area_popular_ranking(label_with_yado_info, "sml_cd")
        lrg_cd_popular_ranking = _make_area_popular_ranking(label_with_yado_info, "lrg_cd")
        ken_cd_popular_ranking = _make_area_popular_ranking(label_with_yado_info, "ken_cd")
        wid_cd_popular_ranking = _make_area_popular_ranking(label_with_yado_info, "wid_cd")

        df = (
            df.with_columns(
                pl.col("sml_cd").cast(pl.UInt32),
                pl.col("lrg_cd").cast(pl.UInt32),
                pl.col("ken_cd").cast(pl.UInt32),
                pl.col("wid_cd").cast(pl.UInt32),
            )
            .join(sml_cd_popular_ranking, on=["sml_cd", "yad_no"], how="left")
            .join(lrg_cd_popular_ranking, on=["lrg_cd", "yad_no"], how="left")
            .join(ken_cd_popular_ranking, on=["ken_cd", "yad_no"], how="left")
            .join(wid_cd_popular_ranking, on=["wid_cd", "yad_no"], how="left")
        )
        return df

    df = _attach_area_popular_ranking(df)

    # area codeが候補と同じかどうか
    df = df.with_columns(
        # last
        (pl.col("last_seen_sml_cd_0") == pl.col("sml_cd")).alias("last_is_same_sml_cd").cast(pl.Int32),
        (pl.col("last_seen_sml_cd_0") == pl.col("lrg_cd")).alias("last_is_same_lrg_cd").cast(pl.Int32),
        (pl.col("last_seen_sml_cd_0") == pl.col("ken_cd")).alias("last_is_same_ken_cd").cast(pl.Int32),
        (pl.col("last_seen_sml_cd_0") == pl.col("wid_cd")).alias("last_is_same_wid_cd").cast(pl.Int32),
        # first
        (pl.col("first_seen_sml_cd_0") == pl.col("sml_cd")).alias("first_is_same_sml_cd").cast(pl.Int32),
        (pl.col("first_seen_sml_cd_0") == pl.col("lrg_cd")).alias("first_is_same_lrg_cd").cast(pl.Int32),
        (pl.col("first_seen_sml_cd_0") == pl.col("ken_cd")).alias("first_is_same_ken_cd").cast(pl.Int32),
        (pl.col("first_seen_sml_cd_0") == pl.col("wid_cd")).alias("first_is_same_wid_cd").cast(pl.Int32),
        # first is same as last
        (pl.col("first_seen_sml_cd_0") == pl.col("last_seen_sml_cd_0"))
        .alias("first_is_same_as_last_sml_cd")
        .cast(pl.Int32),
        (pl.col("first_seen_lrg_cd_0") == pl.col("last_seen_sml_cd_0"))
        .alias("first_is_same_as_last_lrg_cd")
        .cast(pl.Int32),
        (pl.col("first_seen_ken_cd_0") == pl.col("last_seen_sml_cd_0"))
        .alias("first_is_same_as_last_ken_cd")
        .cast(pl.Int32),
        (pl.col("first_seen_wid_cd_0") == pl.col("last_seen_sml_cd_0"))
        .alias("first_is_same_as_last_wid_cd")
        .cast(pl.Int32),
    )

    # logの埋め込みからユーザの傾向をベクトルに埋め込みたい
    if phase == "train":
        session_to_yad_no_list = log_df.group_by("session_id").agg(pl.col("yad_no"))
        encoders["word2vec"] = word2vec.Word2Vec(
            session_to_yad_no_list["yad_no"].to_list(), vector_size=32, window=5, min_count=1, workers=4
        )

    if encoders["word2vec"] is None:
        raise ValueError("word2vec is None in encoders")

    yad_to_vec = {yad_no: encoders["word2vec"].wv[yad_no] for yad_no in encoders["word2vec"].wv.index_to_key}
    df = (
        df.with_columns(
            pl.col("yad_no").cast(pl.Int32).replace(yad_to_vec, default=None).alias("session_yad_no_log_vec"),
        )
        .with_columns(*[pl.col("session_yad_no_log_vec").list.get(i).alias(f"yad_no_vec_{i}") for i in range(32)])
        .drop("session_yad_no_log_vec")
    )

    # 同じ候補は消す
    df = df.unique(subset=["session_id", "yad_no"])
    df = df.filter(pl.col("session_id").is_in(session_ids))

    df = df.select(pl.all().shrink_dtype()).fill_null(0)
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
    encoders: Encoders = {
        "wid_cd": LabelEncoder(),
        "ken_cd": LabelEncoder(),
        "lrg_cd": LabelEncoder(),
        "sml_cd": LabelEncoder(),
        "word2vec": None,
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
        print("num cols: ", len(cols))
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
