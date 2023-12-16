import dataclasses
import pathlib
import pprint
from logging import INFO

import numpy as np
import polars as pl
from gensim.models import word2vec
from sklearn import model_selection
from tqdm.auto import tqdm

from src import constants
from src.training.common import load_dataframes
from src.utils.logger import get_root_logger

LOGGER = get_root_logger(INFO)


@dataclasses.dataclass
class Config:
    seed: int = 42
    n_splits: int = 5

    output_candidates_dir: pathlib.Path = constants.OUTPUT_DIR / "candidates"
    output_features_dir: pathlib.Path = constants.OUTPUT_DIR / "features"


def get_session_ids(log: pl.DataFrame) -> pl.DataFrame:
    return log.group_by("session_id").head(1).select(["session_id"])


def make_session2area_yado_list(log: pl.DataFrame, yado: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    # uniqueなarea_cdの数はlog全体でsml_cd(95%が1), lrg_cd(98%が1), ken_cd(99.7%が1)
    log_yado = log.join(yado, on="yad_no", how="left")
    session2area_cd_list = (
        log_yado.group_by("session_id")
        .agg(
            pl.col("sml_cd"),
            pl.col("lrg_cd"),
            pl.col("ken_cd"),
            pl.col("wid_cd"),
        )
        .with_columns(
            pl.col("sml_cd").list.unique().alias("sml_uni_cd"),
            pl.col("lrg_cd").list.unique().alias("lrg_uni_cd"),
            pl.col("ken_cd").list.unique().alias("ken_uni_cd"),
            pl.col("wid_cd").list.unique().alias("wid_uni_cd"),
        )
        .with_columns(
            pl.col("sml_uni_cd").list.len().alias("num_unique_sml_cd"),
            pl.col("lrg_uni_cd").list.len().alias("num_unique_lrg_cd"),
            pl.col("ken_uni_cd").list.len().alias("num_unique_ken_cd"),
            pl.col("wid_uni_cd").list.len().alias("num_unique_wid_cd"),
        )
    )
    session2area_cd_0 = session2area_cd_list.with_columns(
        pl.col("sml_uni_cd").list.get(0).alias("session_interested_in_sml_cd_0"),
        pl.col("lrg_uni_cd").list.get(0).alias("session_interested_in_lrg_cd_0"),
        pl.col("ken_uni_cd").list.get(0).alias("session_interested_in_ken_cd_0"),
        pl.col("wid_uni_cd").list.get(0).alias("session_interested_in_wid_cd_0"),
    ).drop(["sml_uni_cd", "lrg_uni_cd", "ken_uni_cd", "wid_uni_cd"])
    for phase in ["train", "test"]:
        for area_col_name in [
            "top10_sml_popular_yado",
            "top10_lrg_popular_yado",
            "top10_ken_popular_yado",
            "top10_wid_popular_yado"
        ]:
            candidates = pl.read_parquet(cfg.output_candidates_dir / f"{phase}_{area_col_name}_candidates.parquet")
            area_cd = {
                "top10_sml_popular_yado": "sml_cd",
                "top10_lrg_popular_yado": "lrg_cd",
                "top10_ken_popular_yado": "ken_cd",
                "top10_wid_popular_yado": "wid_cd",
            }[area_col_name]
            area2yad_list = (
                candidates.group_by(area_cd)
                .agg(pl.col("yad_no"))
                .rename({area_cd: f"session_interested_in_{area_cd}_0", "yad_no": f"{area_cd}_yad_no_list"})
            )
            session2area_cd_0 = session2area_cd_0.join(
                area2yad_list, how="left", on=f"session_interested_in_{area_cd}_0"
            )
    session2area_cd_0 = session2area_cd_0.select([
        "session_id",
        "session_interested_in_sml_cd_0",
        "session_interested_in_lrg_cd_0",
        "session_interested_in_ken_cd_0",
        "session_interested_in_wid_cd_0",
        "num_unique_sml_cd",
        "num_unique_lrg_cd",
        "num_unique_ken_cd",
        "num_unique_wid_cd",
        "sml_cd_yad_no_list",
        "lrg_cd_yad_no_list",
        "ken_cd_yad_no_list",
        "wid_cd_yad_no_list",
    ])
    return session2area_cd_0


def main() -> None:
    cfg = Config()

    dfs = load_dataframes()
    train_log = dfs.train_log_df
    label = dfs.train_label_df
    test_log = dfs.test_log_df
    yado = dfs.yad_df

    # -- Make fold
    kf = model_selection.KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_assignments = np.full(label.height, -1, dtype=np.int32)
    for i, (_, valid_index) in enumerate(kf.split(label.to_numpy())):
        fold_assignments[valid_index] = i
    label = label.with_columns(pl.Series("fold", fold_assignments, dtype=pl.UInt32))

    # -- Make candidates
    candidates_name_list = [
        "past_view_yado_candidates",
        # "top10_popular_yado_candidates",
        "latest_next_booking_top20_candidates",
        "top10_sml_popular_yado_candidates",
        "top10_lrg_popular_yado_candidates",
        "top10_ken_popular_yado_candidates",
        "top10_wid_popular_yado_candidates",
        "co_visit_matrix_topk_candidates",
        "log_next_seq_candidates",
    ]

    session2area_yado_list = make_session2area_yado_list(pl.concat([train_log, test_log]), yado, cfg)

    def make_candidates():
        train_session_ids = get_session_ids(train_log)
        train_session_ids = train_session_ids.join(label.select(["fold", "session_id"]), how="left", on="session_id")
        test_session_ids = get_session_ids(test_log)
        LOGGER.info(f"num train session_ids {train_session_ids.height}")
        LOGGER.info(f"num test session_ids {test_session_ids.height}")

        candidates = dict(train=[], test=[])
        for phase in ["train", "test"]:
            for candidate_name in tqdm(candidates_name_list):
                candidate = pl.read_parquet(cfg.output_candidates_dir / f"{phase}_{candidate_name}.parquet")

                if "session_id" in candidate.columns:
                    candidates[phase].append(
                        candidate.select((["session_id", "yad_no"])).unique(["session_id", "yad_no"])
                    )
                    del candidate

                elif "latest_yad_no" in candidate.columns:
                    if phase == "train":
                        latest_yad_no = (
                            train_log.group_by("session_id")
                            .tail(1)
                            .select(["session_id", "yad_no"])
                            .rename({"yad_no": "latest_yad_no"})
                        )
                        latest_yad_no = latest_yad_no.join(
                            label.select(["session_id", "fold"]), how="left", on="session_id"
                        )
                        latest_yad_no = latest_yad_no.with_columns(pl.col("fold").cast(pl.Int32))

                        # co_visit
                        is_covisit = "fold" not in candidate
                        if is_covisit:
                            candidate = candidate.group_by("latest_yad_no").agg(pl.col("yad_no"))
                            candidate = latest_yad_no.join(candidate, how="left", on="latest_yad_no")
                            candidate = candidate.explode("yad_no").rename({"yad_no": "yad_no"})
                            print(candidate)

                        else:
                            candidate = latest_yad_no.join(candidate, how="inner", on=["latest_yad_no", "fold"])
                        candidates["train"].append(candidate.select(["session_id", "yad_no"]))
                        del latest_yad_no, candidate
                    else:
                        latest_yad_no = (
                            test_log.group_by("session_id")
                            .tail(1)
                            .select(["session_id", "yad_no"])
                            .rename({"yad_no": "latest_yad_no"})
                        )
                        candidate = latest_yad_no.join(candidate, how="inner", on=["latest_yad_no"])
                        candidates["test"].append(candidate.select(["session_id", "yad_no"]))
                        del latest_yad_no, candidate

                else:
                    if phase == "train":
                        if "fold" in candidate.columns:
                            candidate_all = pl.DataFrame()
                            for fold in range(cfg.n_splits):
                                train_session_ids_fold = train_session_ids.filter(pl.col("fold") == fold)
                                if candidate_name in [
                                    "top10_sml_popular_yado_candidates",
                                    "top10_lrg_popular_yado_candidates",
                                    "top10_ken_popular_yado_candidates",
                                    "top10_wid_popular_yado_candidates"
                                ]:
                                    area_cd = {
                                        "top10_sml_popular_yado_candidates": "sml_cd",
                                        "top10_lrg_popular_yado_candidates": "lrg_cd",
                                        "top10_ken_popular_yado_candidates": "ken_cd",
                                        "top10_wid_popular_yado_candidates": "wid_cd"
                                    }[candidate_name]
                                    session2area_yad_ranking = session2area_yado_list.select([
                                        "session_id",
                                        f"{area_cd}_yad_no_list",
                                    ])
                                    candidate = train_session_ids_fold.unique(["session_id"]).join(
                                        session2area_yad_ranking, how="left", on="session_id"
                                    )
                                    candidate_fold = (
                                        candidate.explode(f"{area_cd}_yad_no_list")
                                        .rename({f"{area_cd}_yad_no_list": "yad_no"})
                                        .with_columns(
                                            "session_id",
                                            pl.col("yad_no").cast(pl.Int64),
                                        )
                                        .lazy()
                                    )

                                else:
                                    candidate_fold = candidate.filter(pl.col("fold") == fold).select(["yad_no"])
                                    candidate_fold = train_session_ids_fold.lazy().join(
                                        candidate_fold.lazy(), how="cross"
                                    )
                                candidate_all = pl.concat([
                                    candidate_all,
                                    candidate_fold.collect(streaming=True),
                                ]).unique(["session_id", "yad_no"])
                                del candidate_fold

                            candidates["train"].append(candidate_all.select(["session_id", "yad_no"]))
                            del candidate_all
                    else:
                        if candidate_name in [
                            "top10_sml_popular_yado_candidates",
                            "top10_lrg_popular_yado_candidates",
                            "top10_ken_popular_yado_candidates",
                            "top10_wid_popular_yado_candidates"
                        ]:
                            area_cd = {
                                "top10_sml_popular_yado_candidates": "sml_cd",
                                "top10_lrg_popular_yado_candidates": "lrg_cd",
                                "top10_ken_popular_yado_candidates": "ken_cd",
                                "top10_wid_popular_yado_candidates": "wid_cd"
                            }[candidate_name]
                            session2area_yad_ranking = session2area_yado_list.select([
                                "session_id",
                                f"{area_cd}_yad_no_list",
                            ])
                            candidate = test_session_ids.join(session2area_yad_ranking, how="left", on="session_id")
                            candidate = (
                                candidate.explode(f"{area_cd}_yad_no_list")
                                .rename({f"{area_cd}_yad_no_list": "yad_no"})
                                .with_columns(
                                    "session_id",
                                    pl.col("yad_no").cast(pl.Int64),
                                )
                            )
                            candidates["test"].append(candidate.select(["session_id", "yad_no"]))
                            del candidate
                        else:
                            candidate_all = test_session_ids.join(candidate.select(["yad_no"]), how="cross")
                            candidates["test"].append(candidate_all.select(["session_id", "yad_no"]))
                            del candidate, candidate_all

        # 重複してる候補は無駄なのでuniqueを取る
        train_candidates = pl.concat([df.with_columns(pl.all().shrink_dtype()) for df in candidates["train"]]).unique([
            "session_id",
            "yad_no",
        ])
        test_candidates = pl.concat([df.with_columns(pl.all().shrink_dtype()) for df in candidates["test"]]).unique([
            "session_id",
            "yad_no",
        ])
        return train_candidates, test_candidates

    train_candidates, test_candidates = make_candidates()
    LOGGER.info(f"num each candidates in train {train_candidates.group_by('session_id').count()}")
    LOGGER.info(f"num each candidates in test {test_candidates.group_by('session_id').count()}")

    # -- Make features
    # | session_id | yad_no | target | fold |
    # | ---------- | ------ | ------ | ---- |
    # | hoge       | 1      | 1      | 0    |
    train_candidates = train_candidates.join(label.rename({"yad_no": "target"}), how="left", on="session_id")
    train_candidates = train_candidates.with_columns(pl.col("fold").cast(pl.Int32))
    train_candidates = train_candidates.with_columns(
        (pl.col("yad_no") == pl.col("target")).alias("target").cast(pl.Int8)
    )

    # | session_id | latest_yad_no |
    # | ---------- | ------------- |
    # | hoge       | 1             |
    train_latest_yad_no = (
        train_log.group_by("session_id").tail(1).select(["session_id", "yad_no"]).rename({"yad_no": "latest_yad_no"})
    )
    test_latest_yad_no = (
        test_log.group_by("session_id").tail(1).select(["session_id", "yad_no"]).rename({"yad_no": "latest_yad_no"})
    )

    # | session_id | yad_no | fold | target | latest_yad_no |
    # | ---------- | ------ | ---- | ------ | ------------- |
    # | hoge       | 1      | 0    | 1      | 1             |
    train_candidates = train_candidates.join(train_latest_yad_no, how="left", on="session_id").with_columns(
        pl.col("yad_no").cast(pl.UInt32)
    )

    # | session_id | yad_no | latest_yad_no |
    # | ---------- | ------ | ------------- |
    # | hoge       | 1      | 1             |
    test_candidates = test_candidates.join(test_latest_yad_no, how="left", on="session_id").with_columns(
        pl.col("yad_no").cast(pl.UInt32)
    )

    features_name_list = [
        "past_view_yado_features",
        "top10_popular_yado_features",
        "latest_next_booking_top20_features",
        "top10_sml_popular_yado_features",
        "top10_lrg_popular_yado_features",
        "top10_ken_popular_yado_features",
        "top10_wid_popular_yado_features",
        "co_visit_matrix_features",
    ]
    for phase in ["train", "test"]:
        for feature_name in tqdm(features_name_list):
            feature = pl.read_parquet(cfg.output_features_dir / f"{phase}_{feature_name}.parquet")
            feature = feature.with_columns(pl.col("yad_no").cast(pl.UInt32))
            print(feature)
            print(feature_name)

            if phase == "train":
                if "session_id" in feature.columns:
                    train_candidates = train_candidates.join(feature, how="left", on=["session_id", "yad_no"])
                elif "latest_yad_no" in feature.columns:
                    if "fold" not in feature.columns:
                        train_candidates = train_candidates.join(feature, how="left", on=["latest_yad_no", "yad_no"])
                    else:
                        train_candidates = train_candidates.join(
                            feature, how="left", on=["latest_yad_no", "fold", "yad_no"]
                        )
                else:
                    train_candidates = train_candidates.join(feature, how="left", on=["yad_no", "fold"])
            else:
                if "session_id" in feature.columns:
                    test_candidates = test_candidates.join(feature, how="left", on=["session_id", "yad_no"])
                elif "latest_yad_no" in feature.columns:
                    test_candidates = test_candidates.join(feature, how="left", on=["latest_yad_no", "yad_no"])
                else:
                    test_candidates = test_candidates.join(feature, how="left", on=["yad_no"])

    train_candidates = train_candidates.fill_null(0)
    test_candidates = test_candidates.fill_null(0)

    use_yad_features_list = [
        "yad_no",
        "yad_type",
        "total_room_cnt",
        "wireless_lan_flg",
        "onsen_flg",
        "kd_stn_5min",
        "kd_bch_5min",
        "kd_slp_5min",
        "kd_conv_walk_5min",
    ]
    train_candidates = train_candidates.join(
        yado.select(use_yad_features_list).with_columns(pl.col("yad_no").cast(pl.UInt32)), how="left", on="yad_no"
    ).join(
        yado.select([
            "yad_no",
            "sml_cd",
            "lrg_cd",
            "ken_cd",
            "total_room_cnt",
            "onsen_flg",
            "kd_conv_walk_5min",
            "yad_type",
        ]).rename({
            "yad_no": "latest_yad_no",
            "sml_cd": "latest_sml_cd",
            "lrg_cd": "latest_lrg_cd",
            "ken_cd": "latest_ken_cd",
            "total_room_cnt": "latest_total_room_cnt",
            "onsen_flg": "latest_onsen_flg",
            "kd_conv_walk_5min": "latest_kd_conv_walk_5min",
            "yad_type": "latest_yad_type",
        }),
        how="left",
        on="latest_yad_no",
    )

    test_candidates = test_candidates.join(
        yado.select(use_yad_features_list).with_columns(pl.col("yad_no").cast(pl.UInt32)), how="left", on="yad_no"
    ).join(
        yado.select([
            "yad_no",
            "sml_cd",
            "lrg_cd",
            "ken_cd",
            "total_room_cnt",
            "onsen_flg",
            "kd_conv_walk_5min",
            "yad_type",
        ]).rename({
            "yad_no": "latest_yad_no",
            "sml_cd": "latest_sml_cd",
            "lrg_cd": "latest_lrg_cd",
            "ken_cd": "latest_ken_cd",
            "total_room_cnt": "latest_total_room_cnt",
            "onsen_flg": "latest_onsen_flg",
            "kd_conv_walk_5min": "latest_kd_conv_walk_5min",
            "yad_type": "latest_yad_type",
        }),
        how="left",
        on="latest_yad_no",
    )

    # logをつかってsessionを特徴づける

    session2yado_list = pl.concat([train_log, test_log]).group_by("session_id").agg(pl.col("yad_no"))
    word2vec_model = word2vec.Word2Vec(
        session2yado_list["yad_no"].to_list(), vector_size=10, window=5, min_count=1, workers=4
    )
    session2yad_vec = {
        session: word2vec_model.wv[yad_no].astype(np.float32).reshape(-1).tolist()
        for session, yad_no in zip(session2yado_list["session_id"].to_list(), session2yado_list["yad_no"].to_list())
    }

    train_candidates = (
        train_candidates.with_columns(
            pl.col("session_id").replace(session2yad_vec, default=None).alias("session2yad_vec")
        )
        .with_columns(*[pl.col("session2yad_vec").list.get(i).alias(f"session2yad_vec_dim_{i}") for i in range(10)])
        .drop("session2yad_vec")
    )
    test_candidates = (
        test_candidates.with_columns(
            pl.col("session_id").replace(session2yad_vec, default=None).alias("session2yad_vec")
        )
        .with_columns(*[pl.col("session2yad_vec").list.get(i).alias(f"session2yad_vec_dim_{i}") for i in range(10)])
        .drop("session2yad_vec")
    )

    train_candidates = train_candidates.join(
        session2area_yado_list.select([
            "session_id",
            "session_interested_in_sml_cd_0",
            "session_interested_in_lrg_cd_0",
            "session_interested_in_ken_cd_0",
            "session_interested_in_wid_cd_0",
            "num_unique_sml_cd",
            "num_unique_lrg_cd",
            "num_unique_ken_cd",
            "num_unique_wid_cd"
        ]),
        on="session_id",
        how="left",
    )
    test_candidates = test_candidates.join(
        session2area_yado_list.select([
            "session_id",
            "session_interested_in_sml_cd_0",
            "session_interested_in_lrg_cd_0",
            "session_interested_in_ken_cd_0",
            "session_interested_in_wid_cd_0",
            "num_unique_sml_cd",
            "num_unique_lrg_cd",
            "num_unique_ken_cd",
            "num_unique_wid_cd"
        ]),
        on="session_id",
        how="left",
    )

    train_candidates = train_candidates.with_columns(
        (pl.col("session_interested_in_sml_cd_0") == pl.col("sml_cd")).alias("is_same_sml_cd").cast(pl.Int8),
        (pl.col("session_interested_in_ken_cd_0") == pl.col("ken_cd")).alias("is_same_lrg_cd").cast(pl.Int8),
        (pl.col("session_interested_in_lrg_cd_0") == pl.col("lrg_cd")).alias("is_same_ken_cd").cast(pl.Int8),
        (pl.col("session_interested_in_wid_cd_0") == pl.col("wid_cd")).alias("is_same_wid_cd").cast(pl.Int8),
    )
    test_candidates = test_candidates.with_columns(
        (pl.col("session_interested_in_sml_cd_0") == pl.col("sml_cd")).alias("is_same_sml_cd").cast(pl.Int8),
        (pl.col("session_interested_in_ken_cd_0") == pl.col("ken_cd")).alias("is_same_lrg_cd").cast(pl.Int8),
        (pl.col("session_interested_in_lrg_cd_0") == pl.col("lrg_cd")).alias("is_same_ken_cd").cast(pl.Int8),
        (pl.col("session_interested_in_wid_cd_0") == pl.col("wid_cd")).alias("is_same_wid_cd").cast(pl.Int8),
    )

    # yadoを特徴づける
    mean_total_room_cnt = yado["total_room_cnt"].mean()
    mean_wireless_lan_flg = yado["wireless_lan_flg"].mean()
    mean_kd_stn_5min = yado["kd_stn_5min"].mean()
    mean_kd_bch_5min = yado["kd_bch_5min"].mean()
    mean_kd_slp_5min = yado["kd_slp_5min"].mean()
    mean_kd_conv_walk_5min = yado["kd_conv_walk_5min"].mean()
    assert (
        mean_total_room_cnt is not None
        and mean_wireless_lan_flg is not None
        and mean_kd_stn_5min is not None
        and mean_kd_bch_5min is not None
        and mean_kd_slp_5min is not None
        and mean_kd_conv_walk_5min is not None
    )

    def _fill_null_mean_and_make_null_flg(df: pl.DataFrame, col_name: str, mean: float) -> list[pl.Expr]:
        """fill null with mean and make null_flg"""
        return [
            pl.col(col_name).is_null().cast(pl.Int32).alias(f"{col_name}_null_flg"),
            pl.col(col_name).fill_null(mean).alias(col_name),
        ]

    yado = yado.with_columns(
        *_fill_null_mean_and_make_null_flg(yado, "total_room_cnt", mean_total_room_cnt),
        *_fill_null_mean_and_make_null_flg(yado, "wireless_lan_flg", mean_wireless_lan_flg),
        *_fill_null_mean_and_make_null_flg(yado, "kd_stn_5min", mean_kd_stn_5min),
        *_fill_null_mean_and_make_null_flg(yado, "kd_bch_5min", mean_kd_bch_5min),
        *_fill_null_mean_and_make_null_flg(yado, "kd_slp_5min", mean_kd_slp_5min),
        *_fill_null_mean_and_make_null_flg(yado, "kd_conv_walk_5min", mean_kd_conv_walk_5min),
    )
    stats_total_room_cnt_with_onsen_flg = yado.group_by("onsen_flg").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_onsen_flg"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_onsen_flg"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_onsen_flg"),
    )
    stats_total_room_cnt_with_yad_type = yado.group_by("yad_type").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_yad_type"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_yad_type"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_yad_type"),
    )
    stats_total_room_cnt_with_kd_stn_5min = yado.group_by("kd_stn_5min").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_kd_stn_5min"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_kd_stn_5min"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_kd_stn_5min"),
    )
    stats_total_room_cnt_with_bcn_5min = yado.group_by("kd_bch_5min").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_kd_bch_5min"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_kd_bch_5min"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_kd_bch_5min"),
    )
    stats_total_roon_cnt_with_conv_walk_5min = yado.group_by("kd_conv_walk_5min").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_kd_conv_walk_5min"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_kd_conv_walk_5min"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_kd_conv_walk_5min"),
    )
    yado = (
        yado.join(stats_total_room_cnt_with_onsen_flg, on="onsen_flg", how="left")
        .join(stats_total_room_cnt_with_yad_type, on="yad_type", how="left")
        .join(stats_total_room_cnt_with_bcn_5min, on="kd_bch_5min", how="left")
        .join(stats_total_room_cnt_with_kd_stn_5min, on="kd_stn_5min", how="left")
        .join(stats_total_room_cnt_with_bcn_5min, on="kd_bch_5min", how="left")
        .join(stats_total_roon_cnt_with_conv_walk_5min, on="kd_conv_walk_5min", how="left")
    )
    yado = yado.with_columns(
        # (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_onsen_flg")).alias(
        #     "diff_total_room_cnt_with_onsen_flg"
        # ),
        (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_onsen_flg")).alias(
            "diff_raw_mean_total_room_cnt_with_onsen_flg"
        ),
        (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_onsen_flg")).alias(
            "diff_raw_median_total_room_cnt_with_onsen_flg"
        ),
        (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_yad_type")).alias(
            "diff_raw_mean_total_room_cnt_with_yad_type"
        ),
        (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_yad_type")).alias(
            "diff_raw_median_total_room_cnt_with_yad_type"
        ),
        (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_kd_stn_5min")).alias(
            "diff_raw_mean_total_room_cnt_with_kd_stn_5min"
        ),
        (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_kd_stn_5min")).alias(
            "diff_raw_median_total_room_cnt_with_kd_stn_5min"
        ),
        (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_kd_bch_5min")).alias(
            "diff_raw_mean_total_room_cnt_with_kd_bch_5min"
        ),
        (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_kd_bch_5min")).alias(
            "diff_raw_median_total_room_cnt_with_kd_bch_5min"
        ),
        (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_kd_conv_walk_5min")).alias(
            "diff_raw_mean_total_room_cnt_with_kd_conv_walk_5min"
        ),
        (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_kd_conv_walk_5min")).alias(
            "diff_raw_median_total_room_cnt_with_kd_conv_walk_5min"
        ),
    )

    # ログ内での登場回数
    yad_no_to_appearance_cnt = (
        pl.concat([train_log, test_log])
        .group_by("yad_no")
        .agg(pl.count("yad_no").alias("appearance_cnt"))
        .with_columns(
            pl.col("appearance_cnt"),
        )
    )
    yado = yado.join(yad_no_to_appearance_cnt, on="yad_no", how="left").with_columns(pl.col("yad_no").cast(pl.UInt32))

    train_candidates = train_candidates.join(yado, on="yad_no", how="left").drop([
        "sml_cd_right",
        "lrg_cd_right",
        "ken_cd_right",
        "wid_cd_right",
    ])
    test_candidates = test_candidates.join(yado, on="yad_no", how="left").drop([
        "sml_cd_right",
        "lrg_cd_right",
        "ken_cd_right",
        "wid_cd_right",
    ])

    # sessionを探してる宿の部屋数の統計で特徴づける
    # 探してる宿はある程度似通ってるなら部屋数にも相関があるのでは？
    # TODO: labelの情報を入れるか迷う。
    session2total_romm_cnt_list = (
        pl.concat([train_log, test_log]).with_columns(pl.col("yad_no").cast(pl.UInt32)).join(yado.select(["yad_no", "total_room_cnt"]), on="yad_no", how="left")
        .group_by("session_id").agg(pl.col("total_room_cnt")).with_columns(
            pl.col("total_room_cnt").map_elements(lambda x: np.mean(x.to_numpy())).alias("session_mean_total_room_cnt"),
            pl.col("total_room_cnt").map_elements(lambda x: np.std(x.to_numpy())).alias("session_std_total_room_cnt"),
            pl.col("total_room_cnt").map_elements(lambda x: np.median(x.to_numpy())).alias("session_median_total_room_cnt"),
            pl.col("total_room_cnt").map_elements(lambda x: np.min(x.to_numpy())).alias("session_min_total_room_cnt"),
            pl.col("total_room_cnt").map_elements(lambda x: np.max(x.to_numpy())).alias("session_max_total_room_cnt"),
        )
        .drop("total_room_cnt")
    )
    train_candidates = train_candidates.join(session2total_romm_cnt_list, on="session_id", how="left").with_columns(
        (pl.col("total_room_cnt") - pl.col("session_mean_total_room_cnt")).alias("diff_total_room_cnt_with_session_mean_room_cnt"),
        (pl.col("total_room_cnt") - pl.col("session_median_total_room_cnt")).alias("diff_total_room_cnt_with_session_median_room_cnt"),
        (pl.col("total_room_cnt") - pl.col("session_min_total_room_cnt")).alias("diff_total_room_cnt_with_session_min_room_cnt"),
        (pl.col("total_room_cnt") - pl.col("session_max_total_room_cnt")).alias("diff_total_room_cnt_with_session_max_room_cnt"),
    )
    test_candidates = test_candidates.join(session2total_romm_cnt_list, on="session_id", how="left").with_columns(
        (pl.col("total_room_cnt") - pl.col("session_mean_total_room_cnt")).alias("diff_total_room_cnt_with_session_mean_room_cnt"),
        (pl.col("total_room_cnt") - pl.col("session_median_total_room_cnt")).alias("diff_total_room_cnt_with_session_median_room_cnt"),
        (pl.col("total_room_cnt") - pl.col("session_min_total_room_cnt")).alias("diff_total_room_cnt_with_session_min_room_cnt"),
        (pl.col("total_room_cnt") - pl.col("session_max_total_room_cnt")).alias("diff_total_room_cnt_with_session_max_room_cnt"),
    )

    for seq in range(8):
        seq_yad_no = (
            train_log.filter(pl.col("seq_no") == seq)
            .select(["session_id", "yad_no"])
            .rename({"yad_no": f"seq_{seq}_yad_no"})
        )
        train_candidates = train_candidates.join(seq_yad_no, how="left", on="session_id")

        seq_yad_no = (
            test_log.filter(pl.col("seq_no") == seq)
            .select(["session_id", "yad_no"])
            .rename({"yad_no": f"seq_{seq}_yad_no"})
        )
        test_candidates = test_candidates.join(seq_yad_no, how="left", on="session_id")

    # 宿ごとに何回目に見られることが多いかの統計値, 素の値との差分
    train_yad2seq_no = (
        train_log.group_by(["yad_no"])
        .agg(pl.col("seq_no"))
        .with_columns(
            pl.col("seq_no").map_elements(lambda x: np.mean(x.to_numpy())).alias("mean_seq_no"),
            pl.col("seq_no").map_elements(lambda x: np.std(x.to_numpy())).alias("std_seq_no"),
            pl.col("seq_no").map_elements(lambda x: np.median(x.to_numpy())).alias("median_seq_no"),
        )
        .drop("seq_no")
        .with_columns(pl.col("yad_no").cast(pl.UInt32))
    )
    train_candidates = train_candidates.join(train_yad2seq_no, on="yad_no", how="left").with_columns(
        (pl.col("max_seq_no") - pl.col("mean_seq_no")).alias("diff_max_mean_seq_no"),
        (pl.col("max_seq_no") - pl.col("median_seq_no")).alias("diff_max_median_seq_no"),
    )

    test_yad2seq_no = (
        pl.concat([train_log, test_log])
        .group_by(["yad_no"])
        .agg(pl.col("seq_no"))
        .with_columns(
            # pl.col("seq_no").map_elements(lambda x: print(x)),
            pl.col("seq_no").map_elements(lambda x: np.mean(x.to_numpy())).alias("mean_seq_no"),
            pl.col("seq_no").map_elements(lambda x: np.std(x.to_numpy())).alias("std_seq_no"),
            pl.col("seq_no").map_elements(lambda x: np.median(x.to_numpy())).alias("median_seq_no"),
        )
        .drop("seq_no")
        .with_columns(pl.col("yad_no").cast(pl.UInt32))
    )
    test_candidates = test_candidates.join(test_yad2seq_no, on="yad_no", how="left").with_columns(
        (pl.col("max_seq_no") - pl.col("mean_seq_no")).alias("diff_max_mean_seq_no"),
        (pl.col("max_seq_no") - pl.col("median_seq_no")).alias("diff_max_median_seq_no"),
    )

    # 写真のデータでyadoを特徴づける
    yado2img_features = (
        dfs.image_embeddings_df
        .group_by("yad_no")
        .agg("category")
        .with_columns(
            pl.col("category").list.unique().alias("unique_category"),
        ).with_columns(
            pl.col("category").list.len().alias("num_image"),
            pl.col("unique_category").list.len().alias("num_unique_category"),
        ).join(
            dfs.image_embeddings_df
            .group_by(["yad_no", "category"])
            .count()
            .pivot(index="yad_no", columns="category", values="count"),
            how="left",
            on="yad_no"
        )
        .sort("yad_no")
        .drop(["category", "unique_category"])
        .fill_null(0)
        .with_columns(pl.col("yad_no").cast(pl.UInt32))
    )
    train_candidates = train_candidates.join(yado2img_features, on="yad_no", how="left")
    test_candidates = test_candidates.join(yado2img_features, on="yad_no", how="left")

    print(train_candidates)
    pprint.pprint(train_candidates.columns)

    train_candidates.write_parquet(constants.OUTPUT_DIR / "train_candidates.parquet")
    print(test_candidates)
    test_candidates.write_parquet(constants.OUTPUT_DIR / "test_candidates.parquet")


if __name__ == "__main__":
    main()
