import dataclasses
import os

import numpy as np
import polars as pl
from sklearn import model_selection

from src import constants
from src.training.common import load_dataframes


@dataclasses.dataclass
class Config:
    seed: int = 42
    n_split: int = 5

    outpput_candidates_dir = constants.OUTPUT_DIR / "candidates"
    outpput_features_dir = constants.OUTPUT_DIR / "features"


cfg = Config()
cfg.outpput_candidates_dir.mkdir(exist_ok=True, parents=True)
cfg.outpput_features_dir.mkdir(exist_ok=True, parents=True)

dfs = load_dataframes()
train_log = dfs.train_log_df
label = dfs.train_label_df
test_log = dfs.test_log_df
yado = dfs.yad_df


kf = model_selection.KFold(n_splits=cfg.n_split, shuffle=True, random_state=cfg.seed)
fold_assignments = np.full(label.height, -1, dtype=np.int32)
for i, (_, valid_idx) in enumerate(kf.split(label.to_numpy())):
    fold_assignments[valid_idx] = i
label = label.with_columns(pl.Series("fold", fold_assignments))


def make_past_view_yado_candidates_and_feats(log: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """アクセスした宿をcandidatesとして作成する。ただし直近の宿は予約してる宿と違うので除く"""
    max_seq_no = log.group_by("session_id").agg(pl.max("seq_no").alias("max_seq_no"))
    log = log.join(max_seq_no, on="session_id", how="left")

    # 最大値に該当する行を除外する, 重複してるのは無駄なのでユニークを取って除外する
    past_yado_candidates = log.filter(pl.col("seq_no") != pl.col("max_seq_no"))
    past_yado_candidates = past_yado_candidates.select(["session_id", "yad_no"]).unique()

    # 特徴量生成
    # 何個前にみたのか、複数回見たときには直近のもののみを取る
    # | session_id | yad_no | max_seq_no | max_seq_no_diff | session_view_count |
    past_yado_features = log.with_columns((pl.col("max_seq_no") - pl.col("seq_no")).alias("max_seq_no_diff"))
    past_yado_features = past_yado_features.filter(pl.col("max_seq_no") != pl.col("seq_no"))
    # 何回みたか
    session_view_count = log.group_by(["session_id", "yad_no"]).count().rename({"count": "sessiog_view_count"})
    past_yado_features = past_yado_features.join(session_view_count, on=["session_id", "yad_no"], how="left").drop(
        "seq_no"
    )
    return past_yado_candidates, past_yado_features


def make_topk_popular_yado_candidates(
    label: pl.DataFrame, phase: str = "train", k: int = 10
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """予約された人気宿をcandidatesとして保存。train/validでリークしないように注意"""
    # labelデータを使うので、学習データはtrain/validで分割して作成する
    topk_yado_candidates = pl.DataFrame()
    popular_yado_features = pl.DataFrame()

    if phase == "train":
        for fold in range(cfg.n_split):
            train_label = label.filter(pl.col("fold") != fold)
            # train_labelの中で予約された回数でソート
            popular_sorted_yado = train_label["yad_no"].value_counts().sort(by="counts", descending=True)

            # candidates作成
            # sortして上からk個取る=人気の高いk個の宿を取る
            topk_yado_candidates_fold = (
                popular_sorted_yado.head(k).with_columns(pl.lit(fold).alias("fold")).select(["yad_no", "fold"])
            )
            topk_yado_candidates = pl.concat([topk_yado_candidates, topk_yado_candidates_fold])

            # 特徴量作成
            popular_yado_feature_fold = popular_sorted_yado.with_columns(pl.lit(fold).alias("fold"))
            # 全体の実際に予約されたyad_noの中で何位か
            popular_yado_feature_fold = popular_yado_feature_fold.with_columns(
                pl.arange(1, len(popular_sorted_yado) + 1).alias("popular_rank")
            )
            popular_yado_features = pl.concat([popular_yado_features, popular_yado_feature_fold])
    else:
        # 候補生成
        popular_sorted_yado = label["yad_no"].value_counts().sort(by="counts", descending=True)
        topk_yado_candidates = popular_sorted_yado.head(k).select(["yad_no"])

        # 簡易特徴量生成
        popular_yado_features = popular_sorted_yado.with_columns(
            pl.arange(1, len(popular_sorted_yado) + 1).alias("popular_rank")
        )

    popular_yado_features = popular_yado_features.rename({"counts": "reservation_counts"})
    return topk_yado_candidates, popular_yado_features


def make_topk_area_popular_yado_candidates(
    label: pl.DataFrame, yado: pl.DataFrame, phase: str = "train", k: int = 10, area: str = "wid_cd"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """エリア単位で予約された人気宿をcandidatesとして作成する。train/validでリークしないように注意"""
    label_yado = label.join(yado, how="left", on="yad_no")

    # labelデータを使うので、学習データはtrain/validで分割して作成する
    topk_yado_area_candidates = pl.DataFrame()
    popular_yado_area_features = pl.DataFrame()
    if phase == "train":
        for fold in range(cfg.n_split):
            train_label = label_yado.filter(pl.col("fold") != fold)
            # | wid_cd | yad_no | count |
            # | ------ | ------ | ----- |
            # | hoge   | 1      | 10    |
            popular_sorted_yado = (
                train_label.group_by([area, "yad_no"]).count().sort(by=[area, "count"], descending=[False, True])
            )

            # 候補生成
            topk_yado_area_candidates_fold = (
                popular_sorted_yado.group_by(area)
                .head(k)
                .with_columns(pl.lit(fold).alias("fold"))
                .select([area, "yad_no", "fold"])
            )
            topk_yado_area_candidates = pl.concat([topk_yado_area_candidates, topk_yado_area_candidates_fold])

            # 特徴量作成
            popular_yado_area_features_fold = popular_sorted_yado.with_columns(pl.lit(fold).alias("fold"))
            popular_yado_area_features_fold = popular_yado_area_features_fold.group_by(area).map_groups(
                lambda group: group.with_columns(
                    pl.col("count").rank(method="dense", descending=True).over(area).alias(f"popular_{area}_rank")
                )
            )
            popular_yado_area_features = pl.concat([popular_yado_area_features, popular_yado_area_features_fold])

    else:
        # 候補生成
        popular_sorted_yado = (
            label_yado.group_by([area, "yad_no"]).count().sort(by=[area, "count"], descending=[False, True])
        )
        topk_yado_area_candidates = popular_sorted_yado.group_by(area).head(k).select([area, "yad_no"])

        # 特徴量作成
        popular_yado_area_features = popular_sorted_yado.group_by(area).map_groups(
            lambda group: group.with_columns(
                pl.col("count").rank(method="dense", descending=True).over(area).alias(f"popular_{area}_rank")
            )
        )

    popular_yado_area_features = popular_yado_area_features.drop("count")

    return topk_yado_area_candidates, popular_yado_area_features


def make_latest_next_booking_topk_candiddates(
    log: pl.DataFrame, label: pl.DataFrame, phase: str = "train", k: int = 10
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """直近見た宿の中で次にどこを予約しやすいか"""

    # | session_id | seq_no | latest_yad_no | yad_no | fold |
    # | ---------- | ------ | ------------- | ------ | ---- |
    # | hoge       | 1      | 1             | 2      | 0    |
    log_latest = train_log.group_by("session_id").tail(1)
    log_latest = log_latest.rename({"yad_no": "latest_yad_no"})
    log_latest = log_latest.join(label, on="session_id", how="left")

    # labelデータを使うので、学習データはtrain/validで分割して作成する
    latest_next_booking_topk_candidates = pl.DataFrame()
    latest_next_booking_topk_features = pl.DataFrame()

    if phase == "train":
        for fold in range(cfg.n_split):
            # | latet_yad_no | yad_no | count |
            # | ------------ | ------ | ----- |
            # | 1            | 2      | 2     |
            train_log_latest = log_latest.filter(pl.col("fold") != fold)
            train_log_latest = (
                train_log_latest.group_by(["latest_yad_no", "yad_no"])
                .count()
                .sort(by=["latest_yad_no", "count"], descending=[False, True])
            )

            # candidatesの作成
            # countでsortしてるので、上からk個取ることで直近見た宿の中で次に予約しやすい宿を取得できる
            # | latest_yad_no | fold | yad_no |
            # | ------------- | ---- | ------ |
            # | 1             | 0    | 2      |
            latest_next_booking_topk_candiddates_fold = (
                train_log_latest.group_by("latest_yad_no")
                .head(k)
                .with_columns(pl.lit(fold).alias("fold"))
                .select(["yad_no", "fold", "latest_yad_no"])
            )
            latest_next_booking_topk_candidates = pl.concat([
                latest_next_booking_topk_candidates,
                latest_next_booking_topk_candiddates_fold,
            ])

            # 特徴量生成
            # | latest_yad_no | yad_no | count | latest_next_booking_rank |
            # | ------------- | ------ | ----- | ------------------------ |
            # | 1             | 2      | 2     | 1                        |
            latest_next_booking_topk_features_fold = train_log_latest.with_columns(pl.lit(fold).alias("fold"))
            latest_next_booking_topk_features_fold = latest_next_booking_topk_features_fold.group_by(
                "latest_yad_no"
            ).map_groups(
                lambda group: group.with_columns(
                    pl.col("count")
                    .rank(method="dense", descending=True)
                    .over("latest_yad_no")
                    .alias("latest_next_booking_rank")
                )
            )
            latest_next_booking_topk_features = pl.concat([
                latest_next_booking_topk_features,
                latest_next_booking_topk_features_fold,
            ])
    else:
        # | latet_yad_no | yad_no | count |
        # | ------------ | ------ | ----- |
        # | 1            | 2      | 2     |
        log_latest = (
            log_latest.group_by(["latest_yad_no", "yad_no"])
            .count()
            .sort(by=["latest_yad_no", "count"], descending=[False, True])
        )

        # 候補生成
        # | latest_yad_no | fold | yad_no |
        # | ------------- | ---- | ------ |
        # | 1             | 0    | 2      |
        latest_next_booking_topk_candidates = (
            log_latest.group_by("latest_yad_no").head(k).select(["yad_no", "latest_yad_no"])
        )
        # 特徴量生成
        # | latest_yad_no | yad_no | count | latest_next_booking_rank |
        # | ------------- | ------ | ----- | ------------------------ |
        # | 1             | 2      | 2     | 1                        |
        latest_next_booking_topk_features = log_latest.group_by("latest_yad_no").map_groups(
            lambda group: group.with_columns(
                pl.col("count")
                .rank(method="dense", descending=True)
                .over("latest_yad_no")
                .alias("latest_next_booking_rank")
            )
        )
    latest_next_booking_topk_features = latest_next_booking_topk_features.drop("count")
    return latest_next_booking_topk_candidates, latest_next_booking_topk_features


def make_co_visit_matrix(df: pl.DataFrame) -> pl.DataFrame:
    df = df.join(df, on="session_id")
    df = df.filter(pl.col("yad_no") != pl.col("yad_no_right"))
    df = df.group_by(["yad_no", "yad_no_right"]).count()
    df = df.rename({"yad_no_right": "candidate_yad_no", "count": "co_visit_count"})
    df = df.select(["yad_no", "candidate_yad_no", "co_visit_count"])
    df = df.sort(by=["co_visit_count", "yad_no"], descending=[True, False])
    return df


def make_next_yad_no_candidates_at_a_seq_no(log: pl.DataFrame, k: int = 10) -> pl.DataFrame:
    log_next_seq = (
        log.with_columns(pl.col("seq_no") + 1)
        .select(["session_id", "seq_no", "yad_no"])
        .rename({"yad_no": "next_yad_no"})
    )
    log_next_seq = log.join(log_next_seq, how="left", on=["session_id", "seq_no"])
    log_next_seq = log_next_seq.filter(pl.col("next_yad_no").is_not_null())
    # | latest_yad_no | yad_no | count |
    log_next_seq_candidates = (
        log_next_seq.group_by(["yad_no", "next_yad_no"])
        .count()
        .sort(by=["yad_no", "count"], descending=[False, True])
        .group_by("yad_no")
        .head(k)
        .rename({"yad_no": "latest_yad_no", "next_yad_no": "yad_no"})
    )
    return log_next_seq_candidates


if __name__ == "__main__":
    # 実行
    train_past_view_yado_candidates, train_past_view_yado_features = make_past_view_yado_candidates_and_feats(
        train_log
    )
    test_past_view_yado_candidates, test_past_view_yado_features = make_past_view_yado_candidates_and_feats(test_log)

    popular_k = 10
    train_topk_popular_yado_candidates, train_topk_popular_yado_features = make_topk_popular_yado_candidates(
        label, phase="train", k=popular_k
    )
    test_topk_popular_yado_candidates, test_topk_popular_yado_features = make_topk_popular_yado_candidates(
        label, phase="test", k=popular_k
    )

    (
        train_top10_wid_popular_yado_candidates,
        train_top10_wid_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="train", k=10, area="wid_cd")

    (
        test_top10_wid_popular_yado_candidates,
        test_top10_wid_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="test", k=10, area="wid_cd")

    (
        train_top10_ken_popular_yado_candidates,
        train_top10_ken_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="train", k=10, area="ken_cd")

    (
        test_top10_ken_popular_yado_candidates,
        test_top10_ken_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="test", k=10, area="ken_cd")

    (
        train_top10_lrg_popular_yado_candidates,
        train_top10_lrg_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="train", k=10, area="lrg_cd")

    (
        test_top10_lrg_popular_yado_candidates,
        test_top10_lrg_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="test", k=10, area="lrg_cd")

    (
        train_top10_sml_popular_yado_candidates,
        train_top10_sml_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="train", k=10, area="sml_cd")

    (
        test_top10_sml_popular_yado_candidates,
        test_top10_sml_popular_yado_features,
    ) = make_topk_area_popular_yado_candidates(label, yado, phase="test", k=10, area="sml_cd")

    (
        train_latest_next_booking_top20_candidates,
        train_latest_next_booking_top20_features,
    ) = make_latest_next_booking_topk_candiddates(train_log, label, phase="train", k=20)

    (
        test_latest_next_booking_top20_candidates,
        test_latest_next_booking_top20_features,
    ) = make_latest_next_booking_topk_candiddates(test_log, label, phase="test", k=20)

    # 共起行列
    train_co_visit_matrix = make_co_visit_matrix(
        pl.concat([train_log.select(["session_id", "yad_no"]), label.select(["session_id", "yad_no"])])
    )
    test_co_visit_matrix = make_co_visit_matrix(
        pl.concat([
            train_log.select(["session_id", "yad_no"]),
            label.select(["session_id", "yad_no"]),
            test_log.select(["session_id", "yad_no"]),
        ])
    )
    train_co_visit_matrix = train_co_visit_matrix.rename({"yad_no": "latest_yad_no", "candidate_yad_no": "yad_no"})
    test_co_visit_matrix = test_co_visit_matrix.rename({"yad_no": "latest_yad_no", "candidate_yad_no": "yad_no"})
    train_co_visit_matrix_topk_candidates = (
        train_co_visit_matrix.sort(["latest_yad_no", "co_visit_count"], descending=[False, True])
        .group_by("latest_yad_no")
        .head(10)
    )
    test_co_visit_matrix_topk_candidates = (
        test_co_visit_matrix.sort(["latest_yad_no", "co_visit_count"], descending=[False, True])
        .group_by("latest_yad_no")
        .head(10)
    )
    # cold start用共起行列(logが一回以下のsessionのみ)
    train_co_visit_matrix_for_cold_start = make_co_visit_matrix(train_log.filter(pl.col("seq_no") <= 1))
    test_co_visit_matrix_for_cold_start = make_co_visit_matrix(
        pl.concat([train_log, test_log]).filter(pl.col("seq_no") <= 1)
    )
    train_co_visit_matrix_topk_candidates_for_cold_start = (
        train_co_visit_matrix_for_cold_start.sort(["latest_yad_no", "co_visit_count"], descending=[False, True])
        .group_by("latest_yad_no")
        .head(10)
    )
    test_co_visit_matrix_topk_candidates_for_cold_start = (
        test_co_visit_matrix_for_cold_start.sort(["latest_yad_no", "co_visit_count"], descending=[False, True])
        .group_by("latest_yad_no")
        .head(10)
    )

    # cold start用共起行列(logが一回未満のsessionのみ)
    train_co_visit_matrix_for_cold_start_less_than_1 = make_co_visit_matrix(train_log.filter(pl.col("seq_no") < 1))
    test_co_visit_matrix_for_cold_start_less_than_1 = make_co_visit_matrix(
        pl.concat([train_log, test_log]).filter(pl.col("seq_no") < 1)
    )
    train_co_visit_matrix_topk_candidates_for_cold_start_less_than_1 = (
        train_co_visit_matrix_for_cold_start.sort(["latest_yad_no", "co_visit_count"], descending=[False, True])
        .group_by("latest_yad_no")
        .head(10)
    )
    test_co_visit_matrix_topk_candidates_for_cold_start_less_than_1 = (
        test_co_visit_matrix_for_cold_start.sort(["latest_yad_no", "co_visit_count"], descending=[False, True])
        .group_by("latest_yad_no")
        .head(10)
    )

    train_log_next_seq_candidates = make_next_yad_no_candidates_at_a_seq_no(train_log, k=10)
    test_log_next_seq_candidates = make_next_yad_no_candidates_at_a_seq_no(test_log, k=10)

    # parquetで保存
    # saveする前にミスを減らすために既存の特徴量/候補ファイルは削除
    for path in cfg.outpput_candidates_dir.glob("*.parquet"):
        os.remove(path)
    for path in cfg.outpput_features_dir.glob("*.parquet"):
        os.remove(path)

    # candidates
    save_candidates: list[tuple[pl.DataFrame, str]] = [
        # 過去に見た宿
        (train_past_view_yado_candidates, "train_past_view_yado_candidates"),
        (test_past_view_yado_candidates, "test_past_view_yado_candidates"),
        # 人気宿
        (train_topk_popular_yado_candidates, f"train_top{popular_k}_popular_yado_candidates"),
        (test_topk_popular_yado_candidates, f"test_top{popular_k}_popular_yado_candidates"),
        # wid_cdごとの人気宿
        (train_top10_wid_popular_yado_candidates, "train_top10_wid_popular_yado_candidates"),
        (test_top10_wid_popular_yado_candidates, "test_top10_wid_popular_yado_candidates"),
        # ken_cdごとの人気宿
        (train_top10_ken_popular_yado_candidates, "train_top10_ken_popular_yado_candidates"),
        (test_top10_ken_popular_yado_candidates, "test_top10_ken_popular_yado_candidates"),
        # lrg_cdごとの人気宿
        (train_top10_lrg_popular_yado_candidates, "train_top10_lrg_popular_yado_candidates"),
        (test_top10_lrg_popular_yado_candidates, "test_top10_lrg_popular_yado_candidates"),
        # sml_cdごとの人気宿
        (train_top10_sml_popular_yado_candidates, "train_top10_sml_popular_yado_candidates"),
        (test_top10_sml_popular_yado_candidates, "test_top10_sml_popular_yado_candidates"),
        # 直近見た宿の中で次に予約しやすい宿
        (train_latest_next_booking_top20_candidates, "train_latest_next_booking_top20_candidates"),
        (test_latest_next_booking_top20_candidates, "test_latest_next_booking_top20_candidates"),
        # 共起行列から候補生成
        (train_co_visit_matrix_topk_candidates, "train_co_visit_matrix_topk_candidates"),
        (test_co_visit_matrix_topk_candidates, "test_co_visit_matrix_topk_candidates"),
        # yad_no単位で次のseq_noでどのyad_noがでやすいか、上位k個
        (train_log_next_seq_candidates, "train_log_next_seq_candidates"),
        (test_log_next_seq_candidates, "test_log_next_seq_candidates"),
        # cold start用共起行列(logが一回以下のsessionのみ)
        (train_co_visit_matrix_for_cold_start, "train_co_visit_matrix_for_cold_start"),
        (test_co_visit_matrix_for_cold_start, "test_co_visit_matrix_for_cold_start"),
        # cold start用共起行列(logが一回未満のsessionのみ)
        (train_co_visit_matrix_for_cold_start_less_than_1, "train_co_visit_matrix_for_cold_start_less_than_1"),
        (test_co_visit_matrix_for_cold_start_less_than_1, "test_co_visit_matrix_for_cold_start_less_than_1"),
    ]
    for df, name in save_candidates:
        df.write_parquet(cfg.outpput_candidates_dir / f"{name}.parquet")
        print(f"save {name} to {cfg.outpput_candidates_dir / f'{name}.parquet'}")

    save_features: list[tuple[pl.DataFrame, str]] = [
        # 過去に見た宿
        (train_past_view_yado_features, "train_past_view_yado_features"),
        (test_past_view_yado_features, "test_past_view_yado_features"),
        # 人気宿
        (train_topk_popular_yado_features, f"train_top{popular_k}_popular_yado_features"),
        (test_topk_popular_yado_features, f"test_top{popular_k}_popular_yado_features"),
        # wid_cdごとの人気宿
        (train_top10_wid_popular_yado_features, "train_top10_wid_popular_yado_features"),
        (test_top10_wid_popular_yado_features, "test_top10_wid_popular_yado_features"),
        # ken_cdごとの人気宿
        (train_top10_ken_popular_yado_features, "train_top10_ken_popular_yado_features"),
        (test_top10_ken_popular_yado_features, "test_top10_ken_popular_yado_features"),
        # lrg_cdごとの人気宿
        (train_top10_lrg_popular_yado_features, "train_top10_lrg_popular_yado_features"),
        (test_top10_lrg_popular_yado_features, "test_top10_lrg_popular_yado_features"),
        # sml_cdごとの人気宿
        (train_top10_sml_popular_yado_features, "train_top10_sml_popular_yado_features"),
        (test_top10_sml_popular_yado_features, "test_top10_sml_popular_yado_features"),
        # 直近見た宿の中で次に予約しやすい宿
        (train_latest_next_booking_top20_features, "train_latest_next_booking_top20_features"),
        (test_latest_next_booking_top20_features, "test_latest_next_booking_top20_features"),
        # 共起行列から特徴量生成
        (train_co_visit_matrix, "train_co_visit_matrix_features"),
        (test_co_visit_matrix, "test_co_visit_matrix_features"),
    ]
    for df, name in save_features:
        df.to_pandas(use_pyarrow_extension_array=True).to_parquet(cfg.outpput_features_dir / f"{name}.parquet")
        print(f"save {name} to {cfg.outpput_features_dir / f'{name}.parquet'}")
