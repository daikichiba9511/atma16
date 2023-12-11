import polars as pl


def make_yad_features(yad_df: pl.DataFrame) -> pl.DataFrame:
    """make features about yad_no

    Args:
        yad_df: yado.csv
                * yad_no: 宿のID
                * yad_type: 宿のタイプ
                * total_room_cnt: 部屋数
                * wireless_lan_flg: 無線LANがあるかないか。ほぼ1
                * onsen_flg: 温泉があるかないか
                * kd_stn_5min: 駅までの徒歩5分圏内かどうか
                * kd_bch_5min: ビーチまでの徒歩5分圏内かどうか
                * kd_slp_5min: スキー場までの徒歩5分圏内かどうか
                * kd_conv_walk_5min: コンビニまでの徒歩5分圏内かどうか
                * wid_cd: 広域エリアコード. 複数の県にまたがる場合がある
                * ken_cd: 県コード
                * lrg_cd: 大エリアコード.
                * karaoke_flg: カラオケがあるかないか
                * sml_cd: 小エリアコード
    """
    _yad_df = yad_df.clone()
    # 欠損値処理
    # 欠損値は欠損値以外で算出した平均値で埋めて、欠損値フラグを作る
    # * total_room_cnt
    # * wireless_lan_flg
    # * kd_stn_5min
    # * kd_bch_5min
    # * kd_slp_5min
    # * kd_conv_walk_5min
    mean_total_room_cnt = _yad_df["total_room_cnt"].mean()
    mean_wireless_lan_flg = _yad_df["wireless_lan_flg"].mean()
    mean_kd_stn_5min = _yad_df["kd_stn_5min"].mean()
    mean_kd_bch_5min = _yad_df["kd_bch_5min"].mean()
    mean_kd_slp_5min = _yad_df["kd_slp_5min"].mean()
    mean_kd_conv_walk_5min = _yad_df["kd_conv_walk_5min"].mean()

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

    _yad_df = _yad_df.with_columns(
        *_fill_null_mean_and_make_null_flg(_yad_df, "total_room_cnt", mean_total_room_cnt),
        *_fill_null_mean_and_make_null_flg(_yad_df, "wireless_lan_flg", mean_wireless_lan_flg),
        *_fill_null_mean_and_make_null_flg(_yad_df, "kd_stn_5min", mean_kd_stn_5min),
        *_fill_null_mean_and_make_null_flg(_yad_df, "kd_bch_5min", mean_kd_bch_5min),
        *_fill_null_mean_and_make_null_flg(_yad_df, "kd_slp_5min", mean_kd_slp_5min),
        *_fill_null_mean_and_make_null_flg(_yad_df, "kd_conv_walk_5min", mean_kd_conv_walk_5min),
    )

    # make stats features
    stats_total_room_cnt_with_onsen_flg = yad_df.group_by("onsen_flg").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_onsen_flg"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_onsen_flg"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_onsen_flg"),
    )
    stats_total_room_cnt_with_yad_type = yad_df.group_by("yad_type").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_yad_type"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_yad_type"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_yad_type"),
    )
    stats_total_room_cnt_with_kd_stn_5min = yad_df.group_by("kd_stn_5min").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_kd_stn_5min"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_kd_stn_5min"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_kd_stn_5min"),
    )
    stats_total_room_cnt_with_bcn_5min = yad_df.group_by("kd_bch_5min").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_kd_bch_5min"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_kd_bch_5min"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_kd_bch_5min"),
    )
    stats_total_roon_cnt_with_conv_walk_5min = yad_df.group_by("kd_conv_walk_5min").agg(
        pl.mean("total_room_cnt").alias("mean_total_room_cnt_with_kd_conv_walk_5min"),
        pl.std("total_room_cnt").alias("std_total_room_cnt_with_kd_conv_walk_5min"),
        pl.median("total_room_cnt").alias("median_total_room_cnt_with_kd_conv_walk_5min"),
    )

    # attach stats features
    _yad_df = (
        _yad_df.join(stats_total_room_cnt_with_onsen_flg, on="onsen_flg", how="left")
        .join(stats_total_room_cnt_with_yad_type, on="yad_type", how="left")
        .join(stats_total_room_cnt_with_bcn_5min, on="kd_bch_5min", how="left")
        .join(stats_total_room_cnt_with_kd_stn_5min, on="kd_stn_5min", how="left")
        .join(stats_total_room_cnt_with_bcn_5min, on="kd_bch_5min", how="left")
        .join(stats_total_roon_cnt_with_conv_walk_5min, on="kd_conv_walk_5min", how="left")
    )

    # make features with stats features
    _yad_df = _yad_df.with_columns(
        (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_onsen_flg")).alias(
            "diff_total_room_cnt_with_onsen_flg"
        ),
        # (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_onsen_flg")).alias(
        #     "diff_raw_mean_total_room_cnt_with_onsen_flg"
        # ),
        # (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_onsen_flg")).alias(
        #     "diff_raw_median_total_room_cnt_with_onsen_flg"
        # ),
        # (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_yad_type")).alias(
        #     "diff_raw_mean_total_room_cnt_with_yad_type"
        # ),
        # (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_yad_type")).alias(
        #     "diff_raw_median_total_room_cnt_with_yad_type"
        # ),
        # (pl.col("total_room_cnt") - pl.col("mean_total_room_cnt_with_kd_stn_5min")).alias(
        #     "diff_raw_mean_total_room_cnt_with_kd_stn_5min"
        # ),
        # (pl.col("total_room_cnt") - pl.col("median_total_room_cnt_with_kd_stn_5min")).alias(
        #     "diff_raw_median_total_room_cnt_with_kd_stn_5min"
        # ),
    )
    return _yad_df


def _test_make_yad_features():
    from src import constants
    from src.utils.common import trace

    yad_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")
    with trace("making yad features..."):
        yad_features_df = make_yad_features(yad_df)
        print(yad_features_df)


if __name__ == "__main__":
    _test_make_yad_features()
