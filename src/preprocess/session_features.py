import polars as pl


def make_session_featuers(
    phase: str, log_df: pl.DataFrame, session_ids: list[str], yad_df: pl.DataFrame
) -> pl.DataFrame:
    """make session features

    Args:
        log_df: log dataframe
                * session_id: セッションID
                * seq_no: ログのシーケンス番号. そのセッション内での順番
                * yad_no: 宿のID
        session_ids: session_ids to make features

    Returns:
        session_features_df: session features dataframe
    """
    filtered_log_df = log_df.filter(pl.col("session_id").is_in(session_ids)).with_columns(
        pl.col("yad_no").cast(pl.Int64)
    )

    # 各sessionのlogの長さ
    session_length_df = filtered_log_df.group_by("session_id").agg(pl.count("session_id").alias("session_length"))

    # 興味があった調べたエリアコード系
    # 見たエリアコードの種類数
    session_to_num_unique_area_cd = (
        filtered_log_df.join(yad_df, on="yad_no", how="left")
        .group_by("session_id")
        .agg([
            pl.col("sml_cd"),
            pl.col("lrg_cd"),
            pl.col("ken_cd"),
            pl.col("wid_cd"),
        ])
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
        .select(["session_id", "num_unique_sml_cd", "num_unique_lrg_cd", "num_unique_ken_cd", "num_unique_wid_cd"])
    )

    # 興味があってみたyad_noを集める
    # | yad_no | sml_cd |
    # | ------ | ------ |
    # | 1      | [1,2,3]|
    # dict[yad_no, list[sml_cd]]
    def _made_yado_to_area_cd(yad_df: pl.DataFrame, area_cd: str) -> dict[int, list[int]]:
        yado_to_area_cd = (
            yad_df.select(["yad_no", area_cd]).group_by("yad_no").agg(pl.col(area_cd)).to_dict(as_series=False)
        )
        yado_to_area_cd = {
            int(k): list(v)
            for k, v in zip(yado_to_area_cd["yad_no"], yado_to_area_cd[area_cd])
            if v is not None and len(v) > 0
        }
        return yado_to_area_cd

    yado_to_sml_cd = _made_yado_to_area_cd(yad_df, "sml_cd")
    yado_to_lrg_cd = _made_yado_to_area_cd(yad_df, "lrg_cd")
    yado_to_ken_cd = _made_yado_to_area_cd(yad_df, "ken_cd")
    yado_to_wid_cd = _made_yado_to_area_cd(yad_df, "wid_cd")

    # 最初に見たyad_noとそれが属するsml_cd, lrg_cd, ken_cd
    first_seen_yad_no_df = (
        filtered_log_df.sort(by="seq_no", descending=False)
        .group_by("session_id")
        .agg(pl.first("yad_no").alias("first_seen_yad_no").cast(pl.Int64))
        .with_columns(
            pl.col("first_seen_yad_no").replace(yado_to_sml_cd, default=None).alias("first_seen_sml_cd"),
            pl.col("first_seen_yad_no").replace(yado_to_lrg_cd, default=None).alias("first_seen_lrg_cd"),
            pl.col("first_seen_yad_no").replace(yado_to_ken_cd, default=None).alias("first_seen_ken_cd"),
            pl.col("first_seen_yad_no").replace(yado_to_wid_cd, default=None).alias("first_seen_wid_cd"),
        )
        .with_columns(
            pl.col("first_seen_sml_cd").map_elements(lambda x: x[0]).alias("first_seen_sml_cd_0"),
            pl.col("first_seen_lrg_cd").map_elements(lambda x: x[0]).alias("first_seen_lrg_cd_0"),
            pl.col("first_seen_ken_cd").map_elements(lambda x: x[0]).alias("first_seen_ken_cd_0"),
            pl.col("first_seen_wid_cd").map_elements(lambda x: x[0]).alias("first_seen_wid_cd_0"),
        )
        .drop(["first_seen_sml_cd", "first_seen_lrg_cd", "first_seen_ken_cd", "first_seen_wid_cd"])
    )
    # 最後に見たyad_noのsml_cd, lrg_cd, ken_cd
    last_seen_yad_no_df = (
        filtered_log_df.sort(by="seq_no", descending=True)
        .group_by("session_id")
        .agg(pl.last("yad_no").alias("last_seen_yad_no"))
        .with_columns(
            pl.col("last_seen_yad_no").replace(yado_to_sml_cd, default=None).alias("last_seen_sml_cd"),
            pl.col("last_seen_yad_no").replace(yado_to_lrg_cd, default=None).alias("last_seen_lrg_cd"),
            pl.col("last_seen_yad_no").replace(yado_to_ken_cd, default=None).alias("last_seen_ken_cd"),
            pl.col("last_seen_yad_no").replace(yado_to_wid_cd, default=None).alias("last_seen_wid_cd"),
        )
        .with_columns(
            pl.col("last_seen_sml_cd").map_elements(lambda x: x[0]).alias("last_seen_sml_cd_0"),
            pl.col("last_seen_lrg_cd").map_elements(lambda x: x[0]).alias("last_seen_lrg_cd_0"),
            pl.col("last_seen_ken_cd").map_elements(lambda x: x[0]).alias("last_seen_ken_cd_0"),
            pl.col("last_seen_wid_cd").map_elements(lambda x: x[0]).alias("last_seen_wid_cd_0"),
        )
        .drop(["last_seen_sml_cd", "last_seen_lrg_cd", "last_seen_ken_cd", "last_seen_wid_cd"])
    )

    # attach session features
    session_df = (
        pl.DataFrame({"session_id": session_ids})
        .join(session_length_df, on="session_id", how="left")
        # .join(session_to_interested_cds_df, on="session_id", how="left")
        .join(last_seen_yad_no_df, on="session_id", how="left")
        .join(first_seen_yad_no_df, on="session_id", how="left")
        .join(session_to_num_unique_area_cd, on="session_id", how="left")
    ).drop(["seq_no", "yad_no"])
    return session_df


def _test_make_session_features():
    from sklearn.preprocessing import LabelEncoder

    from src import constants
    from src.utils.common import trace

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    log_df = pl.concat([train_log_df, test_log_df], how="vertical")
    yad_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")

    encoders = {
        "wid_cd": LabelEncoder(),
        "ken_cd": LabelEncoder(),
        "lrg_cd": LabelEncoder(),
        "sml_cd": LabelEncoder(),
    }

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

    session_ids = train_log_df["session_id"].unique().to_list()
    with trace("making session features..."):
        session_features_df = make_session_featuers("train", log_df, session_ids, yad_df)

    print("Session Features DataFrame: ", session_features_df)
    print("Used columns: ", session_features_df.columns)
    print("num cols: ", len(session_features_df.columns))
    print("Unique Session ids: ", session_features_df["session_id"].unique())
    print(session_features_df.filter(pl.col("first_seen_yad_no") != pl.col("last_seen_yad_no")))


if __name__ == "__main__":
    _test_make_session_features()
