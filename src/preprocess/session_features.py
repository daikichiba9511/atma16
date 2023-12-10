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
    # 興味があってみたyad_noを集める
    # print("Yad_df: ", yad_df)

    # | yad_no | sml_cd |
    # | ------ | ------ |
    # | 1      | [1,2,3]|
    #
    # dict[yad_no, list[sml_cd]]
    # yado_to_sml_cd = (
    #     yad_df.select(["yad_no", "sml_cd"]).group_by("yad_no").agg(pl.col("sml_cd")).to_dict(as_series=False)
    # )
    # yado_to_sml_cd = {
    #     int(k): list(set(v)) for k, v in zip(yado_to_sml_cd["yad_no"], yado_to_sml_cd["sml_cd"]) if v is not None
    # }
    # yado_to_lrg_cd = (
    #     yad_df.select(["yad_no", "lrg_cd"]).group_by("yad_no").agg(pl.col("lrg_cd")).to_dict(as_series=False)
    # )
    # yado_to_lrg_cd = {
    #     int(k): list(set(v)) for k, v in zip(yado_to_lrg_cd["yad_no"], yado_to_lrg_cd["lrg_cd"]) if v is not None
    # }

    # session_cds_df = filtered_log_df.with_columns(
    #     pl.col("yad_no").cast(pl.Int64),
    # ).with_columns(
    #     pl.col("yad_no").replace(yado_to_sml_cd).alias("session_interested_sml_cd"),
    #     pl.col("yad_no").replace(yado_to_lrg_cd).alias("session_interested_lrg_cd"),
    # )

    # print("session_cds_df: ", session_cds_df)

    # raise NotImplementedError("TODO: ここから先を実装する")

    # session_to_cds = session_cds_df.group_by("session_id").agg(
    #     pl.col("yad_no").alias("yad_no"),
    #     pl.col("session_interested_sml_cd").alias("session_interested_sml_cd"),
    #     pl.col("session_interested_lrg_cd").alias("session_interested_lrg_cd"),
    # )
    # print("session_to_cds: ", session_to_cds)

    # raise NotImplementedError("TODO: ここから先を実装する")

    # 最後に見たyad_no
    # TODO: 単純にいれるよりはたぶん埋め込みつくったほうがいい
    # 単純に埋め込むと次元数が膨れ上がる
    # last_seen_yad_no_df = (
    #     filtered_log_df.sort(by="seq_no", descending=True)
    #     .group_by("session_id")
    #     .agg(pl.first("yad_no").alias("last_seen_yad_no"))
    # )

    # attach session features
    session_df = (
        pl.DataFrame({"session_id": session_ids}).join(session_length_df, on="session_id", how="left")
        # .join(last_seen_yad_no_df, on="session_id", how="left")
    )
    return session_df


def _test_make_session_features():
    from src import constants
    from src.utils.common import trace

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    log_df = pl.concat([train_log_df, test_log_df], how="vertical")
    yad_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")
    session_ids = train_log_df["session_id"].unique().to_list()
    with trace("making session features..."):
        session_features_df = make_session_featuers("train", log_df, session_ids, yad_df)
        print("Session Features DataFrame: ", session_features_df)
        print("Unique Session ids: ", session_features_df["session_id"].unique())


if __name__ == "__main__":
    _test_make_session_features()
