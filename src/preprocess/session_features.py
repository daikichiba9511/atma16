import polars as pl


def make_session_featuers(phase: str, log_df: pl.DataFrame, session_ids: list[str]) -> pl.DataFrame:
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
    print("LOG_DF: ", log_df)

    filtered_log_df = log_df.filter(pl.col("session_id").is_in(session_ids))

    # 各sessionのlogの長さ
    session_length_df = filtered_log_df.group_by("session_id").agg(pl.count("session_id").alias("session_length"))

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
        pl.DataFrame({"session_id": session_ids})
        .join(session_length_df, on="session_id", how="left")
        # .join(last_seen_yad_no_df, on="session_id", how="left")
    )
    return session_df


def _test_make_session_features():
    from src import constants
    from src.utils.common import trace

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    log_df = pl.concat([train_log_df, test_log_df], how="vertical")
    session_ids = train_log_df["session_id"].unique().to_list()
    with trace("making session features..."):
        session_features_df = make_session_featuers("train", log_df, session_ids)
        print("Session Features DataFrame: ", session_features_df)
        print("Unique Session ids: ", session_features_df["session_id"].unique())


if __name__ == "__main__":
    _test_make_session_features()
