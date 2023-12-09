import numpy as np
import polars as pl

from src.preprocess import candidates, session_features, yad_features


def make_dataset(
    phase: str,
    yad_df: pl.DataFrame,
    train_log_df: pl.DataFrame,
    test_log_df: pl.DataFrame,
    train_label_df: pl.DataFrame,
    session_ids: list[str],
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
    popular_candidates = candidates.make_popular_candidates(log_df, train_label_df)
    candidates_for_a_session_id = popular_candidates
    print(f"candidates_for_a_session_id: {candidates_for_a_session_id.dtype}, {candidates_for_a_session_id.shape}")

    if any(np.isnan(candidates_for_a_session_id)):
        # test
        raise ValueError("session_ids contains nan")

    session_features_df = session_features.make_session_featuers(phase, log_df, session_ids)
    yad_features_df = yad_features.make_yad_features(yad_df)

    target_df = log_df.select(["session_id", "yad_no"]).with_columns(pl.lit(1).alias("target"))

    # make pairs of (session_id, yad_no)
    df = (
        pl.DataFrame({"session_id": session_ids})
        .with_columns([
            pl.col("session_id"),
            pl.Series("yad_no", [candidates_for_a_session_id for _ in range(len(session_ids))], dtype=pl.List),
        ])
        .explode("yad_no")
    )

    # attach features
    df = df.join(session_features_df, on="session_id", how="left").join(yad_features_df, on="yad_no", how="left")

    # make label
    df = df.join(target_df, on=["session_id", "yad_no"], how="left").with_columns(
        pl.col("target").fill_null(0).alias("target")
    )

    return df


def _test_make_dataframe():
    from src import constants
    from src.utils.common import trace

    pl.Config.set_tbl_cols(100)

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    train_label_df = pl.read_csv(constants.INPUT_DIR / "train_label.csv")
    yad_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    with trace("making dataframe..."):
        train_df = make_dataset(
            phase="train",
            yad_df=yad_df,
            train_log_df=train_log_df,
            test_log_df=test_log_df,
            train_label_df=train_label_df,
            session_ids=train_label_df["session_id"].unique().to_list(),
        )
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