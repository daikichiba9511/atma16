import numpy as np
import polars as pl


def make_top20_selected_yad_no(train_label_df: pl.DataFrame) -> np.ndarray:
    """make top20 selected yad_no for a session_id

    実際に選ばれたyad_noの上位20個を返す. これらは選ばれたということなので, これらを候補として返す.

    Args:
        train_lebel_df:
    """
    top20_selected_yad_no = (
        train_label_df["yad_no"].value_counts().sort(by="counts", descending=True).head(20).select("yad_no").to_numpy()
    )
    return top20_selected_yad_no


def make_top20_seen_yad_no(log_df: pl.DataFrame) -> np.ndarray:
    """make top20 seen yad_no for a session_id

    よく見られたyad_noの上位20個を返す. よく見られるということは選ばれる可能性が高いということなので, これらを候補として返す.

    Args:
        log_df: conatenated train_log_df and test_log_df
    """
    top20_seen_yad_no = (
        log_df["yad_no"].value_counts().sort(by="counts", descending=True).head(20).select("yad_no").to_numpy()
    )
    return top20_seen_yad_no


def make_popular_candidates(log_df: pl.DataFrame, train_label_df: pl.DataFrame) -> np.ndarray:
    """make popular candidates of yad_no for a session_id

    Args:
        log_df: dataframe of concatenated train_log.csv and test_log.csv
        train_lebel_df:

    Returns:
        candidates: popular candidates of yad_no for a session_id
    """
    top20_selected_yad_no = make_top20_selected_yad_no(train_label_df)
    top20_seen_yad_no = make_top20_seen_yad_no(log_df)
    candidates = np.concatenate([top20_selected_yad_no, top20_seen_yad_no], dtype=np.int64)
    return candidates.reshape(-1)


def make_seen_candidates(log_df: pl.DataFrame, session_ids: list[str]) -> pl.DataFrame:
    """make seen candidates of yad_no for a session_id

    すでに見たことのあるyad_noを候補として返す.

    """
    seen_history_for_a_session_id = (
        log_df.filter(pl.col("session_id").is_in(session_ids)).group_by("session_id").agg(pl.col("yad_no"))
    ).explode("yad_no")
    return seen_history_for_a_session_id


def make_covisit_candidates(df: pl.DataFrame, covist_matrix: np.ndarray, k: int) -> pl.DataFrame:
    """make covisit candidates of yad_no for a session_id

    共起行列を使ってｋ個候補を生成する.
    """
    yad_covisit_dict = {yad_no: [] for yad_no in range(covist_matrix.shape[0])}
    for i in range(len(covist_matrix)):
        yad_covisit = np.argsort(covist_matrix[i])[::-1][:k]
        yad_covisit_dict[i].extend(yad_covisit.tolist())
