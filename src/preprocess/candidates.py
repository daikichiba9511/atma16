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
