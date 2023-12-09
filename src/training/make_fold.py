import polars as pl


def make_fold(df: pl.DataFrame) -> pl.DataFrame:
    """make fold for train

    Args:
        df: dataframe with session_id and yad_no

    Returns:
        dataframe with session_id, yad_no and fold
    """
    ...