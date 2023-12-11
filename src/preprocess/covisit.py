import numpy as np
import polars as pl
from tqdm.auto import tqdm


def make_covisit_matrix(log_df: pl.DataFrame) -> np.ndarray:
    """make co-visitation matrix

    Args:
        log_df: conatenated train_log_df and test_log_df

    Returns:
        covisit_matrix: co-visitation matrix.
                        this represents how many times each yado_no is visited from another yado_no.
    """
    max_yad_no = log_df["yad_no"].max()
    assert max_yad_no is not None
    covisit_matrix = np.zeros((max_yad_no + 1, max_yad_no + 1), dtype=np.int64)  # type: ignore
    # O(n^2), n=13562≒1.0e4
    for yado_no in tqdm(
        log_df["yad_no"].unique(),
        total=log_df["yad_no"].n_unique(),
        desc="processing to make co-visitation matrix",
    ):
        session_id_see_this_yado = log_df.filter(
            pl.col("session_id").is_in(
                log_df.filter(pl.col("yad_no").is_in([yado_no]))["session_id"].to_numpy().tolist()
            )
        )
        unique_yad_no_see_this_yado = session_id_see_this_yado["yad_no"].unique()
        for covisit_yad_no in unique_yad_no_see_this_yado:
            if yado_no == covisit_yad_no:
                continue
            covisit_matrix[yado_no, covisit_yad_no] += 1
    return covisit_matrix
