import numpy as np
import polars as pl


def make_topk_selected_yad_no(train_label_df: pl.DataFrame, k: int) -> np.ndarray:
    """make topk selected yad_no for a session_id

    実際に選ばれたyad_noの上位k個を返す. これらは選ばれたということなので, これらを候補として返す.

    Args:
        train_lebel_df:
    """
    top20_selected_yad_no = (
        train_label_df["yad_no"].value_counts().sort(by="counts", descending=True).head(k).select("yad_no").to_numpy()
    )
    return top20_selected_yad_no


def make_topk_seen_yad_no(log_df: pl.DataFrame, k: int) -> np.ndarray:
    """make topk seen yad_no for a session_id

    よく見られたyad_noの上位k個を返す. よく見られるということは選ばれる可能性が高いということなので, これらを候補として返す.

    Args:
        log_df: conatenated train_log_df and test_log_df
    """
    top20_seen_yad_no = (
        log_df["yad_no"].value_counts().sort(by="counts", descending=True).head(k).select("yad_no").to_numpy()
    )
    return top20_seen_yad_no


def make_popular_candidates(log_df: pl.DataFrame, train_label_df: pl.DataFrame, k: int) -> np.ndarray:
    """make popular candidates of yad_no for a session_id

    Args:
        log_df: dataframe of concatenated train_log.csv and test_log.csv
        train_lebel_df:

    Returns:
        candidates: popular candidates of yad_no for a session_id
    """
    topk_selected_yad_no = make_topk_selected_yad_no(train_label_df, k=k)
    topk_seen_yad_no = make_topk_seen_yad_no(log_df, k=k)
    candidates = np.concatenate([topk_selected_yad_no, topk_seen_yad_no], dtype=np.int64)
    return candidates.reshape(-1)


def make_popular_candidates_at_seq_0(log_df: pl.DataFrame, k: int) -> np.ndarray:
    """make popular candidates of yad_no for a session_id

    Args:
        log_df: dataframe of concatenated train_log.csv and test_log.csv
        train_lebel_df:

    Returns:
        candidates: popular candidates of yad_no for a session_id
    """
    df_seq_0 = log_df.filter(pl.col("seq_no") == 0)["yad_no"].value_counts().sort(by="counts", descending=True)
    popular_candidates_seq_0 = df_seq_0.head(k).select("yad_no").to_numpy().reshape(-1)
    return popular_candidates_seq_0



def make_seen_candidates(log_df: pl.DataFrame, session_ids: list[str]) -> pl.DataFrame:
    """make seen candidates of yad_no for a session_id

    すでに見たことのあるyad_noを候補として返す.

    """
    # 最後に見たyad_noは候補から除外しておく
    seen_history_for_a_session_id = (
        log_df.filter(pl.col("session_id").is_in(session_ids))
        .group_by("session_id")
        .agg(pl.col("yad_no"))
        .apply(lambda x: (x[0], x[1][:-1]))  # [1,2,3] => [1, 2] 最後に見たyad_noは除外する
        .with_columns(pl.col("column_0").alias("session_id"), pl.col("column_1").alias("yad_no"))
        .select(["session_id", "yad_no"])
        .sort(by="session_id")
    )
    seen_history_for_a_session_id = seen_history_for_a_session_id.explode("yad_no").drop_nulls()
    return seen_history_for_a_session_id


def make_covisit_candidates(df: pl.DataFrame, covist_matrix: np.ndarray, k: int) -> pl.DataFrame:
    """make covisit candidates of yad_no for a session_id

    共起行列を使ってｋ個候補を生成する.
    """
    df_ = df.clone().select(["session_id", "yad_no"])

    # dict[yad_no, list[covisit_yad_no]]
    yad_covisit_dict = {yad_no: [] for yad_no in range(covist_matrix.shape[0])}
    for i in range(len(covist_matrix)):
        # i番目のyad_noについて, 共起するyad_noを頻度の多い順位にk個取得する
        yad_covisit = np.argsort(covist_matrix[i])[::-1][:k]
        yad_covisit_dict[i].extend(yad_covisit.tolist())

    # 共起するyad_noを候補として返す
    # | yad_no | covisit_yad_no |
    # | ------ | -------------- |
    # | 1      | [2, 3, 4]      |
    covisit_df = pl.DataFrame({
        "yad_no": list(yad_covisit_dict.keys()),
        "covisit_yad_no": list(yad_covisit_dict.values()),
    })

    df = (
        df.join(covisit_df, on="yad_no", how="left")
        .explode("covisit_yad_no")  # listを展開
        .select(["session_id", "covisit_yad_no"])
        .with_columns(pl.col("covisit_yad_no").alias("yad_no"))  # 列名を変更; covisit_yad_no -> yad_no
        .select(["session_id", "yad_no"])
    )
    df = pl.concat([df_, df], how="vertical")
    return df


def _test_make_covisit_candidates():
    from src import constants

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    log_df = pl.concat([train_log_df, test_log_df], how="vertical")
    covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    covisit_matrix = make_covisit_candidates(log_df, covisit_matrix, k=10)
    print(covisit_matrix.shape)
    print(covisit_matrix)


if __name__ == "__main__":
    _test_make_covisit_candidates()
