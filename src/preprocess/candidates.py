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
    df_ = df.clone().select(["session_id", pl.col("yad_no").cast(pl.Int32).alias("yad_no")])

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
        df.join(
            covisit_df.select(pl.col("yad_no").cast(pl.Int32).alias("yad_no"), "covisit_yad_no"),
            on="yad_no",
            how="left",
        )
        .explode("covisit_yad_no")  # listを展開
        .select(["session_id", "covisit_yad_no"])
        .with_columns(pl.col("covisit_yad_no").alias("yad_no"))  # 列名を変更; covisit_yad_no -> yad_no
        .select(["session_id", pl.col("yad_no").cast(pl.Int32).alias("yad_no")])
    )
    df = pl.concat([df_, df], how="vertical")
    return df


def _make_ranking_in_area(df_with_yad_info: pl.DataFrame, area_col_name: str) -> pl.DataFrame:
    """指定されたエリアコードのランキングを作成する

    Args:
        df_with_yad_info: yado.csvをjoinしたlogのDataFrame
        area_col_name: wid_cd, ken_cd, lrg_cd, sml_cdのいずれか

    Returns:
        ranking: 指定されたエリアコードのランキング
    """
    if area_col_name not in ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]:
        raise ValueError(
            f"area_col_name must be one of ['wid_cd', 'ken_cd', 'lrg_cd', 'sml_cd'], but got {area_col_name}"
        )

    yad_no_freq = df_with_yad_info["yad_no"].value_counts().sort(by="counts", descending=True).to_dict(as_series=False)
    yad_no_freq = {int(k): int(v) for k, v in zip(yad_no_freq["yad_no"], yad_no_freq["counts"])}

    df = df_with_yad_info.lazy().group_by(area_col_name).agg(pl.col("yad_no"))
    # logの登場回数が多い順にソートする
    # | sml_cd | yad_no  |
    # | ------ | ------  |
    # | sml_cd | [1,2,3] |
    df = (
        df.with_columns(
            pl.col("yad_no").map_elements(
                lambda x: sorted(x.unique().to_list(), key=lambda y: yad_no_freq[y], reverse=True)
            )
        )
        .sort(by=area_col_name)
        .collect()
    )
    return df


def _ranking_to_candidates(
    ranking: pl.DataFrame, k: int, df_with_info: pl.DataFrame, session_ids: list[str], area_col_name: str
) -> pl.DataFrame:
    """指定されたエリアコードのランキングから候補を作成する

    Args:
        ranking: 指定されたエリアコードのランキング
        k: 候補の数
        df_with_info: yado.csvをjoinしたlogのDataFrame
        session_ids: session_idのリスト
        area_col_name: wid_cd, ken_cd, lrg_cd, sml_cdのいずれか

    Returns:
        candidates: 候補
    """
    # | sml_cd | yad_no  |
    # | ------ | ------  |
    # | sml_cd | [1,2,3] |
    ranking_top_k = ranking.with_columns(pl.col("yad_no").map_elements(lambda x: x[:k]))

    # area_col_nameのエリアに興味があるsession_idを取得する
    # area_col_nameの値はencodingされたあとの値を想定
    # | session_id | area_col_name  |
    # | ---------- | -------------  |
    # | sample     |              1 |
    # | sample     |              2 |
    # | sample     |              3 |
    session_to_interested_area = (
        df_with_info.filter(pl.col("session_id").is_in(session_ids))
        .group_by("session_id")
        .agg(pl.col(area_col_name).unique().alias(area_col_name))
    ).explode(area_col_name)

    # area_col_nameのエリアに興味があるsession_idに対して、候補を付与する
    session_to_candiates_of_interested_area = (
        (session_to_interested_area.join(ranking_top_k, on=area_col_name, how="left"))
        .select(["session_id", "yad_no"])
        .explode("yad_no")
    )
    return session_to_candiates_of_interested_area


def make_popular_candidates_in_interested_area(
    df: pl.DataFrame, session_ids: list[str], yad_df: pl.DataFrame, k: int
) -> pl.DataFrame:
    """過去の行動履歴から興味のあるsml_cdのエリアで人気のあるyad_noを候補として返す

    Args:
        df: dataframe of concatenated train_log.csv and test_log.csv
        session_ids: session_ids for train or test
        yad_df: yado.csv

    Returns:
        candidates: popular candidates of yad_no for a session_id
    """
    df_with_yad_info = df.join(yad_df, on="yad_no", how="left")

    ranking_in_sml_cd = _make_ranking_in_area(df_with_yad_info, area_col_name="sml_cd")
    ranking_in_lrg_cd = _make_ranking_in_area(df_with_yad_info, area_col_name="lrg_cd")

    candidates_in_the_interested_sml_cd = _ranking_to_candidates(
        ranking_in_sml_cd, k, df_with_yad_info, session_ids, area_col_name="sml_cd"
    )
    candidates_in_the_interested_lrg_cd = _ranking_to_candidates(
        ranking_in_lrg_cd, k, df_with_yad_info, session_ids, area_col_name="lrg_cd"
    )

    candidates_interested_areas = pl.concat(
        [candidates_in_the_interested_sml_cd, candidates_in_the_interested_lrg_cd], how="vertical"
    ).unique(subset=["session_id", "yad_no"])
    return candidates_interested_areas


def _test_make_covisit_candidates():
    from src import constants

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    log_df = pl.concat([train_log_df, test_log_df], how="vertical")
    covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    covisit_matrix = make_covisit_candidates(log_df, covisit_matrix, k=10)
    print(covisit_matrix.shape)
    print(covisit_matrix)


def _test_make_popular_candidates_in_interested_sml_area():
    from sklearn.preprocessing import LabelEncoder

    from src import constants

    train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
    test_log_df = pl.read_csv(constants.INPUT_DIR / "test_log.csv")
    train_label_df = pl.read_csv(constants.INPUT_DIR / "train_label.csv")
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

    df = pl.concat(
        [log_df.clone().select(["session_id", "yad_no"]), train_label_df.clone().select(["session_id", "yad_no"])],
        how="vertical",
    )
    session_ids = test_log_df["session_id"].unique().to_list()
    df = df.filter(pl.col("session_id").is_in(session_ids))
    candidates = make_popular_candidates_in_interested_area(df, session_ids, yad_df, k=10)
    print("Candidates: ", candidates)


if __name__ == "__main__":
    # _test_make_covisit_candidates()
    _test_make_popular_candidates_in_interested_sml_area()
