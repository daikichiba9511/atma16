import dataclasses
import pathlib

import polars as pl

from src import constants

DEFAULT_TRAIN_LOG_DF_PATH = constants.INPUT_DIR / "train_log.csv"
DEFAULT_TRAIN_LABEL_DF_PATH = constants.INPUT_DIR / "train_label.csv"
DEFAULT_TEST_LOG_DF_PATH = constants.INPUT_DIR / "test_log.csv"
DEFAULT_YAD_DF_PATH = constants.INPUT_DIR / "yado.csv"
DEFAULT_IMAGE_EMBEDDING_PATH = constants.INPUT_DIR / "image_embeddings.parquet"


@dataclasses.dataclass
class DataFrames:
    train_log_df: pl.DataFrame
    """train_log.csv"""
    train_label_df: pl.DataFrame
    """train_label.csv"""
    test_log_df: pl.DataFrame
    """test_log.csv"""
    yad_df: pl.DataFrame
    """yado.csv"""
    image_embeddings_df: pl.DataFrame
    """image_embeddings.parquet"""

    def sample(self, n: int = 10) -> "DataFrames":
        """sample dataframes for debug

        Args:
            n: number of session_ids to sample
        """
        unique_session_ids = self.train_log_df["session_id"].unique().to_list()[:n]
        return DataFrames(
            train_log_df=self.train_log_df.filter(pl.col("session_id").is_in(unique_session_ids)),
            train_label_df=self.train_label_df.filter(pl.col("session_id").is_in(unique_session_ids)),
            test_log_df=self.test_log_df,
            yad_df=self.yad_df,
            image_embeddings_df=self.image_embeddings_df,
        )


def load_dataframes(
    train_log_path: pathlib.Path = DEFAULT_TRAIN_LOG_DF_PATH,
    train_label_path: pathlib.Path = DEFAULT_TRAIN_LABEL_DF_PATH,
    test_log_path: pathlib.Path = DEFAULT_TEST_LOG_DF_PATH,
    yad_df_path: pathlib.Path = DEFAULT_YAD_DF_PATH,
    image_embeddings_path: pathlib.Path = DEFAULT_IMAGE_EMBEDDING_PATH,
) -> DataFrames:
    train_log_df = pl.read_csv(train_log_path)
    train_label_df = pl.read_csv(train_label_path)
    test_log_df = pl.read_csv(test_log_path)
    yad_df = pl.read_csv(yad_df_path)
    image_embeddings_df = pl.read_parquet(image_embeddings_path)
    return DataFrames(
        train_log_df=train_log_df,
        train_label_df=train_label_df,
        test_log_df=test_log_df,
        yad_df=yad_df,
        image_embeddings_df=image_embeddings_df,
    )
