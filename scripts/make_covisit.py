import dataclasses
import pathlib

import numpy as np
import polars as pl

from src import constants
from src.preprocess import covisit


@dataclasses.dataclass
class Config:
    input_dir: pathlib.Path = constants.INPUT_DIR
    output_dir: pathlib.Path = constants.OUTPUT_DIR / "covisit"


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_log_df = pl.read_csv(cfg.input_dir / "train_log.csv")
    test_log_df = pl.read_csv(cfg.input_dir / "test_log.csv")

    log_df = pl.concat([train_log_df, test_log_df], how="vertical")
    print("train_log_df", train_log_df)
    print("test_log_df", test_log_df)
    print("log_df", log_df)

    covisit_matrix = covisit.make_covisit_matrix(log_df)
    # print("COVISIT_MATRIX: ", covisit_matrix)
    np.save(str(cfg.output_dir / "covisit_matrix.npy"), covisit_matrix)


if __name__ == "__main__":
    main()
