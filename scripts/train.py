import argparse
import pprint
from logging import INFO

import polars as pl

from src.config.common import load_config
from src.preprocess import dataset
from src.utils.logger import attach_file_handler, get_root_logger

logger = get_root_logger(INFO)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp000")
    return parser.parse_args()


def main() -> None:
    args = parse()
    cfg = load_config(args.config)
    attach_file_handler(logger, cfg.output_dir / "train.log", INFO)

    logger.info(f"Training start {cfg.name}")
    logger.info(f"config: {pprint.pformat(cfg)}")

    train_log_df = pl.read_csv(cfg.input_dir / "train_log.csv")
    train_label_df = pl.read_csv(cfg.input_dir / "train_label.csv")
    test_log_df = pl.read_csv(cfg.input_dir / "test_log.csv")
    yad_df = pl.read_csv(cfg.input_dir / "yado.csv")
    df = dataset.make_dataset(
        phase="train",
        yad_df=yad_df,
        train_log_df=train_log_df,
        test_log_df=test_log_df,
        train_label_df=train_label_df,
        session_ids=train_label_df["session_id"].unique().to_list(),
    )


if __name__ == "__main__":
    main()
