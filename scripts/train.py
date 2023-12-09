import argparse
import pprint
from logging import INFO

import polars as pl

from src.config.common import load_config
from src.preprocess import dataset
from src.training import xgb
from src.training.common import load_dataframes
from src.training.make_fold import make_fold
from src.utils import common as utils_common
from src.utils.logger import attach_file_handler, get_root_logger

logger = get_root_logger(INFO)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp000")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse()
    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    attach_file_handler(logger, str(cfg.output_dir / "train.log"), INFO)
    utils_common.seed_everything(cfg.seed)

    logger.info(f"Training start {cfg.name}")
    logger.info(f"config: {pprint.pformat(cfg)}")

    def _make_folded_df():
        # リソースの開放のために関数にしている
        dfs = load_dataframes()
        if args.debug:
            dfs = dfs.sample(n=1000)
        df = dataset.make_dataset(
            phase="train",
            yad_df=dfs.yad_df,
            train_log_df=dfs.train_log_df,
            test_log_df=dfs.test_log_df,
            train_label_df=dfs.train_label_df,
            session_ids=dfs.train_label_df["session_id"].unique().to_list(),
        )
        folded_df = make_fold(df, n_splits=cfg.n_splits)
        return pl.DataFrame._from_pandas(folded_df)

    df = _make_folded_df()
    for fold in range(cfg.n_splits):
        xgb.train_one_fold(
            cfg=cfg, fold=fold, train_df=df.filter(pl.col("fold") != fold), valid_df=df.filter(pl.col("fold") == fold)
        )
        break


if __name__ == "__main__":
    main()
