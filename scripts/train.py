import argparse
import pprint
from logging import INFO

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

from src import constants
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
    parser.add_argument("--remake", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse()
    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    attach_file_handler(logger, str(cfg.output_dir / "train.log"), INFO)
    utils_common.seed_everything(cfg.seed)

    logger.info(f"Training start {cfg.name}")
    logger.info(f"config: {pprint.pformat(cfg)}")

    encoders = {
        "wid_cd": LabelEncoder(),
        "ken_cd": LabelEncoder(),
        "lrg_cd": LabelEncoder(),
        "sml_cd": LabelEncoder(),
    }

    def _make_folded_df():
        # リソースの開放のために関数にしている
        dfs = load_dataframes()
        covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
        if args.debug:
            dfs = dfs.sample(n=1000)
        df = dataset.make_dataset(
            phase="train",
            yad_df=dfs.yad_df,
            train_log_df=dfs.train_log_df,
            test_log_df=dfs.test_log_df,
            train_label_df=dfs.train_label_df,
            session_ids=dfs.train_label_df["session_id"].unique().to_list(),
            covisit_matrix=covisit_matrix,
            encoders=encoders,
        )
        df = dataset.make_target(df, dfs.train_label_df)
        folded_df = pl.DataFrame._from_pandas(make_fold(df, n_splits=cfg.n_splits))
        return folded_df

    # TODO: ファイル分けたほうがいいかも, もっと良い効率的なdfの作り方ある
    folded_df_cache_path = cfg.input_dir / f"folded{cfg.n_splits}_df.parquet"
    if args.remake:
        logger.info("remake folded_df")
        with utils_common.trace("making folded_df..."):
            df = _make_folded_df()
        df.write_parquet(folded_df_cache_path)
    else:
        df = pl.read_parquet(folded_df_cache_path)

    for fold in range(cfg.n_splits):
        with utils_common.trace(f"training fold: {fold}..."):
            xgb.train_one_fold(
                cfg=cfg,
                fold=fold,
                train_df=df.filter(pl.col("fold") != fold),
                valid_df=df.filter(pl.col("fold") == fold),
                negative_sampling_rate=0.5
            )
        break

    for col_name, encoder in encoders.items():
        joblib.dump(encoder, cfg.output_dir / f"{col_name}_encoder.pkl")

    logger.info(f"Training end {cfg.name}")


if __name__ == "__main__":
    main()
