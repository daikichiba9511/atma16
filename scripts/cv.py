import argparse

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from gensim.models import word2vec

from src import constants
from src.config.common import load_config
from src.inference.predict import make_submission, predict
from src.training import metrics
from src.training.common import load_dataframes
from src.utils import common as utils_common
from src.utils.logger import get_root_logger

logger = get_root_logger()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp000")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--remake", action="store_true")
    return parser.parse_args()


def main():
    args = parse()
    cfg = load_config(args.config)
    folded_df_cache_path = cfg.input_dir / f"folded{cfg.n_splits}_df.parquet"
    valid_df = pl.read_parquet(folded_df_cache_path).filter(pl.col("fold") == args.fold)
    session_ids = valid_df["session_id"].unique().to_list()
    if args.debug:
        session_ids = session_ids[:10]

    dfs = load_dataframes()
    model = xgb.Booster()

    wpath = constants.OUTPUT_DIR / args.config / "xgb_model_fold0.ubj"
    print(wpath)
    if wpath.exists():
        print("LOAD MODEL FROM: ", wpath)
        model.load_model(str(wpath))
    models = [model]

    encoders = {
        "wid_cd": joblib.load(constants.OUTPUT_DIR / cfg.name / "wid_cd_encoder.pkl"),
        "ken_cd": joblib.load(constants.OUTPUT_DIR / cfg.name / "ken_cd_encoder.pkl"),
        "lrg_cd": joblib.load(constants.OUTPUT_DIR / cfg.name / "lrg_cd_encoder.pkl"),
        "sml_cd": joblib.load(constants.OUTPUT_DIR / cfg.name / "sml_cd_encoder.pkl"),
        "word2vec": word2vec.Word2Vec.load(str(constants.OUTPUT_DIR / cfg.name / "word2vec")),
    }
    covisit_matrix = np.load(constants.OUTPUT_DIR / "covisit" / "covisit_matrix.npy")
    with utils_common.trace("predicting..."):
        preds = predict(
            models,
            session_ids,
            dfs,
            covisit_matrix=covisit_matrix,
            encoders=encoders,
            phase="train",
            # phase="test",
        )
        sub = make_submission(preds)

    label = pl.DataFrame({"session_id": session_ids}).join(
        dfs.train_label_df.filter(pl.col("session_id").is_in(session_ids)), how="left", on="session_id"
    )
    sub = (
        pl.DataFrame({"session_id": session_ids})
        .join(sub, how="left", on="session_id")
        .select(["session_id", "yad_no"])
    )

    map10 = metrics.mean_average_precision_at_k(label["yad_no"].to_list(), sub["yad_no"].to_list(), k=10)
    print("MAP@10: ", map10)

    oof_cv_df = label.join(
        sub.explode("yad_no").with_columns(pl.col("yad_no").alias("predict")).select(["session_id", "predict"]),
        how="left",
        on="session_id",
    )
    print(oof_cv_df)
    print("rate of label in predict: ", oof_cv_df.filter(pl.col("yad_no") == pl.col("predict"))["session_id"].unique().shape[0] / oof_cv_df["session_id"].unique().shape[0])

    oof_cv_df.write_csv(constants.OUTPUT_DIR / args.config / f"cv_fold{args.fold}.csv")


if __name__ == "__main__":
    main()
