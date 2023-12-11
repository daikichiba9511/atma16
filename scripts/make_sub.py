import argparse

import joblib
import numpy as np
import polars as pl
import xgboost as xgb

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
    dfs = load_dataframes()

    session_ids = dfs.test_log_df["session_id"].unique().to_list()
    if args.debug:
        session_ids = session_ids[:10]

    # モデルの初期化と学習済み重みのロード
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
    print(sub)

    test_session = pl.read_csv(constants.INPUT_DIR / "test_session.csv")
    sub = test_session.select("session_id").join(sub.drop("yad_no"), on="session_id", how="left")
    print(sub)
    sub.drop("session_id").write_csv("submission.csv")


if __name__ == "__main__":
    main()
