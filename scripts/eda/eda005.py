import polars as pl

from src import constants

pl.Config.set_tbl_cols(100)

output_dir = constants.OUTPUT_DIR / __file__.split("/")[-1].split(".")[0]
output_dir.mkdir(parents=True, exist_ok=True)

##############################################################
# check some session_id log and label

# train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
# train_label_df = pl.read_csv(constants.INPUT_DIR / "train_label.csv")
yad_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")

# check null
# print("train_log_df: ", train_log_df)
# print("train_label_df: ", train_label_df)
print("yad_df: ", yad_df)
print("yad_df null cnt: ", yad_df.describe())
# nullのあるカラム
# * total_room_cnt
# * wireless_lan_flg
# * kd_stn_5min
# * kd_bch_5min
# * kd_slp_5min
# * kd_conv_walk_5min
