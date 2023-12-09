import polars as pl

from src import constants

output_dir = constants.OUTPUT_DIR / "eda002"
output_dir.mkdir(parents=True, exist_ok=True)

###########################################################
# check train label
# 各session_idに一つのラベルが付与されてる
train_label_df = pl.read_csv(constants.INPUT_DIR / "train_label.csv")
print(train_label_df)
print("CNT: ", train_label_df.group_by("session_id").count())
print("CNT_MIN: ", train_label_df.group_by("session_id").count().min())  #
print("CNT_MAX: ", train_label_df.group_by("session_id").count().max())  #

print("UNIQ_N_SESSION_ID: ", len(train_label_df.select("session_id").unique()))
