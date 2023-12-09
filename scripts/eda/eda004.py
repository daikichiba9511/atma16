import polars as pl

from src import constants

output_dir = constants.OUTPUT_DIR / __file__.split("/")[-1].split(".")[0]
output_dir.mkdir(parents=True, exist_ok=True)

##############################################################
# check some session_id log and label

train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")
train_label_df = pl.read_csv(constants.INPUT_DIR / "train_label.csv")

# logの数が1とそれ以外
train_once_log_df = train_log_df.filter(
    pl.col("session_id").is_in(
        train_log_df.group_by("session_id").count().filter(pl.col("count") == 1)["session_id"].to_numpy().tolist()
    )
)
train_many_log_df = train_log_df.filter(
    ~pl.col("session_id").is_in(train_once_log_df["session_id"].to_numpy().tolist())
)

print("N_UNIQUE_LABEL: ", len(train_label_df["yad_no"].unique()))

print("train_label_df: ", train_label_df)
print("train_log_df: ", train_log_df)
print("train_once_log_df: ", train_once_log_df)
print("train_many_log_df: ", train_many_log_df)

print(train_once_log_df.group_by("session_id").count().max())

targets = [
    "000007603d533d30453cc45d0f3d119f",
]

for target in targets:
    log_this_target = train_log_df.filter(pl.col("session_id").is_in([target]))
    label_this_target = train_label_df.filter(pl.col("session_id").is_in([target]))

    print("\n###############################################\n")
    print("TARGET: ", target)
    print("LOG: ", log_this_target)
    print("LABEL: ", label_this_target)

    yad_no = label_this_target["yad_no"].to_numpy()[0]
    same_yad_no_log = train_log_df.filter(pl.col("yad_no").is_in([yad_no]))
    print("SAME_YAD_NO_LOG: ", same_yad_no_log)
