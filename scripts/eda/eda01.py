import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import axes, figure

from src import constants

output_dir = constants.OUTPUT_DIR / "eda"
train_log_df = pl.read_csv(constants.INPUT_DIR / "train_log.csv")

output_dir.mkdir(parents=True, exist_ok=True)

# following colums
# * session_id
# * seq_no
# * yad_no

print(train_log_df.head())
print("CNT: ", train_log_df.group_by("session_id").count())
print("CNT_MIN: ", train_log_df.group_by("session_id").count().min())  # 000007603d533d30453cc45d0f3d119f
print("CNT_MAX: ", train_log_df.group_by("session_id").count().max())  # fffffa7baf370083ebcdd98f26a7e31a

max_n_log_series = train_log_df.filter(pl.col("session_id").is_in(["fffffa7baf370083ebcdd98f26a7e31a"]))
print("MAX_N_LOG_SERIES: ", max_n_log_series)

min_n_log_series = train_log_df.filter(pl.col("session_id").is_in(["000007603d533d30453cc45d0f3d119f"]))
print("MIN_N_LOG_SEIRIES: ", min_n_log_series)


# 各session_idのlogの登場回数の分布を可視化する
cnt_df = train_log_df.group_by("session_id").count()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
assert isinstance(ax, axes.Axes)
assert isinstance(fig, figure.Figure)

hist, bins, _ = ax.hist(cnt_df.select("count").to_numpy(), bins=100)
assert isinstance(hist, np.ndarray)

# 各binの値を表示する
for i in range(len(bins) - 1):
    if int(hist[i]) == 0:
        continue
    ax.text(
        bins[i] + (bins[i + 1] - bins[i]) / 2,
        hist[i],
        f"{int(hist[i])}",
        va="bottom",
        ha="center",
        fontsize=8,
        color="black",
    )

ax.set_title("Distribution of the number of logs per session_id")
ax.grid()
fig.tight_layout()
fig.savefig(str(output_dir / "n_log_per_session_id.png"))
