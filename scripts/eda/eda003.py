import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import axes, figure

from src import constants

output_dir = constants.OUTPUT_DIR / "eda003"
output_dir.mkdir(parents=True, exist_ok=True)

###########################################################
# check yado
yado_df = pl.read_csv(constants.INPUT_DIR / "yado.csv")
print(yado_df)

print("ROOM_MAX_CNT: ", yado_df["total_room_cnt"].max())
print("WIRELESS_LAN_FLG: ", yado_df["wireless_lan_flg"].max(), yado_df["wireless_lan_flg"].min())

# 無線LANがある・ないでの部屋数の分布を可視化する
room_cnt_df_with_wifi = yado_df.filter(pl.col("wireless_lan_flg") == 1)
room_cnt_df_without_wifi = yado_df.filter(pl.col("wireless_lan_flg") == 0)
print("ROOM_CNT_WITH_WIFI: ", room_cnt_df_with_wifi)
print("ROOM_CNT_WITHOUT_WIFI: ", room_cnt_df_without_wifi)


def plot_hist_with_text(data: np.ndarray, label: str, color: str, ax: axes.Axes, with_text: bool = False) -> None:
    hist, bins, _ = ax.hist(
        data,
        label=label,
        color=color,
        alpha=0.5,
        bins=100,
    )
    assert isinstance(hist, np.ndarray)
    if with_text:
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


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
assert isinstance(ax, axes.Axes)
assert isinstance(fig, figure.Figure)
plot_hist_with_text(room_cnt_df_without_wifi["total_room_cnt"].to_numpy(), "without_wifi", "orange", ax)
plot_hist_with_text(room_cnt_df_with_wifi["total_room_cnt"].to_numpy(), "with_wifi", "blue", ax)
ax.set_title("Distribution of the number of rooms with/without wireless_lan_flg")
ax.set_xlabel("number of rooms")
ax.grid()
fig.tight_layout()
fig.savefig(str(output_dir / "n_room_with_wireless.png"))


# 温泉がある・ないでの部屋数の分布を可視化する
room_cnt_df_with_onsen = yado_df.filter(pl.col("onsen_flg") == 1)
room_cnt_df_without_onsen = yado_df.filter(pl.col("onsen_flg") == 0)

print("ROOM_CNT_WITH_ONSEN: ", room_cnt_df_with_onsen)
print("ROOM_CNT_WITHOUT_ONSEN: ", room_cnt_df_without_onsen)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_hist_with_text(room_cnt_df_without_onsen["total_room_cnt"].to_numpy(), "without_onsen", "orange", ax)
plot_hist_with_text(room_cnt_df_with_onsen["total_room_cnt"].to_numpy(), "with_onsen", "blue", ax)
ax.set_title("Distribution of the number of rooms")
ax.set_xlabel("number of rooms")
ax.grid()
fig.legend()
fig.tight_layout()
fig.savefig(str(output_dir / "n_room_with_onsen.png"))
