# -*- coding: utf-8 -*-
"""
Visualization and analysis of nocturnal migration detection data.
Explores detection timing, species phenology, accumulation curves,
diversity metrics, and species-specific heatmaps.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import moving_average as MA
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PKL       = "../data/all_detections_dataframe.pkl"
INPUT_SUNSET_CSV = "../data/df_all_season_location.csv"
OUTPUT_DIR      = "../output"

LOCATIONS = ["Hatzeva", "Mitzpe_ramon", "Shita", "Neot_smadar", "Timna", "Har_Yoash"]

REGION_MAP = {
    "Hatzeva":      "north",
    "Mitzpe_ramon": "north",
    "Shita":        "center",
    "Neot_smadar":  "center",
    "Timna":        "south",
    "Har_Yoash":    "south",
}

TOPO_MAP = {
    "Hatzeva":      "valley",
    "Mitzpe_ramon": "mountain",
    "Shita":        "valley",
    "Neot_smadar":  "mountain",
    "Timna":        "valley",
    "Har_Yoash":    "mountain",
}

PALETTE = {"fall": "#1f77b4", "spring": "#ff7f0e"}

# Season date ranges
FALL_DATES   = pd.date_range("2024-09-05", "2024-11-21", freq="D")
SPRING_DATES = pd.date_range("2025-02-21", "2025-05-16", freq="D")
SPRING_START = pd.to_datetime("2025-03-02")  # for accumulation curve

# Night-bin settings
N_BINS_NIGHT = 10

# Heatmap settings
SPECIES_TO_PLOT = "song_thrush_turdus_philomelos"
NUM_Y_BINS      = 6
HEATMAP_CMAP    = "magma"  # options: 'viridis', 'coolwarm', 'YlGnBu'

# Accumulation curve settings
SAMPLING = "all_season"  # "all_season" or "with_data"

# Font sizes
FONT = {
    "title":      14,
    "axis_label": 12,
    "tick":       10,
    "annot":      9,
}

# ============================================================
# LOAD DATA
# ============================================================

df_all = pd.read_pickle(INPUT_PKL)

# ============================================================
# SUMMARY TABLES
# ============================================================

def shannon_from_group(group):
    """Compute Shannon diversity index for a group."""
    counts = group["species_1"].value_counts()
    p = counts / counts.sum()
    return -(p * np.log(p)).sum()


def build_level_summary(df, groupby_cols):
    """Aggregate detections and species richness, then merge Shannon diversity."""
    summary = (
        df.groupby(groupby_cols)
        .agg(
            detections=("species_1", "size"),
            n_species=("species_1", pd.Series.nunique),
        )
        .reset_index()
    )
    diversity = (
        df.groupby(groupby_cols)
        .apply(shannon_from_group)
        .rename("shannon")
        .reset_index()
    )
    return summary.merge(diversity, on=groupby_cols)


night_level    = build_level_summary(df_all, ["date1", "location", "Season", "region", "topography"])
week_level     = build_level_summary(df_all, ["week",  "location", "Season", "region", "topography"])
location_level = build_level_summary(df_all, ["location", "Season", "region", "topography"])

# ============================================================
# SPECIES × SITE SUMMARY TABLES
# ============================================================

site_season_species = (
    df_all
    .groupby(["location", "Season", "species_1"])
    .size()
    .unstack(fill_value=0)
)

site_species = (
    df_all
    .groupby(["location", "species_1"])
    .size()
    .unstack(fill_value=0)
)

# ============================================================
# SUNSET / SUNRISE — NIGHT PROGRESS
# ============================================================

df_ss  = df_all.copy().sort_values(["date", "time"]).reset_index(drop=True)
df_ss_sr = pd.read_csv(INPUT_SUNSET_CSV)

det = pd.to_datetime(df_ss["date"].astype(str) + " " + df_ss["time"].astype(str),        errors="coerce")
ss  = pd.to_datetime(df_ss["date"].astype(str) + " " + df_ss_sr["sunset"].astype(str),   errors="coerce")
sr  = pd.to_datetime(df_ss["date"].astype(str) + " " + df_ss_sr["sunrise"].astype(str),  errors="coerce")

sr  = sr.where(sr > ss,   sr  + pd.Timedelta(days=1))
det = det.where(det >= ss, det + pd.Timedelta(days=1))

dur = sr - ss
df_ss["night_duration_hours"] = dur.dt.total_seconds()
df_ss["night_progress"]       = ((det - ss) / dur).clip(lower=0, upper=1)


def assign_bins(series, n_bins=10):
    edges  = np.linspace(0, 1, n_bins + 1)
    labels = range(1, n_bins + 1)
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True, right=True)


df_ss["night_bin_10"] = assign_bins(df_ss["night_progress"], n_bins=N_BINS_NIGHT)
df_ssr = df_ss  # alias used below

# ============================================================
# Q1 — DETECTION TIMING BY LOCATION AND SEASON
# ============================================================

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df_ssr.dropna(subset=["night_progress"]),
    x="location",
    y="night_progress",
    hue="Season",
    inner="quartile",
    cut=0,
    split=True,
    palette=PALETTE,
    order=LOCATIONS,
)
plt.ylabel("Relative time within night (0=sunset, 1=sunrise)", fontsize=FONT["axis_label"])
plt.title("Distribution of detection timing within night",    fontsize=FONT["title"])
plt.tick_params(labelsize=FONT["tick"])
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/detection_timing_by_location.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# Q8 — SPECIES ACCUMULATION CURVE
# ============================================================

df_acc = df_all[["species_1", "date1", "Season"]].dropna().copy()
df_acc["date1"]  = pd.to_datetime(df_acc["date1"]).dt.normalize()
df_acc["Season"] = df_acc["Season"].str.lower().str.strip()

# trim spring to start from SPRING_START
df_acc = df_acc[
    ~((df_acc["Season"] == "spring") & (df_acc["date1"] < SPRING_START))
].copy()
df_acc = df_acc.drop_duplicates(subset=["Season", "date1", "species_1"])

accum_list = []
for season in sorted(df_acc["Season"].unique()):
    sub = df_acc[df_acc["Season"] == season].copy()

    if SAMPLING == "all_season":
        sub = sub.sort_values("date1")
        sub["sampling_day"] = (sub["date1"] - sub["date1"].min()).dt.days + 1
    else:
        unique_dates = sorted(sub["date1"].unique())
        day_map = {date: i + 1 for i, date in enumerate(unique_dates)}
        sub["sampling_day"] = sub["date1"].map(day_map)

    first_appearance   = sub.groupby("species_1", as_index=False)["sampling_day"].min()
    new_species_per_day = (
        first_appearance.groupby("sampling_day")
        .size()
        .rename("new_species")
        .reset_index()
    )

    full_days = pd.DataFrame({"sampling_day": range(1, sub["sampling_day"].max() + 1)})
    new_species_per_day = full_days.merge(new_species_per_day, on="sampling_day", how="left")
    new_species_per_day["new_species"]        = new_species_per_day["new_species"].fillna(0)
    new_species_per_day["cumulative_species"] = new_species_per_day["new_species"].cumsum()
    new_species_per_day["Season"]             = season
    accum_list.append(new_species_per_day)

accum_df = pd.concat(accum_list, ignore_index=True)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=accum_df,
    x="sampling_day",
    y="cumulative_species",
    hue="Season",
    palette=PALETTE,
    linewidth=2,
)
plt.xlabel("Sampling day in season" if SAMPLING == "all_season" else "Sampling day with data",
           fontsize=FONT["axis_label"])
plt.ylabel("Cumulative number of species", fontsize=FONT["axis_label"])
plt.title("Species Accumulation Curve",    fontsize=FONT["title"])
plt.tick_params(labelsize=FONT["tick"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/species_accumulation_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# RICHNESS ESTIMATORS (Chao2, Jackknife)
# ============================================================

def richness_estimators(df):
    inc = (
        df.assign(present=1)
        .pivot_table(index="date1", columns="species_1", values="present",
                     aggfunc="max", fill_value=0)
    )
    m            = inc.shape[0]
    species_freq = inc.sum(axis=0)
    S_obs = (species_freq > 0).sum()
    Q1    = (species_freq == 1).sum()
    Q2    = (species_freq == 2).sum()

    chao2 = (
        S_obs + ((m - 1) / m) * (Q1**2 / (2 * Q2))
        if Q2 > 0
        else S_obs + ((m - 1) / m) * (Q1 * (Q1 - 1) / 2)
    )
    jack1 = S_obs + Q1 * ((m - 1) / m)
    jack2 = (
        S_obs + Q1 * ((2 * m - 3) / m) - Q2 * ((m - 2)**2 / (m * (m - 1)))
        if m > 1 else np.nan
    )
    return pd.Series({
        "n_sampling_days": m, "S_obs": S_obs,
        "Q1": Q1, "Q2": Q2,
        "Chao2": chao2, "Jackknife1": jack1, "Jackknife2": jack2,
    })


richness_table = df_acc.groupby("Season").apply(richness_estimators).reset_index()
print(richness_table)

# ============================================================
# HEATMAP — SPECIES-SPECIFIC VOCALOGRAM
# ============================================================

bin_labels = np.arange(1, NUM_Y_BINS + 1)

df_heat = df_ssr[["species_1", "date1", "night_bin_10"]].dropna().copy()
df_heat = df_heat[df_heat["species_1"] == SPECIES_TO_PLOT]
df_heat["date1"] = pd.to_datetime(df_heat["date1"]).dt.normalize()

fall_set, spring_set = set(FALL_DATES), set(SPRING_DATES)
df_heat["Season"] = np.where(
    df_heat["date1"].isin(fall_set),   "fall",
    np.where(df_heat["date1"].isin(spring_set), "spring", np.nan)
)
df_heat = df_heat.dropna(subset=["Season"])

edges = np.linspace(0.5, 10.5, NUM_Y_BINS + 1)
df_heat["bin_y"] = pd.cut(
    df_heat["night_bin_10"].astype(int), bins=edges, labels=bin_labels
).astype(int)


def season_grid(dfs, date_index):
    piv = dfs.groupby(["bin_y", "date1"]).size().unstack(fill_value=0)
    return piv.reindex(index=bin_labels, columns=date_index, fill_value=0)


def month_ticks(ax, date_index):
    firsts = [i for i, d in enumerate(date_index) if d.day == 1]
    if not firsts:
        firsts = np.linspace(0, len(date_index) - 1, 4, dtype=int).tolist()
    ax.set_xticks(firsts)
    ax.set_xticklabels(
        [date_index[i].strftime("%b %d") for i in firsts],
        rotation=0, fontsize=FONT["tick"],
    )


fall_piv   = season_grid(df_heat[df_heat["Season"] == "fall"],   FALL_DATES)
spring_piv = season_grid(df_heat[df_heat["Season"] == "spring"], SPRING_DATES)

vmax = int(max(fall_piv.to_numpy().max(), spring_piv.to_numpy().max()))

sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(14, 4), gridspec_kw={"wspace": 0}
)

im1 = ax1.imshow(fall_piv.values,   aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap=HEATMAP_CMAP)
im2 = ax2.imshow(spring_piv.values, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap=HEATMAP_CMAP)

ax1.set_title(f"{SPECIES_TO_PLOT} — Fall", fontsize=FONT["title"])
ax2.set_title("Spring",                    fontsize=FONT["title"])
ax1.set_xlabel("Date",                     fontsize=FONT["axis_label"])
ax2.set_xlabel("Date",                     fontsize=FONT["axis_label"])
ax1.set_ylabel("Night bin (early → late)", fontsize=FONT["axis_label"])

month_ticks(ax1, FALL_DATES)
month_ticks(ax2, SPRING_DATES)

ax1.set_yticks(np.arange(NUM_Y_BINS))
ax1.set_yticklabels(bin_labels.astype(str), fontsize=FONT["tick"])

for ax, piv in ((ax1, fall_piv), (ax2, spring_piv)):
    ax.grid(False)
    ax.set_yticks(np.arange(NUM_Y_BINS + 1) - 0.5, minor=True)
    ax.yaxis.grid(True, which="minor", color="w", lw=0.6, alpha=0.25)
    ax.set_xticks(np.arange(piv.shape[1] + 1) - 0.5, minor=True)
    ax.xaxis.grid(True, which="major", color="w", lw=0.4, alpha=0.15)

# visual axis break between panels
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
d  = 0.015
kw = dict(transform=ax1.transAxes, color="k", clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kw)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
kw.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kw)
ax2.plot((-d, +d), (1 - d, 1 + d), **kw)

fig.subplots_adjust(right=0.90)
plt.tight_layout(rect=[0, 0, 0.90, 0.98])

cax  = fig.add_axes([0.92, 0.15, 0.015, 0.70])
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Detections", fontsize=FONT["axis_label"])

fig.suptitle("Vocalogram by date × night bin", y=0.995, fontsize=FONT["title"])
plt.savefig(f"{OUTPUT_DIR}/vocalogram_{SPECIES_TO_PLOT}.png", dpi=150, bbox_inches="tight")
plt.show()
