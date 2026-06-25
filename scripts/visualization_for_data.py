# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:08:48 2025

@author: aya

exploring more options with data using chat gpt
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import moving_average as MA
import matplotlib.dates as mdates

palette = {'fall': '#1f77b4',   # blue
           'spring': '#ff7f0e'} # orange 

"""
---------------------------------------------------------------------------------
pre-prossecing of the data - orgenizing the csv and columns for analysis
---------------------------------------------------------------------------------

"""

## read saved dataframe-

df_all = pd.read_pickle('../data/all_detections_dataframe.pkl')


n_files = 6
locations = ["Hatzeva", "Mitzpe_ramon", "Shita", "Neot_smadar" , "Timna", "Har_Yoash"]

region_map = {
    # fill with your site names
    'Hatzeva': 'north',
    'Mitzpe_ramon': 'north',
    'Shita': 'center',
    'Neot_smadar': 'center',
    'Timna': 'south',
    'Har_Yoash': 'south',
}

topo_map = {
    'Hatzeva': 'valley',
    'Mitzpe_ramon': 'mountain',
    'Shita': 'valley',
    'Neot_smadar': 'mountain',
    'Timna': 'valley',
    'Har_Yoash': 'mountain',
}


"""
---------------------------------------------------------------------------------
Summary tables-
Night- level summary

---------------------------------------------------------------------------------

"""

def shannon_from_group(group):
    counts = group['species_1'].value_counts()
    p = counts / counts.sum()
    return -(p * np.log(p)).sum()  # natural log


# NIGHT/WEEK/LOCATION LEVEL - 

night_level = (
    df_all
    .groupby(['date1', 'location', 'Season', 'region', 'topography'])
    .agg(
        detections=('species_1', 'size'),
        n_species=('species_1', pd.Series.nunique),
    )
    .reset_index()
)

week_level = (
    df_all
    .groupby(['week', 'location', 'Season', 'region', 'topography'])
    .agg(
        detections=('species_1', 'size'),
        n_species=('species_1', pd.Series.nunique),
    )
    .reset_index()
)

location_level = (
    df_all
    .groupby(['location', 'Season', 'region', 'topography'])
    .agg(
        detections=('species_1', 'size'),
        n_species=('species_1', pd.Series.nunique),
    )
    .reset_index()
)

# per night - 
#------------------------------------------------
diversity = (
    df_all
    .groupby(['date1', 'location', 'Season', 'region', 'topography'])
    .apply(shannon_from_group)
    .rename('shannon')
    .reset_index()
)

night_level = night_level.merge(
    diversity,
    on=['date1', 'location', 'Season', 'region', 'topography']
)

# per week - 
#------------------------------------------------
diversity = (
    df_all
    .groupby(['week', 'location', 'Season', 'region', 'topography'])
    .apply(shannon_from_group)
    .rename('shannon')
    .reset_index()
)

week_level = week_level.merge(
    diversity,
    on=['week', 'location', 'Season', 'region', 'topography']
) 

# per location - 
#-------------------------------------------------
diversity = (
    df_all
    .groupby(['location', 'Season', 'region', 'topography'])
    .apply(shannon_from_group)
    .rename('shannon')
    .reset_index()
)

location_level = location_level.merge(
    diversity,
    on=['location', 'Season', 'region', 'topography']
)

"""
---------------------------------------------------------------------------------
Summary tables-
Site-Season X species summary

---------------------------------------------------------------------------------

"""

site_season_species = (
    df_all
    .groupby(['location', 'Season', 'species_1'])
    .size()
    .unstack(fill_value=0)  # rows: (location, Season), columns: species
)

site_species = (
    df_all
    .groupby(['location', 'species_1'])
    .size()
    .unstack(fill_value=0)  # rows: (location, Season), columns: species
)

"""
---------------------------------------------------------------------------------
Q1 -  Is time of detection correlated with location (north–south or mountain–valley)?
Is it different between seasons?

---------------------------------------------------------------------------------
"""
# # Hour vs region, split by Season
# plt.figure(figsize=(10, 6))
# sns.violinplot(
#     data=df_all,
#     x='region',
#     y='hour_from_sunset',
#     hue='Season',
#     split=True
# )
# plt.ylabel('Hour of detection')
# plt.title('Detection time by region and season')
# plt.show()


# # Histogram of hours by topography × season
# g = sns.displot(
#     data=df_all,
#     x='hour_from_sunset',
#     col='Season',
#     row='topography',
#     kind='hist',
#     bins=24,
#     facet_kws=dict(margin_titles=True)
# )
# g.set_axis_labels('Hour of detection', 'Count')
# plt.show()

"""
applying sunset time and sunrise time for analysis
------------------------------------------------------
"""
df_ss = df_all.copy()
df_sss = df_ss.sort_values(by=['date', 'time'])
df_ssr = df_sss.reset_index(drop=True)

df_ss_sr = pd.read_csv('../data/df_all_season_location.csv')
det = pd.to_datetime(df_ssr['date'].astype(str) + ' ' + df_ssr['time'].astype(str), errors='coerce')
ss = pd.to_datetime(df_ssr['date'].astype(str) + ' ' + df_ss_sr['sunset'].astype(str), errors='coerce')
sr = pd.to_datetime(df_ssr['date'].astype(str) + ' ' + df_ss_sr['sunrise'].astype(str), errors='coerce')

sr = sr.where(sr > ss, sr + pd.Timedelta(days=1))
det = det.where(det >= ss, det + pd.Timedelta(days=1))


#  Night duration (hours) & relative progress [0,1] ---
dur = (sr - ss)
df_ssr['night_duration_hours'] = dur.dt.total_seconds()

# progress = 0 at sunset, 1 at sunrise
prog = (det - ss) / dur
df_ssr['night_progress'] = prog.clip(lower=0, upper=1)  # keep within [0,1]


# Bin assignment (choose N=10 or 20) ---
def assign_bins(series, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    labels = range(1, n_bins + 1)
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True, right=True)

df_ssr['night_bin_10'] = assign_bins(df_ssr['night_progress'], n_bins=10)

# Hour vs site, split by Season
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df_ssr.dropna(subset=['night_progress']),
    x='location',
    y='night_progress',
    hue='Season',
    inner='quartile',
    cut=0,
    split=True,
    palette=palette ,
    order = ["Hatzeva", "Mitzpe_ramon", "Shita", "Neot_smadar" , "Timna", "Har_Yoash"]
)

plt.ylabel('Relative time within night (0=sunset, 1=sunrise)')
plt.title('Distribution of detection timing within night')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


"""
---------------------------------------------------------------------------------
Q5- Species-specific phenology
---------------------------------------------------------------------------------

"""
merged_df = pd.DataFrame()

# Setting the time fram of the seasons-
#------------------------------------------------------------------
# Define full date range- fall
start_date = pd.to_datetime('2024-09-05')
end_date = pd.to_datetime('2024-11-21')
fall_dates = pd.date_range(start=start_date, end=end_date)

# Define full date range- spring
start_date = pd.to_datetime('2025-02-21')
end_date = pd.to_datetime('2025-05-16')
spring_dates = pd.date_range(start=start_date, end=end_date)

# Combine the fall and spring date ranges
fall_combos = pd.MultiIndex.from_product([fall_dates], names=['date2']).to_frame(index=False)
fall_combos['Season'] = 'fall'
spring_combos = pd.MultiIndex.from_product([spring_dates], names=['date2']).to_frame(index=False)
spring_combos["Season"] = "spring"
all_dates = pd.concat([fall_combos, spring_combos], ignore_index=True)


top_species = df_all['species_1'].value_counts().head(10).index
print(top_species)

sub = df_all[df_all['species_1'].isin(top_species)]

g = sns.displot(
    data=sub,
    x='hour',
    col='species_1',
    row='Season',
    hue='topography',
    kind='kde',
    fill=True
)


species_list = top_species[2:7]

# Count identifications per location, date, and season
group_sub = df_all.groupby(['species_1', 'date1']).size().reset_index(name='count')

sub_df = group_sub[group_sub['species_1'].isin(species_list)]

sub_df['date2'] = pd.to_datetime(sub_df['date1'])
#Merge complete_df with your existing DataFrame - 
merged_df = all_dates.merge(sub_df, on='date2', how='left')
#Fill NaN values in the 'detections' column with 0
merged_df['count'] = merged_df['count'].fillna(0).astype(int)
merged_df['species_1'] = merged_df['species_1'].fillna('european_bee-eater_merops_apiaster').astype(str)
merged_df['date1'] = pd.to_datetime(merged_df['date2'], format = "%Y%m%d").dt.date


# Pivot the data:
pivoted_sub = merged_df.pivot_table(index='date1', columns='species_1', values='count')

# Ensure the index is datetime for comparison
pivoted_sub.index = pd.to_datetime(pivoted_sub.index)
# pivoted_sub['ortolan_bunting_emberiza_hortulana'] = pivoted_sub['ortolan_bunting_emberiza_hortulana'].fillna(0).astype(int)
pivoted_sub['tree_pipit_anthus_trivialis'] = pivoted_sub['tree_pipit_anthus_trivialis'].fillna(0).astype(int)
pivoted_sub['grey_heron_ardea_cinerea'] = pivoted_sub['grey_heron_ardea_cinerea'].fillna(0).astype(int)
pivoted_sub['cretzschmars_bunting_emberiza_caesia'] = pivoted_sub['cretzschmars_bunting_emberiza_caesia'].fillna(0).astype(int)

# Define masks using your existing date ranges
fall_mask = pivoted_sub.index.isin(fall_dates)
spring_mask = pivoted_sub.index.isin(spring_dates)

# ---- Toggles ----
relative = False      # True => per-day composition (rows sum to 1)
weekly   = False      # True => aggregate to weeks before plotting
smooth_w = 5       # e.g., 3 or 7 for rolling mean window (in days or weeks)

plot_df = pivoted_sub.copy()

# Optional weekly aggregation
if weekly:
    # sum counts per week (Mon-based). For relative, we'll normalize after this step.
    plot_df = plot_df.resample("W-MON").sum()

# Relative abundance (row-wise proportions)
if relative:
    plot_df = plot_df.div(plot_df.sum(axis=1), axis=0).fillna(0)

# Optional smoothing (rolling mean)
if smooth_w:
    plot_df = plot_df.rolling(smooth_w, min_periods=1).mean()

# Split into seasons using your masks
fall_df   = plot_df.loc[fall_mask]
spring_df = plot_df.loc[spring_mask]

species_order = list(plot_df.columns)  # preserves your column order

# ---- Figure & two panels ----
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]}
)

# Helper: plot lines for all species on one axis
def plot_lines(ax, df_part, title):
    for sp in species_order:
        ax.plot(df_part.index, df_part[sp].values, label=sp, linewidth=1.8)
    ax.set_xlim(df_part.index.min(), df_part.index.max())
    ax.set_title(title)

# --- Fall ---
plot_lines(ax1, fall_df, "Fall 2024")

# --- Spring ---
plot_lines(ax2, spring_df, "Spring 2025")

# X-axis formatting
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis='x', rotation=0)

# Break styling (like your original)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax2.tick_params(labelright=False)

# Diagonal break marks
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Labels & legend
# fig.suptitle(
#     ("Daily" if not weekly else "Weekly")
#     + (" relative " if relative else " ")
#     + "abundance of species across fall and spring",
#     fontsize=14
# )
fig.text(0.5, 0.04, "Date", ha='center')
fig.text(
    0.04, 0.5,
    ("Relative abundance (proportion)" if relative else "Detections per " + ("week" if weekly else "day")),
    va='center', rotation='vertical'
)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 6), frameon=False)

plt.tight_layout(rect=[0.04, 0.07, 1, 0.90])
plt.show()

"""
Top 10 species relative aboundancee-
---------------------------------------------------
"""

# ---------------- Toggles ----------------
relative = True          # True => stack proportions (each day/week sums to 1)
weekly   = False          # True => aggregate to weeks ("W-MON")

# ---------------- Build daily counts (wide) ----------------

dt = pd.to_datetime(df_all['date1'], errors= 'coerce')
tmp = df_all.assign(dt=dt)
tmp = tmp.dropna(subset=['dt'])

# daily counts per species (wide)
counts_wide = (
    tmp
    .groupby([tmp['dt'].dt.normalize(), 'species_1'])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

# ---------------- Top-N selection (fixed across seasons) ----------------
top_species = counts_wide.sum(axis=0).nlargest(12).index.tolist()
plot_wide   = counts_wide[top_species].copy()

# Optional weekly aggregation
if weekly:
    plot_wide = plot_wide.resample("W-MON").sum()

# Relative abundance (row-wise proportions)
if relative:
    plot_wide = plot_wide.div(plot_wide.sum(axis=1), axis=0).fillna(0)

# Ensure continuous index (optional, nice for bars)
idx_full = pd.date_range(plot_wide.index.min(), plot_wide.index.max(),
                         freq=("W-MON" if weekly else "D"))
plot_wide = plot_wide.reindex(idx_full, fill_value=0)

# ---------------- Season split (set ranges OR reuse masks) ----------------

# Define date ranges:
fall_start   = pd.to_datetime("2024-09-05")
fall_end     = pd.to_datetime("2024-11-21")
spring_start = pd.to_datetime("2025-02-21")
spring_end   = pd.to_datetime("2025-05-16")

fall_df   = plot_wide.loc[(plot_wide.index >= fall_start) & (plot_wide.index <= fall_end)]
spring_df = plot_wide.loc[(plot_wide.index >= spring_start) & (plot_wide.index <= spring_end)]

# ---------------- Colors (12 distinct from tab20) ----------------
import itertools
cmap = plt.cm.get_cmap('tab20')
order = [0,2,4,6,8,10,12,14,16,18,1,3]
color12 = [cmap(i) for i in order]
color_map = dict(zip(top_species, color12))

# ---------------- Plot: two panels with break ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6),
                               gridspec_kw={'width_ratios': [1, 1]})

def stacked_bars(ax, df_part, title):
    bottom = np.zeros(len(df_part), dtype=float)
    for sp in top_species:
        vals = df_part[sp].to_numpy()
        ax.bar(df_part.index, vals, bottom=bottom, width=0.9 if weekly else 0.8,
               edgecolor='none', label=sp, color=color_map[sp])
        bottom += vals
    ax.set_xlim(df_part.index.min(), df_part.index.max())
    ax.set_title(title)

# Panels
stacked_bars(ax1, fall_df,   ("Fall "   + str(fall_df.index.min().year) if len(fall_df) else "Fall"))
stacked_bars(ax2, spring_df, ("Spring " + str(spring_df.index.min().year) if len(spring_df) else "Spring"))

# X-axis formatting
for ax in (ax1, ax2):
    if weekly:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis='x', rotation=0)

# Break styling
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax2.tick_params(labelright=False)

d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Labels & legend
# fig.suptitle(
#     ("Weekly" if weekly else "Daily") +
#     (" relative " if relative else " ") +
#     f"stacked composition – Top 12 species",
#     fontsize=14
# )
fig.text(0.5, 0.04, "Date", ha='center')
fig.text(0.04, 0.5,
         ("Relative abundance (proportion)" if relative else
          ("Detections per week" if weekly else "Detections per day")),
         va='center', rotation='vertical')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles[:len(top_species)], labels[:len(top_species)],
           loc='upper center', ncol=min(len(top_species), 5), frameon=False)

plt.tight_layout(rect=[0.04, 0.07, 1, 0.90])
plt.show()



"""
---------------------------------------------------------------------------------
Q7- Diversity on an altitude scale and a latitude scale
---------------------------------------------------------------------------------

"""

site_covars = pd.read_csv('../data/effort table.csv')

df = location_level.merge(site_covars[['location','Season',
                          'altitude','latitude']], on=['location', 'Season'], 
                          how='left', validate='m:1')

# -----------  scale continuous covariates for stability
df['alt_sc']  = (df['altitude'] - df['altitude'].mean())/df['altitude'].std()
df['lat_sc'] = (df['latitude'] - df['latitude'].mean())/df['latitude'].std()

# ---------- one marker per location (up to 10 here; extend if you have more)
marker_symbols = ['o','P','X','v','D','*']
marker_map = {loc: marker_symbols[i % len(marker_symbols)] for i, loc in enumerate(locations)}
season_order = sorted(df['Season'].astype(str).unique())
palette = sns.color_palette('tab10', n_colors=len(season_order))
season_color = dict(zip(season_order, palette))


# ---- Dot plot: altitude vs shannon ----
sns.set(style='whitegrid')
ax = sns.scatterplot(data=df, x='alt_sc', y='shannon',
                     hue='Season', palette='tab10',
                     style='location', markers=marker_map,
                     s=60, edgecolor='white')
# per-season trend lines (Altitude)
for s in season_order:
    sub = df[df['Season'] == s]
    if len(sub) >= 2:
        sns.regplot(
            data=sub, x='alt_sc', y='shannon',
            scatter=False, ci=None, color=season_color[s],
            line_kws={'alpha': 0.35, 'lw': 2}, ax=ax
        )
ax.set_title('Shannon diversity vs Altitude')
ax.set_xlabel('Altitude')
ax.set_ylabel('Shannon diversity')
ax.legend(title=None , bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---- 3) Dot plot: latitude vs shannon ----
ax = sns.scatterplot(data=df, x='lat_sc', y='shannon',
                     hue='Season', palette='tab10',
                     style='location', markers=marker_map,
                     s=60, edgecolor='white')
# per-season trend lines (Altitude)
for s in season_order:
    sub = df[df['Season'] == s]
    if len(sub) >= 2:
        sns.regplot(
            data=sub, x='lat_sc', y='shannon',
            scatter=False, ci=None, color=season_color[s],
            line_kws={'alpha': 0.35, 'lw': 2}, ax=ax
        )
ax.set_title('Shannon diversity vs Latitude')
ax.set_xlabel('Latitude')
ax.set_ylabel('Shannon diversity')
ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

from scipy.stats import pearsonr

def corr_by_season(df, xcol, ycol='shannon'):
    results = []

    for s in sorted(df['Season'].dropna().unique()):
        sub = df[df['Season'] == s].dropna(subset=[xcol, ycol])

        if len(sub) < 2:
            results.append({'Season': s, 'n': len(sub), 'r': np.nan, 'p_value': np.nan, 'R2': np.nan})
            continue

        r, p = pearsonr(sub[xcol], sub[ycol])

        results.append({
            'Season': s,
            'n': len(sub),
            'r': r,
            'p_value': p,
            'R2': r**2
        })

    return pd.DataFrame(results)

# altitude
alt_stats = corr_by_season(df, 'alt_sc')
print("Altitude vs Shannon")
print(alt_stats)

# latitude
lat_stats = corr_by_season(df, 'lat_sc')
print("\nLatitude vs Shannon")
print(lat_stats)

"""
---------------------------------------------------------------------------------
Q8- Species Accumulation Curve where x axis is the day in the season
---------------------------------------------------------------------------------

"""
sampling = 'all_season' # all_season / with_data

df_acc = df_all[['species_1', 'date1', 'Season']].dropna().copy()

df_acc['date1'] = pd.to_datetime(df_acc['date1']).dt.normalize()
df_acc['Season'] = df_acc['Season'].str.lower().str.strip()

# start spring from 2025-03-02
spring_start = pd.to_datetime('2025-03-02')
df_acc = df_acc[
    ~((df_acc['Season'] == 'spring') & (df_acc['date1'] < spring_start))
].copy()

# keep only one record per species per day per season
df_acc = df_acc.drop_duplicates(subset=['Season', 'date1', 'species_1'])

accum_list = []

for season in sorted(df_acc['Season'].unique()):
    sub = df_acc[df_acc['Season'] == season].copy()
    if sampling == 'all_season':
        sub = sub.sort_values('date1')
        # define day number within season
        start_date = sub['date1'].min()
        sub['sampling_day'] = (sub['date1'] - start_date).dt.days + 1
    else:
        # sort sampled dates
        unique_dates = sorted(sub['date1'].unique())

        # assign sampling-day number: 1, 2, 3, ...
        day_map = {date: i + 1 for i, date in enumerate(unique_dates)}
        sub['sampling_day'] = sub['date1'].map(day_map)

    # find the first day each species appears
    first_appearance = (
        sub.groupby('species_1', as_index=False)['sampling_day']
        .min()
    )

    # count how many new species appear on each day
    new_species_per_day = (
        first_appearance.groupby('sampling_day')
        .size()
        .rename('new_species')
        .reset_index()
    )
    
    # create full day range
    full_days = pd.DataFrame({
        'sampling_day': range(1, sub['sampling_day'].max() + 1)
        })

    # merge and fill missing days with 0
    new_species_per_day = full_days.merge(
        new_species_per_day, on='sampling_day', how='left'
    )
    new_species_per_day['new_species'] = new_species_per_day['new_species'].fillna(0)

    # cumulative richness
    new_species_per_day['cumulative_species'] = new_species_per_day['new_species'].cumsum()
    new_species_per_day['Season'] = season

    accum_list.append(new_species_per_day)

accum_df = pd.concat(accum_list, ignore_index=True)

# plot
sns.set(style='whitegrid')
plt.figure(figsize=(8, 5))

sns.lineplot(
    data=accum_df,
    x='sampling_day',
    y='cumulative_species',
    hue='Season',
    palette={'fall': '#1f77b4', 'spring': '#ff7f0e'},
    linewidth=2
)

if sampling == 'all_season':
    plt.xlabel('Sampling Day in season')
else:
    plt.xlabel('Sampling Day with data')
plt.ylabel('Cumulative number of species')
plt.title('Species Accumulation Curve')
plt.tight_layout()
plt.show()


#---------------------------------------------------------------

def richness_estimators(df):
    # incidence matrix: rows = days, columns = species, values = 0/1
    inc = (
        df.assign(present=1)
        .pivot_table(index='date1', columns='species_1', values='present',
                     aggfunc='max', fill_value=0)
    )

    m = inc.shape[0]                      # number of sampling days
    species_freq = inc.sum(axis=0)        # number of days each species was detected in

    S_obs = (species_freq > 0).sum()
    Q1 = (species_freq == 1).sum()
    Q2 = (species_freq == 2).sum()

    # Chao2
    if Q2 > 0:
        chao2 = S_obs + ((m - 1) / m) * (Q1**2 / (2 * Q2))
    else:
        # bias-corrected version when Q2 = 0
        chao2 = S_obs + ((m - 1) / m) * (Q1 * (Q1 - 1) / 2)

    # Jackknife 1
    jack1 = S_obs + Q1 * ((m - 1) / m)

    # Jackknife 2
    if m > 1:
        jack2 = S_obs + Q1 * ((2 * m - 3) / m) - Q2 * ((m - 2)**2 / (m * (m - 1)))
    else:
        jack2 = np.nan

    return pd.Series({
        'n_sampling_days': m,
        'S_obs': S_obs,
        'Q1': Q1,
        'Q2': Q2,
        'Chao2': chao2,
        'Jackknife1': jack1,
        'Jackknife2': jack2
    })


richness_table = (
    df_acc.groupby('Season')
    .apply(richness_estimators)
    .reset_index()
)

print(richness_table)


"""
Heatmap for species - specific
------------------------------------------------------
"""

# --- choose one species ---
species_to_plot = "song_thrush_turdus_philomelos"
# tree_pipit_anthus_trivialis
# european_bee-eater_merops_apiaster
# spotted_flycatcher_muscicapa_striata
# grey_heron_ardea_cinerea
# western_yellow_wagtail_motacilla_flava
# ortolan_bunting_emberiza_hortulana
# song_thrush_turdus_philomelos
# cretzschmars_bunting_emberiza_caesia

num_y_bins = 6
cmap = 'magma' #'viridis', 'coolwarm' , 'YlGnBu'

# explicit date ranges
fall_dates   = pd.date_range('2024-09-05', '2024-11-21', freq='D')
spring_dates = pd.date_range('2025-02-21', '2025-05-16', freq='D')

# --- prep ---
df = df_ssr[['species_1','date1','night_bin_10']].dropna().copy()   # your df here
df = df[df['species_1'] == species_to_plot]
df['date1'] = pd.to_datetime(df['date1']).dt.normalize()

# tag Season by membership in your ranges
fall_set, spring_set = set(fall_dates), set(spring_dates)
df['Season'] = np.where(df['date1'].isin(fall_set), 'fall',
                 np.where(df['date1'].isin(spring_set), 'spring', np.nan))
df = df.dropna(subset=['Season'])

# down-bin 10 → num_y_bins (rectangular y cells)
edges  = np.linspace(0.5, 10.5, num_y_bins + 1)
labels = np.arange(1, num_y_bins + 1)
df['bin_y'] = pd.cut(df['night_bin_10'].astype(int), bins=edges, labels=labels).astype(int)

def season_grid(dfs, date_index):
    # counts per (bin_y × date), then force your exact date range
    piv = (dfs.groupby(['bin_y','date1']).size().unstack(fill_value=0))
    piv = piv.reindex(index=labels, columns=date_index, fill_value=0)  # <- enforce range
    return piv

fall_piv   = season_grid(df[df['Season']=='fall'],   fall_dates)
spring_piv = season_grid(df[df['Season']=='spring'], spring_dates)

# consistent color scale
vmax = int(max(fall_piv.to_numpy().max(), spring_piv.to_numpy().max()))
vmin = 0

# --- plot (side-by-side with a visual break) ---
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,4), gridspec_kw={'wspace': 0})

im1 = ax1.imshow(fall_piv.values, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
ax1.set_title(f"{species_to_plot} — Fall")
ax1.set_xlabel("Date"); ax1.set_ylabel("Night bin (early → late)")

im2 = ax2.imshow(spring_piv.values, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
ax2.set_title("Spring"); ax2.set_xlabel("Date")

# x ticks at the first of each month within the explicit ranges
def month_ticks(ax, date_index):
    firsts = [i for i,d in enumerate(date_index) if d.day==1]
    if not firsts:  # fallback: ~4 evenly spaced ticks
        firsts = np.linspace(0, len(date_index)-1, 4, dtype=int).tolist()
    ax.set_xticks(firsts)
    ax.set_xticklabels([date_index[i].strftime("%b %d") for i in firsts], rotation=0)

month_ticks(ax1, fall_dates)
month_ticks(ax2, spring_dates)

# y ticks (bin labels)
ax1.set_yticks(np.arange(num_y_bins))
ax1.set_yticklabels(labels.astype(str))

for ax, piv in ((ax1, fall_piv), (ax2, spring_piv)):
    ax.grid(False)  # disable major grid (the one cutting cells)

    # horizontal lines at cell edges
    ax.set_yticks(np.arange(num_y_bins+1) - 0.5, minor=True)
    ax.yaxis.grid(True, which='minor', color='w', lw=0.6, alpha=0.25)

    # (optional) vertical lines at day boundaries
    ax.set_xticks(np.arange(piv.shape[1]+1) - 0.5, minor=True)
    ax.xaxis.grid(True, which='major', color='w', lw=0.4, alpha=0.15)   # keep major vertical grid off
    # ax.xaxis.grid(True, which='minor', color='w', lw=0.4, alpha=0.15)  # enable if you want

# keep your existing center ticks/labels:
# ax1.set_yticks(np.arange(num_y_bins)); ax1.set_yticklabels(labels.astype(str))

# visual “break”
ax1.spines['right'].set_visible(False); ax2.spines['left'].set_visible(False)
d=.015; kw=dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d),(-d,+d), **kw); ax1.plot((1-d,1+d),(1-d,1+d), **kw)
kw.update(transform=ax2.transAxes)
ax2.plot((-d,+d),(-d,+d), **kw); ax2.plot((-d,+d),(1-d,1+d), **kw)

fig.subplots_adjust(right=0.90)                   # leave 10% for cbar
plt.tight_layout(rect=[0, 0, 0.90, 0.98])         # keep that margin; top room for suptitle

cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])     # [left, bottom, width, height] in fig coords
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Detections")

# cbar = fig.colorbar(im2, ax=[ax1,ax2], location='right', shrink=0.95, pad=0.02)
fig.suptitle("Vocalograma by date × night bin", y=0.995, fontsize=12)
plt.show()
