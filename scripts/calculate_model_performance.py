# -*- coding: utf-8 -*-
"""
Read model predictions on the test set, calculate performance metrics
(precision, recall, average probability), and plot confusion matrices.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================

"""
Set either "test" or "valid"
"""
# TEST SET -

INPUT_CSV       = "../output/custom_NFC_model_on_testset.csv"
OUTPUT_CSV      = "../output/nocturnal_migration_test_performance.csv"

# VALIDATION TEST -

# INPUT_CSV       = "../output/custom_NFC_model_on_validated_set.csv"
# OUTPUT_CSV      = "../output/nocturnal_migration_valid_performance.csv"

OUTPUT_DIR      = "../output"

THRESHOLD       = 0.1
FIGURE_SIZE     = (17, 15)

# Font sizes for confusion matrix plots
FONT = {
    "title":      16,
    "axis_label": 14,
    "tick":       11,
    "annot":      10,
}

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df['species_1'] = df['species_1'].str.lower()
df['recorder_id'] = df['recorder_id'].str.lower()

# ============================================================
# OPEN TRAINING-DATA DICTIONARY  {class_name: file_count}
# ============================================================

with open("training_data_dict.pkl", "rb") as file:
    training_data_dict = pickle.load(file)

# ============================================================
# COMPUTE PER-SPECIES PERFORMANCE
# ============================================================

columns = ["species", "average probability", "recall", "precision", "training data"]
species_performance_df = pd.DataFrame()

df = df.drop(df[df.probability_1 < THRESHOLD].index)
species_list = np.unique(df.recorder_id)

for species in species_list:
    species1_list_total = df.groupby("species_1")["species_1"].count()
    species1_total = species1_list_total.get(species, 0)

    species_df = df[df.recorder_id == species]
    test_data  = len(species_df)

    out = (
        species_df
        .groupby(["recorder_id", "species_1"])
        .agg({"probability_1": ["size", "mean", "std"]})
        .reset_index()
    )
    out.columns = ["_".join(col) for col in out.columns]

    tp, prob = 0, 0
    for _, row in out.iterrows():
        if row["recorder_id_"] in row["species_1_"]:
            tp   = row["probability_1_size"]
            prob = row["probability_1_mean"]


    precision = tp / species1_total if species1_total > 0 else 0   # how many retrieved items are relevant?
    recall    = tp / test_data        # how many relevant items are retrieved?

    row = {
        "species":             species,
        "recall":              recall,
        "average probability": prob,
        "precision":           precision,
        "training data":       training_data_dict[species],
    }
    species_performance_df = pd.concat(
        [species_performance_df, pd.DataFrame([row], columns=columns)],
        ignore_index=True,
    )

species_performance_df.to_csv(OUTPUT_CSV, index=False, quoting=1)
print(f"Performance CSV saved to {OUTPUT_CSV}")

# ============================================================
# CONFUSION MATRICES
# ============================================================

y_true  = df["recorder_id"]
y_pred  = df["species_1"]
labels  = sorted(set(y_true))


def plot_confusion_matrix(cm_df, title, fmt , cmap="crest"):
    """Plot a seaborn heatmap with consistent font sizes."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=fmt,
        cmap= cmap,
        ax=ax,
        annot_kws={"size": FONT["annot"]},
    )
    ax.set_title(title,      fontsize=FONT["title"])
    ax.set_xlabel("Predicted Label", fontsize=FONT["axis_label"])
    ax.set_ylabel("True Label",      fontsize=FONT["axis_label"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    plt.tight_layout()
    
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()


# --- Raw counts ---
all_labels = sorted(set(y_true) | set(y_pred))
cm         = confusion_matrix(y_true, y_pred, labels=all_labels)
cm_df      = pd.DataFrame(cm, index=all_labels, columns=all_labels)
plot_confusion_matrix(cm_df, "Confusion Matrix (counts)", fmt="d", cmap="RdPu")

# --- Normalised ---
cmn    = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cmn_df = pd.DataFrame(cmn, index=labels, columns=sorted(set(y_pred)))
plot_confusion_matrix(cmn_df, "Confusion Matrix (normalised)", fmt=".1f" ,cmap="crest" )

