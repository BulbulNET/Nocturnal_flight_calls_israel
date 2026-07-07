# -*- coding: utf-8 -*-
"""
Run custom model for applying BirdNET 2.4 on a new dataset - validated or not.
2024_05
"""

import os
import time
from operator import itemgetter

import librosa
import pandas as pd

import analyze1

# ============================================================
# CONFIGURATION — edit this section before running
# ============================================================

# Set MODE to either "real" or "valid"
MODE = "valid"

PATHS = {
    "real": "../data/example_recordings/",
    "valid": "../data/validated_set/",
}

OUTPUT_NAMES = {
    "real": "custom_NFC_model_on_new_data",
    "valid": "custom_NFC_model_on_validated_set",
}

# In "real" mode the recordings sit directly in the folder (flat).
# In "test" mode the folder contains one sub-folder per class (nested).
NESTED = {
    "real": False,
    "valid": True,
}

OUTPUT_DIR = "../output"
TOP_N_BNT = 3
SEGMENT_LENGTH = 3      # seconds
SAMPLE_RATE = 48_000
CONF_SCORE = 0.1        # kept for reference; extend as needed

# ============================================================
# COLUMNS
# ============================================================

COLUMNS = [
    "recorder_id",
    "start_seg", "end_seg",
    "species_1", "probability_1",
    "species_2", "probability_2",
    "species_3", "probability_3",
]

# ============================================================
# HELPERS
# ============================================================

def process_bnt_pred(preds_bnt_in, top_n, brdnt_train):
    """Filter and keep only the top-N BirdNET predictions."""
    if not brdnt_train:
        # Remove labels without a match in agamon (non-integer keys)
        preds_bnt_in = {k: v for k, v in preds_bnt_in.items() if isinstance(k, int)}

    n = min(top_n, len(preds_bnt_in))
    return dict(sorted(preds_bnt_in.items(), key=itemgetter(1), reverse=True)[:n])


def append_event(df, event, metadata):
    """Append one prediction event as a new row to *df* and return it."""
    species = list(event.keys())
    probabilities = list(event.values())

    row = {
        "recorder_id":   metadata["recorder_id"],
        "start_seg":     metadata["start_seg"],
        "end_seg":       metadata["end_seg"],
        "species_1":     species[0],      "probability_1": probabilities[0],
        "species_2":     species[1],      "probability_2": probabilities[1],
        "species_3":     species[2],      "probability_3": probabilities[2],
    }
    return pd.concat([df, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)


def analyze_audio(file_path, df, label):
    """Segment *file_path* into 3-second clips, run BirdNET, append results."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=audio, sr=sr)

    for start in range(0, int(duration), SEGMENT_LENGTH):
        segment = audio[start * sr : (start + SEGMENT_LENGTH) * sr]
        _, preds_raw = analyze1.birdnet_predict(segment, sr)
        preds = process_bnt_pred(preds_raw, TOP_N_BNT, brdnt_train=True)
        print(preds)

        metadata = {"recorder_id": label, "start_seg": start, "end_seg": start + SEGMENT_LENGTH}
        df = append_event(df, preds, metadata)
        print("--" * 27)

    return df


def iter_files(directory, nested):
    """
    Yield (file_path, label) pairs.

    - nested=False : files sit directly in *directory*; label = filename
    - nested=True  : sub-folders are class labels; label = sub-folder name
    """
    if nested:
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if not os.path.isdir(folder_path):
                continue
            for file in os.listdir(folder_path):
                yield os.path.join(folder_path, file), folder
    else:
        for file in os.listdir(directory):
            yield os.path.join(directory, file), file


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    directory_path = PATHS[MODE]
    output_name    = OUTPUT_NAMES[MODE]
    nested         = NESTED[MODE]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"BirdNET loaded from local environment  |  mode={MODE}")

    df = pd.DataFrame(columns=COLUMNS)
    start_time = time.time()

    for file_path, label in iter_files(directory_path, nested):
        print(label)
        df = analyze_audio(file_path, df, label)

    out_csv = os.path.join(OUTPUT_DIR, output_name + ".csv")
    df.to_csv(out_csv, index=False, quoting=1)

    print(f"--- {time.time() - start_time:.1f} seconds ---")
    print(f"Created file: {out_csv}")
