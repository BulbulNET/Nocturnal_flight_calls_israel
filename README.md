# Nocturnal Flight Calls Israel
This repository contains the trained BirdNET model, example audio recordings, analysis scripts, and supporting data associated with the manuscript:

**"Listening to the Night Sky: Characterizing Nocturnal Bird Migration in a Globally Important Flyway Using Bioacoustics and Machine Learning.
"**

## Overview

This project uses passive acoustic monitoring (PAM) and a retrained BirdNET model to detect and classify nocturnal flight calls (NFCs) of migratory birds in Israel.

The custom classifier was trained on recordings of 34 common migratory bird species and 4 background-noise classes. The repository provides:

* The trained BirdNET classifier
* Example audio recordings
* Scripts for running the classifier
* Scripts for reproducing analyses and figures from the manuscript
* Processed datasets used in the statistical analyses

model can be downloaded here - https://github.com/BulbulNET/Nocturnal_flight_calls_israel/releases/tag/v1.0

## Repository structure

```text
model/      Trained BirdNET model and model dacription, labels and parameters (https://github.com/BulbulNET/Nocturnal_flight_calls_israel/releases/tag/v1.0)
data/       Processed datasets and example recordings
scripts/    Python and R scripts
results/    Example outputs
figures/    Figures reproduced from the manuscript
docs/       Additional documentation
```

## Requirements

Python scripts were developed using Python 3.10.

Required packages are listed in:

```text
environment.yml
```

Statistical analyses were performed in R (version 4.4.2).

Key R packages:

* glmmTMB
* DHARMa
* emmeans
* car

## Running the classifier

Example command:

```bash
python scripts/run_costum_model.py \
    --model model/NFC_southern_israel.tflite \
    --input data/example_recordings/
```

The output will include species predictions and confidence scores for each audio segment.

for further species inspection and a saving option:

```bash
python scripts/species_list_and_save_files.py
```

## Using the model in BirdNET Analyzer (GUI)

The custom classifier can be used directly through the BirdNET Analyzer graphical user interface (GUI) without requiring any programming experience.

To run the model:

1. Download the custom classifier files from 'release'
2. Open BirdNET Analyzer (version 2.4 or later).
3. Select **Batch Analysis**.
4. Under **Species Selection**, choose **Custom Classifier**.
5. Load the classifier provided in this repository.
6. Select the audio files to analyze.
7. Run the analysis.

BirdNET will generate detection tables containing species predictions and confidence scores for each detected call.

This workflow allows users to apply the classifier to their own recordings without modifying any code.

## Evaluating model performance on the test set or validated set

In addition to running the custom classifier on new recordings, the repository includes scripts for evaluating model performance on a labeled validated set or on the test set.

To run the custom model on the validated set:

```bash
python scripts/run_custom_model.py --input data/validated_set/ --model model/
```

This produces a prediction table with the model output for each recording.

To calculate model performance metrics:

```bash
python scripts/calculate_model_performance.py --predictions results/test_set_predictions.csv --labels data/test_set_labels.csv
```

The performance script calculates standard classification metrics, including precision and recall, based on the agreement between model predictions and the known labels of the test set.

## Reproducing manuscript analyses

Processed detection tables used in the manuscript are provided in the data directory 'all_detections_dataframe.pkl'.

To reproduce figures and statistical analyses use :

/scripts/visualization_for_data.py

## Data availability

This repository contains:

* Trained BirdNET model (https://github.com/BulbulNET/Nocturnal_flight_calls_israel/releases/tag/v1.0)
* Example recordings
* Processed detection data
* Metadata
* Analysis scripts

## Citation

If you use this repository, please cite:

Marck A., Sapir N., Lavner Y.
Listening to the Night Sky: Characterizing Nocturnal Bird Migration in a Globally Important Flyway Using Bioacoustics and Machine Learning.

```
```
