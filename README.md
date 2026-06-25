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

## Repository structure

```text
model/      Trained BirdNET model, labels and parameters
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

## Reproducing manuscript analyses

Processed detection tables used in the manuscript are provided in the data directory.

To reproduce the statistical analyses:

```bash
Rscript scripts/glmm_analysis.R
```

## Data availability

This repository contains:

* Trained BirdNET model
* Example recordings
* Processed detection data
* Metadata
* Analysis scripts

Raw recordings collected during the study are not included in this repository.

## Citation

If you use this repository, please cite:

Marck A., Sapir N., Lavner Y.
Listening to the Night Sky: Characterizing Nocturnal Bird Migration in a Globally Important Flyway Using Bioacoustics and Machine Learning.

```
```
