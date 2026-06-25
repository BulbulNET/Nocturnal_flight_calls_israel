# Model Description

## Overview

This repository contains a custom BirdNET classifier developed for the detection and classification of nocturnal flight calls (NFCs) of migratory birds in Israel.

The model was created by retraining the BirdNET framework using a region-specific dataset compiled from public sound libraries and validated field recordings. The classifier was designed to support automated monitoring of nocturnal migration in the Afro-Palearctic flyway.

model can be downloaded - https://github.com/BulbulNET/Nocturnal_flight_calls_israel/releases/tag/v1.0

## Base Model

* Framework: BirdNET Analyzer
* Version: 2.4
* Training approach: Transfer learning using BirdNET "train" mode

## Training Dataset

The training dataset consisted of:

* 34 migratory bird species
* 4 background-noise classes
* 8,280 labeled audio recordings

Training recordings were obtained from:

* Xeno-canto
* Macaulay Library
* Expert-validated field recordings

The inclusion of noise classes was intended to improve the model's ability to distinguish true flight calls from environmental sounds and recording artefacts.

## Performance

Model performance was evaluated using an independent test dataset comprising 20% of the available recordings.

Performance metrics:

| Metric         | Value |
| -------------- | ----- |
| Mean Precision | 0.96  |
| Mean Recall    | 0.94  |

Performance was calculated as the average species-level precision and recall across all target classes.

The model was additionally evaluated on manually validated field recordings collected during the study, demonstrating successful application under real acoustic conditions.

## Intended Use

The classifier is intended for:

* Detection of nocturnal flight calls
* Species-level classification of migratory birds
* Passive acoustic monitoring of migration
* Long-term migration phenology studies

## Reference

This model was developed for the study:

Marck A., Sapir N., Lavner Y. et al.

Automated Bioacoustic Monitoring Reveals Species-Level Patterns of Nocturnal Bird Migration in a Major Flyway.
