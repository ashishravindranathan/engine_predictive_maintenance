---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/engine_data.csv
---

# Engine Predictive Maintenance Dataset

## Description
This dataset contains engine sensor readings used for predictive maintenance.

## Files
- `data/engine_data.csv`

## Features
- Engine rpm
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- lub oil temp
- Coolant temp

## Target Variable
- Engine Condition
  - 0 = Normal
  - 1 = Faulty

## Dataset Shape
- Rows: 19535
- Columns: 7

## Use Case
Binary classification for predicting whether an engine is in normal condition or faulty.
