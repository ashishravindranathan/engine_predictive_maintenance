---
license: mit
tags:
- predictive-maintenance
- engine-health
- classification
- tabular-data
---

# Engine Predictive Maintenance Model

This repository stores the trained model artifact for deployment.

## Source Dataset
- Dataset Repository: `AshishTatapuzha/engine-predictive-maintenance-dataset`
- Dataset File in Hub: `data/engine_data.csv`
- Local Dataset File: `/root/ashiravi/predictive_maintenance_project/data/engine_data.csv`

## Saved Model Details
- Notebook model variable used: `best_model`
- Model artifact file: `engine_condition_model.joblib`

## Inference Features
[
  "Engine rpm",
  "Lub oil pressure",
  "Fuel pressure",
  "Coolant pressure",
  "lub oil temp",
  "Coolant temp"
]

## Target Labels
{
  "0": "Normal",
  "1": "Faulty"
}
