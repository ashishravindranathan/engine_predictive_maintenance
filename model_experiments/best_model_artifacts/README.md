---
library_name: scikit-learn
license: mit
tags:
- predictive-maintenance
- tabular-classification
- scikit-learn
- xgboost
- ensemble-learning
datasets:
- AshishTatapuzha/engine-predictive-maintenance-dataset
---

# Engine Predictive Maintenance - Best Model

## Model Summary
This repository contains the best-performing machine learning model for the engine predictive maintenance classification task.

## Business Objective
The objective of this model is to classify engine condition as:
- **0 = Normal**
- **1 = Faulty**

This can help reduce unplanned maintenance, improve engine reliability, and support proactive service decisions.

## Training Data
The model was trained using the prepared training split stored in the Hugging Face dataset repository:
- `AshishTatapuzha/engine-predictive-maintenance-dataset`
- Training file: `prepared_data/train.csv`
- Testing file: `prepared_data/test.csv`

## Best Model
- **Algorithm:** Random Forest
- **Primary selection metric:** Test Recall (minimize missed faulty engines)

## Features Used
- Engine rpm
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- lub oil temp
- Coolant temp

## Best Hyperparameters
{
  "max_depth": null,
  "min_samples_leaf": 1,
  "n_estimators": 300
}

## Test Performance
- **Recall (primary):** 0.6654
- **Precision:** 0.7317
- **F1 Score:** 0.6970
- **ROC AUC:** 0.6700
- **Accuracy:** 0.6353

## Repository Contents
- `best_model.joblib` : serialized trained model
- `feature_columns.json` : list of input feature names
- `best_model_params.json` : best hyperparameters
- `best_model_metrics.json` : evaluation metrics
- `model_comparison.csv` : comparison of all tuned models
- `all_model_tuning_results.csv` : full tuning log across all models
- `best_model_classification_report.csv` : classification report
- `requirements.txt` : package versions for reproducibility

## Notes
This model was selected after tuning and evaluating the following algorithms:
- Decision Tree
- Bagging
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
