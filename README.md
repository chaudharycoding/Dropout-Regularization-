# Sonar Signal Classification with XGBoost and SHAP Explainability

## Project Overview

This project focuses on classifying sonar signals as either "Rock" or "Mine" using the Sonar dataset. The goal is to build a robust classification model that addresses class imbalance and provides model explainability. 

Key techniques include:
- **XGBoost** for classification with regularization to prevent overfitting.
- **SMOTE** for handling imbalanced data.
- **SHAP explainability** to interpret feature importance and model behavior.
- Performance evaluation using metrics like accuracy, precision, recall, and F1-score.

## Features

- **XGBoost with Regularization**: Applied both L1 (Lasso) and L2 (Ridge) regularization to enhance model generalization and reduce overfitting.
- **Class Imbalance Handling**: Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Model Explainability**: Utilized SHAP (SHapley Additive exPlanations) to visualize feature importance and provide insights into the model's decision-making process.
- **Evaluation**: Assessed model performance through confusion matrix visualization, and key classification metrics including precision, recall, and F1-score.

## Dataset

The dataset used is the [Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)), which contains 208 instances of sonar signals classified into two categories: Rock (R) and Mine (M).

## Methodology

1. **Data Preprocessing**:
   - Scaled the features using MinMaxScaler.
   - Addressed class imbalance using SMOTE to oversample the minority class.

2. **Modeling**:
   - Implemented an XGBoost classifier with the following parameters:
     - `n_estimators=400`
     - `max_depth=6`
     - `learning_rate=0.04`
     - `reg_alpha=0.01` (L1 regularization)
     - `reg_lambda=1` (L2 regularization)
     - `subsample=0.8`
   - Split the data into training and test sets (75% training, 25% test).
   
3. **Evaluation**:
   - Visualized training progress using Log Loss over epochs.
   - Evaluated the model with a confusion matrix and key classification metrics (precision, recall, F1-score, and accuracy).
   - Used SHAP explainability for feature importance and detailed analysis of model predictions.

## Results

- **Accuracy**: 88%
- **Precision**: 0.95 (Class M), 0.84 (Class R)
- **Recall**: 0.80 (Class M), 0.96 (Class R)
- **F1 Score**: 0.87 (Class M), 0.90 (Class R)
  
Feature importance was visualized using SHAP values, providing transparency into how each feature influenced the model's predictions.

## Visualizations

### Model Performance Metrics

<img src="performance_metrics.png" alt="Performance Metrics" width="600">

### Confusion Matrix

<img src="confusion_matrix.png" alt="Confusion Matrix" width="600">

### SHAP Feature Importance

<img src="shap_summary.png" alt="SHAP Summary Plot" width="600">

## Setup

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sonar-signal-classification.git
   cd sonar-signal-classification
