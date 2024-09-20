# Sonar Signal Classification

This project aims to classify sonar signals as either "Rock" or "Mine" using machine learning. The goal is to develop a model that accurately classifies these signals and provides insights into the factors influencing the classification. The project leverages XGBoost for its strong performance in classification tasks, with SHAP used to interpret feature importance and provide model transparency.

## Project Structure

### Data Exploration:
- **Dataset Examination**: The dataset contains sonar signals with 60 features representing energy within specific frequency bands. We start by understanding the dataset structure, checking for missing values, and visualizing the distribution of features.
- **Feature Transformation**: The target variable is transformed from categorical labels ("R" for Rock and "M" for Mine) into numerical labels for easier processing. Additionally, features are scaled using MinMaxScaler to ensure consistent scaling across all variables.

### Data Preprocessing:
- **Handling Missing Values**: The dataset is checked for missing values, although none were found.
- **Balancing the Dataset**: SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle any class imbalance between the "Rock" and "Mine" labels, ensuring the model doesn't bias toward the majority class.

### Model Building:
- **XGBoost Classifier**: XGBoost is chosen for its ability to handle large datasets efficiently and its built-in regularization features. The model is fine-tuned with hyperparameters such as `n_estimators`, `max_depth`, `learning_rate`, `reg_alpha` (L1 regularization), and `reg_lambda` (L2 regularization).
- **Train-Test Split**: The dataset is split into training and testing sets to evaluate the model’s generalization capabilities.

### Model Evaluation:
- **Performance Metrics**: The model is evaluated using accuracy, precision, recall, and F1-score. These metrics are calculated to assess the classifier’s effectiveness in distinguishing between "Rock" and "Mine" signals.
- **Confusion Matrix**: A confusion matrix is used to visualize the model’s predictions, offering insights into how well the model is performing on each class.
- **Loss Curves**: Plotting log-loss over training epochs helps monitor the training process, allowing detection of overfitting or underfitting.

### Feature Importance:
- **XGBoost Feature Importance**: XGBoost’s built-in feature importance functionality is used to quickly identify the most important features for classification.
- **SHAP Explainability**: SHAP (SHapley Additive exPlanations) is used to provide a more detailed visualization of how each feature contributes to the model's predictions. SHAP values are visualized both as summary plots and bar charts to show the relative impact of each feature on the model’s decision-making.

## Key Tools and Libraries
- **XGBoost**: For training a robust classification model with built-in regularization techniques.
- **SMOTE**: To balance the dataset and address class imbalance between Rock and Mine signals.
- **SHAP**: For visualizing and interpreting the impact of individual features on model predictions.
- **Scikit-learn**: For data preprocessing, train-test splitting, and evaluation metrics like precision, recall, and F1-score.
- **Matplotlib/Seaborn**: For plotting visualizations such as confusion matrices, feature importance, and loss curves.

## Results

- **Accuracy**: 88%
- **Precision**: 0.95 (Class M), 0.84 (Class R)
- **Recall**: 0.80 (Class M), 0.96 (Class R)
- **F1 Score**: 0.87 (Class M), 0.90 (Class R)
  





