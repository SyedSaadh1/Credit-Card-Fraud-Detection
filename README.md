# Credit Card Fraud Detection

## Overview
This project aims to develop machine learning models to detect fraudulent transactions using the Kaggle Credit Card Fraud Detection dataset. The task focuses on:
- Data handling and preprocessing
- Training supervised and unsupervised models
- Evaluating performance using metrics and visualizations
- Providing explainability for model decisions using SHAP.

## Dataset
The dataset consists of anonymized credit card transaction data, including 28 anonymized features (`V1` to `V28`), the `Amount` of the transaction, and a target label `Class`:
- `Class = 1`: Fraudulent transaction
- `Class = 0`: Non-fraudulent transaction

Dataset Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Project Workflow

### 1. Data Exploration and Preprocessing
- **Data Inspection**: Checked for missing values, data types, and class imbalance.
- **Scaling**: Scaled the `Amount` feature using `StandardScaler` to standardize its distribution.
- **Class Imbalance Handling**: Addressed severe class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

### 2. Supervised Model Development
- **Baseline Model**: Logistic Regression to establish a baseline performance.
- **Primary Model**: XGBoost Classifier for improved accuracy and recall, with basic hyperparameter tuning.

### 3. Unsupervised Model Development
- **Isolation Forest**: Anomaly detection model to identify rare fraudulent transactions. Configured with contamination parameter to capture 1% anomalies.

### 4. Model Evaluation
- **Supervised Models**:
  - Metrics: Accuracy, Recall, Precision, F1-Score.
  - Visualizations: Confusion Matrix and ROC Curve.
- **Unsupervised Models**:
  - Evaluated anomaly detection against known fraudulent transactions.

### 5. Explainability
- **Feature Importance**: SHAP values to identify key features influencing fraud detection.
- **Local Interpretability**: Explained individual predictions with SHAP force plots.

## Results

### Supervised Models
| Metric        | Logistic Regression | XGBoost |
|---------------|---------------------|---------|
| Accuracy      | 96.8%              | 98.3%   |
| Recall (Fraud)| 84.0%              | 92.5%   |
| Precision     | 77.0%              | 88.1%   |
| F1-Score      | 80.3%              | 90.2%   |

### Unsupervised Model
- **Isolation Forest**:
  - Recall: ~70%
  - Precision: ~50%
  - Detected anomalies effectively but with higher false positives.

### Explainability
- SHAP analysis highlighted key features like `V12`, `V14`, and `V17` as critical in predicting fraud.

## Reproducing the Results
### Requirements
Install the necessary libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

### Execution Steps
1. **Download Dataset**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` directory.
2. **Run the Script**:
   - Execute the provided Python script or Jupyter Notebook.
3. **Visualize Results**:
   - Outputs include confusion matrices, ROC curves, and SHAP visualizations.

### Key Files
- `fraud_detection.ipynb`: Contains all steps from data exploration to model evaluation.
- `README.md`: Overview and reproduction instructions.

## Conclusion
- XGBoost performed best among supervised models, achieving high recall and precision.
- Isolation Forest demonstrated the potential for anomaly detection but needs optimization to reduce false positives.
- SHAP provided insights into feature importance, aiding model interpretability.
