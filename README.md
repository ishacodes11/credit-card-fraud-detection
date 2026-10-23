# Credit Card Fraud Detection Using Machine Learning

## Overview
This project detects potentially fraudulent credit card transactions using machine learning. It compares Logistic Regression with XGBoost and provides a Streamlit web application for testing transaction inputs.

## Objectives
- Analyze transaction data and handle class imbalance.
- Train and evaluate machine learning models.
- Compare fraud detection performance using appropriate metrics.
- Provide an interactive fraud prediction dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- Joblib

## Dataset
The project uses the Credit Card Fraud Detection dataset from Kaggle. The target column is `Class`, where `0` represents a legitimate transaction and `1` represents a fraudulent transaction.

## Model Performance
**XGBoost — held-out test set**

| Metric | Result |
|---|---:|
| Precision (fraud class) | 84.44% |
| Recall (fraud class) | 80.00% |
| F1-score (fraud class) | 0.8216 |
| ROC-AUC | 0.9714 |
| Average Precision | 0.8200 |
| Test transactions | 56,746 |

Confusion matrix:

```text
[[56637    14]
 [   19    76]]
```

## Project Files
- `app.py` — Streamlit application
- `xgboost_fraud_model.json` — saved XGBoost model
- `feature_scaler.pkl` — fitted feature scaler
- `feature_columns.pkl` — feature column order

## Run Locally
1. Install Python and Git.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit matplotlib joblib
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## Limitations
This is an educational prototype, not a production banking fraud detection system. Predictions depend on the training data, preprocessing, and decision threshold. The model should not be used to make real financial decisions.

Rashmika
