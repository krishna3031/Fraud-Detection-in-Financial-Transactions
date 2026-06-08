# 🛡️ Credit Card Fraud Detection In Financial Transactions

An unsupervised anomaly detection system for identifying fraudulent Financial transactions using the **Isolation Forest** algorithm.

---

## 📋 Overview

This project detects fraudulent transactions in credit card data without requiring a large labeled dataset. By treating fraud as an anomaly, the Isolation Forest model isolates outliers in the transaction feature space — transactions that deviate significantly from normal behavior are flagged as potential fraud.

---

## 🚀 Features

- Unsupervised fraud detection using **Isolation Forest**
- Feature scaling with **StandardScaler** for optimal model performance
- Stratified train/test split to preserve class imbalance ratios
- Model and scaler persistence with **joblib** for production inference
- Real-time prediction on new transactions

---

## 📁 Project Structure

```
fraud-detection/
│
├── Fraud_Detection.ipynb          # Main notebook
├── creditcard.csv                 # Dataset (not included — see below)
├── fraud_detection_model.pkl      # Saved Isolation Forest model
├── scaler.pkl                     # Saved StandardScaler
└── README.md
```

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

| Library      | Purpose                              |
|--------------|--------------------------------------|
| `pandas`     | Data loading and manipulation        |
| `numpy`      | Numerical operations                 |
| `scikit-learn` | Model training and evaluation      |
| `joblib`     | Model serialization                  |

---

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- **Rows:** 284,807 transactions
- **Features:** 30 (V1–V28 PCA-transformed features, `Time`, `Amount`)
- **Target:** `Class` — `0` = Legitimate, `1` = Fraud
- **Fraud rate:** ~0.17% (highly imbalanced)

> ⚠️ Download `creditcard.csv` from Kaggle and place it in the project root before running the notebook.

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Features and target are separated (`X`, `y`)
- All features are standardized using `StandardScaler`
- Data is split 80/20 into train and test sets with stratification

### 2. Model Training
```python
model = IsolationForest(
    n_estimators=100,
    contamination=0.0017,  # Matches the ~0.17% fraud rate
    random_state=42
)
model.fit(X_train)
```

### 3. Prediction Mapping
Isolation Forest outputs:
- `1`  → Normal transaction → mapped to `0`
- `-1` → Anomaly (fraud) → mapped to `1`

### 4. Evaluation
Model performance is measured using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)

---

## 💾 Saving & Loading the Model

```python
import joblib

# Save
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")
```

---

## 🔍 Real-Time Inference

To predict on a new transaction (29 features: V1–V28 + Amount):

```python
import numpy as np

new_transaction = np.array([[/* 29 feature values */]])
new_transaction = scaler.transform(new_transaction)

prediction = model.predict(new_transaction)

if prediction[0] == -1:
    print("⚠️ Fraud Alert")
else:
    print("✅ Legitimate Transaction")
```

---

## 📈 Model Parameters

| Parameter       | Value  | Description                          |
|----------------|--------|--------------------------------------|
| `n_estimators`  | 100    | Number of isolation trees            |
| `contamination` | 0.0017 | Expected proportion of anomalies     |
| `random_state`  | 42     | Seed for reproducibility             |
| `test_size`     | 0.2    | 20% held out for evaluation          |

---

## ⚠️ Limitations

- Isolation Forest is unsupervised — it does not learn from labeled fraud examples
- Performance is sensitive to the `contamination` hyperparameter
- May produce false positives on unusual but legitimate transactions
- Feature engineering beyond PCA (already applied in the dataset) could improve results

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
