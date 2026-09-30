# 💳 Real-Time Fraud Detection System

An end-to-end Machine Learning system that detects fraudulent transactions in real-time using a trained model, FastAPI backend, and interactive dashboard.

---

## 🚀 Overview

This project simulates a real-world fraud detection pipeline used in financial systems. It processes transaction data, predicts fraud probability, and categorizes risk levels instantly through an API.

The system is designed to handle highly imbalanced data and prioritize fraud detection (high recall) while balancing false positives using thresold tuning.

---

## 🎯 Key Features

- 🔍 Fraud detection using Machine Learning
- ⚡ Real-time prediction via FastAPI
- 📊 Interactive dashboard using Streamlit
- 🔄 Live transaction simulation
- ⚖️ Threshold tuning for precision-recall trade-off
- 📉 Handles imbalanced dataset effectively
- 🧠 Business-oriented risk scoring (Low / Medium / High)

---

## 🧠 Machine Learning Approach

- Dataset: Credit Card Fraud Detection Dataset
- Problem Type: Binary Classification (Fraud vs Non-Fraud)

### Key Steps:
- Data preprocessing & scaling
- Handling class imbalance
- Model training using Random Forest
- Evaluation using:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

### ⚠️ Important Insight:
Accuracy is misleading due to imbalance.  
This system prioritizes **Recall (Fraud Detection Rate)** and tunes thresholds to improve precision.

---

## ⚙️ Tech Stack

| Component        | Technology Used |
|----------------|----------------|
| ML Model       | Scikit-learn (Random Forest) |
| Backend API    | FastAPI |
| Real-time Server | Uvicorn |
| Dashboard      | Streamlit |
| Data Handling  | Pandas, NumPy |
| Model Storage  | Joblib |

---

## 🏗️ Project Architecture

fraud-system/
│
├── data/
│ └── raw/
│ └── creditcard.csv
│
├── notebooks/
│ └── fraud_detection.ipynb
│
├── src/
│ ├── train.py
│ ├── predict.py
│ └── simulate.py
│
├── api/
│ └── main.py
│
├── dashboard/
│ └── app.py
│
├── models/
│ ├── fraud_model.pkl
│ └── scaler.pkl
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sk-bhaskar/fraud-system.git
cd fraud-system

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model
python src/train.py

5️⃣ Start FastAPI server
uvicorn api.main:app --reload

Open:

http://127.0.0.1:8000/docs

6️⃣ Run transaction simulation
python src/simulate.py

7️⃣ Launch dashboard
streamlit run dashboard/app.py

📊 Sample API Response
{
  "prediction": 1,
  "fraud_probability": 0.92,
  "risk_level": "high"
}
