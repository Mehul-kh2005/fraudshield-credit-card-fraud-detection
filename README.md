# 🛡️ FRAUDSHIELD – Credit Card Fraud Detection 💳

![Python](https://img.shields.io/badge/Python-3.11-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Project Overview  

**FRAUDSHIELD** is a **machine learning–based credit card fraud detection system** designed to identify fraudulent transactions in real-time.  
The goal is to safeguard customers and financial institutions by detecting high-risk activity with high precision and recall.  

This project provides:  

- ✅ End-to-end ML workflow from preprocessing → training → evaluation → deployment  
- ✅ Comparison of **multiple machine learning models**  
- ✅ Final tuned **Random Forest classifier** for robust detection  
- ✅ Metrics: **Accuracy, ROC-AUC, Precision, Recall, F1-score**  
- ✅ Interactive **Streamlit web app** for real-time predictions  

---

## 🚀 Key Features  

- ✅ **Data Preprocessing** (scaling, resampling for imbalance handling)  
- ✅ Training with Logistic Regression, Decision Tree, Random Forest, KNN, SVM  
- ✅ **Model comparison table** with ROC-AUC & accuracy  
- ✅ Final **Random Forest (tuned)** model with superior fraud detection  
- ✅ **Confusion matrix, classification reports, ROC/PR curves** for evaluation  
- ✅ Deployed via **Streamlit app** for transaction testing  

---

## 📂 Project Structure

```plaintext
📦 FRAUDSHIELD-Credit-Card-Fraud-Detection
│
├── app.py                          # Streamlit application
├── credit_card.csv                 # Original dataset
├── data_preprocessing.ipynb        # Data preprocessing & EDA
├── model_training.ipynb            # Model training, tuning & evaluation
├── fraudshield_best_model.pkl      # Final trained Random Forest model
├── scaler.pkl                      # StandardScaler for preprocessing
├── single_transaction_examples.csv # Example inputs for testing in app
├── X_train_resampled.csv           # Balanced training features
├── y_train_resampled.csv           # Balanced training labels
├── X_test.csv                      # Testing features
├── y_test.csv                      # Testing labels
├── logo.png                        # Project logo for Streamlit UI
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Ignore unnecessary files
```

---

## 📊 Problem Statement

Credit card fraud is a **major financial threat**, causing billions in losses annually.  
This project builds an **ML-based fraud detection system** to classify transactions as:

- **1 → Fraudulent**  
- **0 → Legitimate**

---

## ⚙️ Tech Stack

| Component         | Technology |
|-------------------|------------|
| **Language**      | Python 3.11 |
| **ML Models**     | Logistic Regression, Decision Tree, Random Forest, KNN, SVM |
| **Data Handling** | Pandas, NumPy |
| **Preprocessing** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment**    | Streamlit |
| **Model Storage** | joblib (.pkl) |

---

## 📈 Model Training & Results

Multiple ML models were trained and compared:

| Model                   | Accuracy | ROC-AUC |
|-------------------------|----------|---------|
| Logistic Regression     | 0.9729   | 0.9621  |
| Decision Tree           | 0.9977   | 0.8675  |
| Random Forest           | 0.9994   | 0.9790  |
| K-Nearest Neighbors     | 0.9977   | 0.9204  |
| Support Vector Machine  | 0.9735   | 0.9651  |

### ✅ Best Model: **Random Forest (Tuned)**

- **Accuracy:** 0.9994  
- **ROC-AUC:** 0.9790  
- **Precision (Fraud):** 0.86  
- **Recall (Fraud):** 0.76  
- **F1-score (Fraud):** 0.80

> Despite extreme class imbalance, the tuned Random Forest achieved excellent fraud detection performance.

---

## 🧪 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Mehul-kh2005/fraudshield-credit-card-fraud-detection.git
cd FRAUDSHIELD-Credit-Card-Fraud-Detection
```

### 2. Create Virtual Environment & Install Dependencies
```bash
conda create -n fraudshield python=3.11 -y
conda activate fraudshield
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📦 Model Artifacts

- ✅ **fraudshield_best_model.pkl** – Final tuned Random Forest model  
- ✅ **scaler.pkl** – StandardScaler used for normalization  
- ✅ **Resampled training datasets** stored for reproducibility

---

## 📚 Dataset

**credit_card.csv**  
**Source:** [Kaggle – European Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 👨‍💻 Author

**Mehul Khandelwal** – [GitHub](https://github.com/Mehul-kh2005) | [LinkedIn](https://www.linkedin.com/in/mehulkhandelwal2005/)

---

## 💡 Inspiration

Fraud detection requires **high recall** while balancing precision due to imbalanced datasets.  
This project demonstrates:

- Handling imbalanced datasets using resampling  
- Model selection based on **ROC-AUC & PR-AUC** instead of raw accuracy  
- Deploying ML models in a **real-time Streamlit app**

---

## ✅ TODOs (Future Enhancements)

- Deploy with **FastAPI** for production APIs  
- Add **MLflow** for experiment tracking  
- Implement **Explainable AI (SHAP/LIME)** for decision transparency  
- Real-time streaming fraud detection with **Kafka + Spark**

---

## 📜 License  

This project is open-source under the **MIT License**.