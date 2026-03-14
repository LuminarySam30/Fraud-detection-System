# 💳 Intelligent Credit Card Fraud Detection System

An end-to-end Machine Learning solution for detecting fraudulent credit card transactions using **XGBoost**, **SMOTE**, and **Streamlit**.

[**🚀 Live Demo**](https://fraud-detection-system-samuel.streamlit.app/)

### **🚀 Project Overview**
This project addresses the critical challenge of credit card fraud detection in highly imbalanced datasets (only 0.58% fraud). By engineering advanced behavioral and geographic features and utilizing synthetic oversampling, we built a model that achieves elite performance in identifying suspicious activities.

**Source Data**: The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) by Kartikey Bartwal on Kaggle.

### **✨ Key Features**
- **Deep EDA**: Analysis of transaction patterns and class imbalance.
- **Advanced Feature Engineering**: 
  - **Temporal**: Hour of day, day of week, and weekend flags.
  - **Geographic**: `distance_from_home` calculated via Haversine formula.
  - **Behavioral**: `amt_ratio`, `amt_deviation`, and merchant familiarity counts.
- **Data Balancing**: Used **SMOTE** to handle extreme class imbalance.
- **High-Performance Model**: Trained **XGBoost** with GPU acceleration.
- **Threshold Optimization**: Fine-tuned decision boundaries to prioritize **Recall (88%)**.
- **Interactive UI**: Real-time fraud checker web app built with **Streamlit**.

## 📊 Model Performance
The model is optimized to "catch the thief" (Minimize False Negatives).

| Metric | Result |
| :--- | :--- |
| **ROC-AUC** | **0.9967** (Elite) |
| **Recall (Fraud Detection)** | **88.02%** |
| **Accuracy** | **99.72%** |
| **Precision** | **59.18%** |

### **Visualization**
The project includes **interactive Plotly graphs** in the notebook for zooming into transaction distributions and viewing performance curves (ROC & Precision-Recall).

## 📁 Repository Structure
- `notebooks/`: Contains the full analysis pipeline (`FAI_Project.ipynb`).
- `data/`: CSV datasets (Training and Testing).
- `reports/`: Detailed ML Pipeline Report PDF.
- `app.py`: Streamlit web application.
- `fraud_model.pkl`: Compressed trained XGBoost brain.

## 🛠️ How to Run

### **1. Notebook Analysis**
To view the full training process:
```bash
jupyter notebook notebooks/FAI_Project.ipynb
```

### **2. Streamlit Web App**
To launch the interactive fraud checker:
```bash
pip install streamlit xgboost plotly pandas numpy
streamlit run app.py
```

## 🧠 Technologies Used
- **Python** (Pandas, NumPy)
- **Scikit-Learn** & **Imbalanced-Learn**
- **XGBoost** (GPU Accelerated)
- **Plotly** & **Seaborn** (Interactive Visualization)
- **Streamlit** (Web Framework)
