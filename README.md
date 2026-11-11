# 💳 Fraud Detection Using Machine Learning

## Project Overview
This project uses machine learning to classify financial transactions as fraudulent or non-fraudulent. It includes data analysis, preprocessing, feature selection, model training, and a Streamlit application for predictions.

## Objectives
- Explore and understand transaction data.
- Clean and preprocess the dataset.
- Analyze numerical and categorical features.
- Handle skewness and outliers where appropriate.
- Select useful features for fraud detection.
- Train and evaluate machine learning models.
- Predict whether a transaction is potentially fraudulent.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SciPy
- Streamlit
- Joblib

## Models
The project evaluates the following algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

## Repository Structure

```text
Fraud-Detection/
├── app.py
├── fraud_detection_model.joblib
├── requirements.txt
└── README.md
```

> Make sure the model filename in `app.py` matches the actual uploaded model file. If preprocessing steps (such as scaling or Yeo-Johnson transformation) were fitted separately, save and load those fitted objects too. Prefer saving the complete fitted preprocessing-and-model pipeline as one artifact.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Fraud-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

## Deployment
Push the project files to GitHub, then connect the repository to [Streamlit Community Cloud](https://share.streamlit.io/). Select `app.py` as the app's main file.

## Notes
- The trained model file (`fraud_detection_model.joblib`) is generated from the fitted model or pipeline in the training notebook; it cannot be recreated from this README.
- Do not upload private credentials, API keys, or sensitive transaction data.

## Author
Fraud Detection Machine Learning Project
