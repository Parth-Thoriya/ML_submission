import streamlit as st
import pandas as pd
import json
from sklearn.metrics import classification_report
import joblib

st.title("Model Performance Comparison")

# Load classification reports
with open("classification_reports.json", "r") as f:
    classification_reports = json.load(f)

# Convert to DataFrame
data = []
for model, report in classification_reports.items():
    accuracy = report["accuracy"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]
    data.append([model, accuracy, precision, recall, f1_score])

df_metrics = pd.DataFrame(data, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

# Display table in Streamlit
st.dataframe(df_metrics)
