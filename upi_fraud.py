import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter

# Page settings
st.set_page_config(layout="wide")
st.title("🔐 UPI Fraud Detection: Multi-Model Comparison")

# Load and prepare data
@st.cache_resource
def load_data():
    df = pd.read_csv("upi_transactions.csv")  # <=== Rename your CSV to this!

    df["is_fraud"] = df["Status"].apply(lambda x: 1 if str(x).strip().lower() == "fraud" else 0)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.dayofweek
    df["upi_provider"] = df["Sender UPI ID"].apply(lambda x: x.split("@")[-1] if "@" in str(x) else "unknown")

    le_receiver = LabelEncoder()
    df["Receiver Name"] = le_receiver.fit_transform(df["Receiver Name"])

    le_provider = LabelEncoder()
    df["upi_provider"] = le_provider.fit_transform(df["upi_provider"])

    features = ["Amount (INR)", "hour", "day", "Receiver Name", "upi_provider"]
    X = df[features]
    y = df["is_fraud"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df, features, scaler, le_receiver, le_provider

# Load the data
X, y, raw_df, features, scaler, le_receiver, le_provider = load_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE with check
counter = Counter(y_train)
if len(counter) < 2:
    st.error("❌ Only one class present in training data. Cannot train models.")
    st.stop()

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

# Select models
st.sidebar.header("🔧 Choose Models")
selected_models = st.sidebar.multiselect(
    "Pick which models you want to compare:",
    list(models.keys()),
    default=list(models.keys())
)

# Train models
results = {}
for name in selected_models:
    model = models[name]
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    results[name] = {
        "model": model,
        "accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "fpr": fpr,
        "tpr": tpr
    }

# Plot ROC Curve
st.subheader("📈 ROC Curve Comparison")
fig, ax = plt.subplots()
for name, res in results.items():
    ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC: {res['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# Accuracy chart
st.subheader("📊 Accuracy Comparison")
acc_df = pd.DataFrame({name: [res["accuracy"]] for name, res in results.items()}, index=["Accuracy"]).T
st.bar_chart(acc_df)

# Real-time prediction input
st.sidebar.title("🔍 Predict Transaction")
amount = st.sidebar.number_input("Amount (INR)", min_value=1.0, value=500.0)
hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.slider("Day of Week", 0, 6, 2)
receiver_name = st.sidebar.selectbox("Receiver Name", raw_df["Receiver Name"].unique())
upi_provider = st.sidebar.selectbox("UPI Provider", raw_df["upi_provider"].unique())

input_df = pd.DataFrame([[amount, hour, day, receiver_name, upi_provider]], columns=features)
combined = pd.concat([raw_df[features], input_df], ignore_index=True)
scaled_combined = scaler.fit_transform(combined)
input_scaled = scaled_combined[-1:]

# Predict button
if st.sidebar.button("Detect Fraud"):
    st.subheader("🧪 Prediction Results")
    for name, res in results.items():
        prob = res["model"].predict_proba(input_scaled)[0][1]
        label = "❌ Fraud" if prob >= 0.5 else "✅ Genuine"
        st.write(f"**{name}** → {label} (Probability: {prob:.2f})")
