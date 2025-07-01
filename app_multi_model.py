import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("🔍 UPI Fraud Detection - Multi-Model Dashboard")

# Load and preprocess data
@st.cache_resource
def load_data():
    df = pd.read_csv("96dde54b-7a00-479a-9ce3-d1df924a63b7.csv")

    # Convert status to binary fraud label
    df["is_fraud"] = df["Status"].apply(lambda x: 1 if x.strip().lower() == "fraud" else 0)

    # Extract features from timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.dayofweek

    # Extract UPI provider from UPI ID
    df["upi_provider"] = df["Sender UPI ID"].apply(lambda x: x.split("@")[-1] if "@" in str(x) else "unknown")

    # Encode categorical features
    le_receiver = LabelEncoder()
    df["Receiver Name"] = le_receiver.fit_transform(df["Receiver Name"])

    le_provider = LabelEncoder()
    df["upi_provider"] = le_provider.fit_transform(df["upi_provider"])

    # Select features for model training
    features = ["Amount (INR)", "hour", "day", "Receiver Name", "upi_provider"]
    X = df[features]
    y = df["is_fraud"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df, features, scaler, le_receiver, le_provider

# Load dataset
X, y, raw_df, features, scaler, le_receiver, le_provider = load_data()

# Train-test split and handle imbalance with SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

# Select models to train
st.sidebar.header("🧠 Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to compare:",
    list(models.keys()),
    default=list(models.keys())
)

# Train and evaluate selected models
results = {}
for name in selected_models:
    model = models[name]
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "fpr": fpr,
        "tpr": tpr
    }

# ROC Curve
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

# Accuracy bar chart
st.subheader("📊 Accuracy Comparison")
acc_df = pd.DataFrame({name: [res["accuracy"]] for name, res in results.items()}, index=["Accuracy"]).T
st.bar_chart(acc_df)

# Transaction input form
st.sidebar.title("🔍 Predict a Transaction")
amount = st.sidebar.number_input("Transaction Amount (INR)", min_value=1.0, value=500.0)
hour = st.sidebar.slider("Hour (0-23)", 0, 23, 12)
day = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
receiver_name = st.sidebar.selectbox("Receiver Name", raw_df["Receiver Name"].unique())
upi_provider = st.sidebar.selectbox("UPI Provider", raw_df["upi_provider"].unique())

# Prepare input
input_data = pd.DataFrame([[amount, hour, day, receiver_name, upi_provider]], columns=features)
combined = pd.concat([raw_df[features], input_data], ignore_index=True)
scaled_combined = scaler.fit_transform(combined)
input_scaled = scaled_combined[-1:]

# Predict button
if st.sidebar.button("🔎 Predict Fraud"):
    st.subheader("🧪 Prediction Results")
    for name, res in results.items():
        prob = res["model"].predict_proba(input_scaled)[0][1]
        label = "❌ Fraud" if prob >= 0.5 else "✅ Genuine"
        st.write(f"**{name}** → {label} (Probability: {prob:.2f})")
