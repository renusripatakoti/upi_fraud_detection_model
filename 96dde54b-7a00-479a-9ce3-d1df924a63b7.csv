import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2000

df = pd.DataFrame({
    "Transaction ID": np.arange(100000, 100000 + n_samples),
    "Timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq="min"),
    "Sender Name": np.random.choice(["Amit", "Ravi", "Neha", "Priya", "Suresh", "Anjali"], n_samples),
    "Sender UPI ID": [f"user{i}@okaxis" for i in np.random.randint(100, 200, n_samples)],
    "Receiver Name": np.random.choice(["Flipkart", "Amazon", "Zomato", "Swiggy", "Paytm", "IRCTC"], n_samples),
    "Receiver UPI ID": [f"merchant{i}@oksbi" for i in np.random.randint(1000, 2000, n_samples)],
    "Amount (INR)": np.round(np.random.uniform(10, 10000, n_samples), 2),
    "Status": np.random.choice(["Success", "Failed", "Fraud"], p=[0.9, 0.04, 0.06], size=n_samples)
})

df.to_csv("enhanced_upi_fraud_dataset.csv", index=False)
print("✅ Dataset saved as 'enhanced_upi_fraud_dataset.csv'")
