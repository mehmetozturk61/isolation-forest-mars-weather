import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

df = pd.read_csv("bank_transactions_data_2.csv")

# PREPROCESSING
# Handle Date Related Content
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"])
df["TimeSinceLastTransaction"] = (df["TransactionDate"] - df["PreviousTransactionDate"]).dt.total_seconds()

# Drop Unnecessary Columns
drop_cols = ["TransactionID", "AccountID", "TransactionDate", "PreviousTransactionDate", "IP Address", "Location",
            "DeviceID", "MerchantID"]
df_model = df.drop(columns=drop_cols)

# Prepare Seperate Preprocesses
numerical_cols = df_model.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df_model.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Apply Preprocessing
X = preprocessor.fit_transform(df_model)

# ISOLATION FOREST
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

df["anomaly_score"] = iso_forest.decision_function(X)
df["anomaly"] = iso_forest.predict(X)

# VISUALIZATION
t = "*******************************************"
anomalies = df[df['anomaly'] == -1]
normal = df[df['anomaly'] == 1]

print(anomalies['TransactionType'].value_counts())
print(df['TransactionType'].value_counts())
print(anomalies['Channel'].value_counts())
print(df['Channel'].value_counts())
print(anomalies['CustomerOccupation'].value_counts())
print(df['CustomerOccupation'].value_counts())

print(t)

print("Anomalies - Avg Transaction Amount:", anomalies['TransactionAmount'].mean())
print("Normal - Avg Transaction Amount:", normal['TransactionAmount'].mean())
print("Anomalies - Avg Login Attempts:", anomalies['LoginAttempts'].mean())
print("Normal - Avg Login Attempts:", normal['LoginAttempts'].mean())

print(t)

print(anomalies['AccountID'].value_counts().head())
print(anomalies['MerchantID'].value_counts().head())


""" print(df[df["anomaly"] == -1])

plt.figure(figsize=(10, 6))
normal = df[df['anomaly'] == 1]
anomalies = df[df['anomaly'] == -1]
plt.scatter(normal['TransactionAmount'], normal['AccountBalance'], label='Normal', alpha=0.5)
plt.scatter(anomalies['TransactionAmount'], anomalies['AccountBalance'], color='red', label='Anomaly', alpha=0.8)
plt.xlabel("TransactionAmount")
plt.ylabel("AccountBalance")
plt.legend()
plt.show() """

