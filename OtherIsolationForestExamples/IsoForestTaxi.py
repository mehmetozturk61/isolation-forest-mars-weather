import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("nyc_taxi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

X = df[["value"]]
X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(contamination=0.01)
df["anomaly"] = model.fit_predict(X_scaled)

""" pd.set_option('display.max_rows', 200)
anomalies = df[df["anomaly"] == -1]
print(anomalies.head(102)) """

plt.figure(figsize=(14,6))
plt.plot(df.index, df["value"], label="Passengers")
plt.scatter(df[df["anomaly"] == -1].index, df[df["anomaly"] == -1]["value"], color="red", label="Anomaly")
plt.legend()
plt.title("Taxi Passengers with Anomalies Detected by Isolation Forest")
plt.show()