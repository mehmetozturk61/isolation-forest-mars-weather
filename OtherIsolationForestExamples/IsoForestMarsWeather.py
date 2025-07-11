import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv("mars-weather.csv")

df.set_index("id", inplace=True)
df["terrestrial_date"] = pd.to_datetime(df["terrestrial_date"])

# print(df.isnull().sum())

df = df.dropna(subset=["min_temp", "max_temp", "pressure"])
features = ["ls", "min_temp", "max_temp", "pressure"]

X = df[features].copy()

iso = IsolationForest(contamination=0.01, random_state=42)
df["anomaly"] = iso.fit_predict(X)

normals = df[df["anomaly"] == 1]
anomalies = df[df["anomaly"] == -1]

# Min Temp vs Max Temp
""" plt.scatter(normals["min_temp"], normals["max_temp"], c="blue")
plt.scatter(anomalies["min_temp"], anomalies["max_temp"], c="red")
plt.xlabel("Min Temp (°C)")
plt.ylabel("Max Temp (°C)")
plt.title("Min Temp vs Max Temp")
plt.show() """

# Min Temp vs Pressure
""" plt.scatter(normals["min_temp"], normals["pressure"], c="blue")
plt.scatter(anomalies["min_temp"], anomalies["pressure"], c="red")
plt.xlabel("Min Temp (°C)")
plt.ylabel("Pressure (Pa)")
plt.title("Min Temp vs Pressure")
plt.show() """

# Max Temp vs Pressure
""" plt.scatter(normals["max_temp"], normals["pressure"], c="blue")
plt.scatter(anomalies["max_temp"], anomalies["pressure"], c="red")
plt.xlabel("Max Temp (°C)")
plt.ylabel("Pressure (Pa)")
plt.title("Max Temp vs Pressure")
plt.show() """

# Min Temp / Max Temp / Pressure vs Date
""" fig, axes = plt.subplots(3, 1, figsize=(18, 30), sharex=True)

axes[0].plot(df["terrestrial_date"], df["min_temp"])
axes[0].scatter(anomalies["terrestrial_date"], anomalies["min_temp"], color='red')
axes[0].set_title("Minimum Temperature (°C)")
axes[0].set_ylabel("°C")

axes[1].plot(df["terrestrial_date"], df["max_temp"])
axes[1].scatter(anomalies["terrestrial_date"], anomalies["max_temp"], color='red')
axes[1].set_title("Maximum Temperature (°C)")
axes[1].set_ylabel("°C")

axes[2].plot(df["terrestrial_date"], df["pressure"])
axes[2].scatter(anomalies["terrestrial_date"], anomalies["pressure"], color='red')
axes[2].set_title("Atmospheric Pressure (Pa)")
axes[2].set_ylabel("Pa")

for ax in axes:
    ax.set_xlabel("Date")
    ax.tick_params(axis='x')

plt.show() """

""" sns.pairplot(df, vars=['ls', 'min_temp', 'max_temp', 'pressure'], hue='anomaly', palette="Set1")
plt.show() """

sns.pairplot(anomalies, vars=['ls', 'min_temp', 'max_temp', 'pressure'])
plt.suptitle("Clusters of Anomalies")
plt.show()