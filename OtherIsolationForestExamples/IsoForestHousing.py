import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest

df = pd.read_csv("housing.csv")
df.dropna(inplace=True)
dfv2 = pd.get_dummies(df, columns=["ocean_proximity"])

dfv2["rooms_per_household"] = dfv2["total_rooms"] / dfv2["households"]
dfv2["bedrooms_per_room"] = dfv2["total_bedrooms"] / dfv2["total_rooms"]
dfv2["population_per_household"] = dfv2["population"] / dfv2["households"]

X = StandardScaler().fit_transform(dfv2)

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X)

dfv2["anomaly_score"] = iso.decision_function(X)
dfv2["anomaly"] = iso.predict(X)

print("Number of anomalies:", (dfv2['anomaly'] == -1).sum())
print("Number of normal points:", (dfv2['anomaly'] == 1).sum())

normal = dfv2[dfv2['anomaly'] == 1]
anomaly = dfv2[dfv2['anomaly'] == -1]

plt.figure(figsize=(10, 6))
sns.boxplot(x="anomaly", y="median_house_value", data=dfv2)
plt.xticks([0, 1], ["Anomalies", "Normal"])
plt.title("Median House Value by Anomaly Status")
plt.ylabel("Median House Value")
plt.xlabel("Data Type")
plt.show()