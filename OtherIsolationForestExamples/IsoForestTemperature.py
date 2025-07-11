""" import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("machine_temperature_system_failure.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

X = df[["value"]]
X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(contamination=0.01, random_state=1)
df["anomaly"] = model.fit_predict(X_scaled)

plt.figure(figsize=(14,6))
plt.plot(df.index, df["value"], label="Temperature")
plt.scatter(df[df["anomaly"] == -1].index, df[df["anomaly"] == -1]["value"], color="red", label="Anomaly")
plt.legend()
plt.title("Machine Temperature with Anomalies Detected by Isolation Forest")
plt.show() """

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Create sample data: mostly normal points clustered, with a few outliers
rng = np.random.RandomState(42)
normal_data = 0.3 * rng.randn(100, 2)
outliers = rng.uniform(low=-4, high=4, size=(10, 2))
X = np.r_[normal_data + 2, normal_data - 2, outliers]

df = pd.DataFrame(X, columns=['x', 'y'])

# Fit Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = clf.fit_predict(df)

# Plot results
plt.scatter(df['x'], df['y'], c=df['anomaly'], cmap='coolwarm')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('x')
plt.ylabel('y')
plt.show()