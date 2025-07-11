import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("insurance.csv")

df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_scaled)

df_encoded['anomaly_score'] = iso.decision_function(X_scaled)
df_encoded['anomaly'] = iso.predict(X_scaled)

df['anomaly'] = df_encoded['anomaly']
df['anomaly_score'] = df_encoded['anomaly_score']

""" print("Number of anomalies:", (df['anomaly'] == -1).sum())
print("Number of normal points:", (df['anomaly'] == 1).sum())

df.to_csv("insurance_with_anomalies.csv", index=False) """

# Charges Distribution by Anomaly
""" sns.boxplot(x='anomaly', y='charges', data=df)
plt.title("Charges Distribution: Normal (1) vs Anomalies (-1)")
plt.show() """

# BMI vs Charges — Highlight Anomalies
""" plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', hue='anomaly', data=df, palette={1: 'blue', -1: 'red'})
plt.title("BMI vs Charges with Anomaly Labels")
plt.show() """

normal = df[df['anomaly'] == 1]
anomaly = df[df['anomaly'] == -1]

""" features_to_compare = ['age', 'bmi', 'children', 'charges']
summary_normal = normal[features_to_compare].describe().T
summary_anomaly = anomaly[features_to_compare].describe().T

print("=== Normal Group Summary ===")
print(summary_normal)
print("\n=== Anomaly Group Summary ===")
print(summary_anomaly) """

""" categorical_cols = ['sex', 'smoker', 'region']

for col in categorical_cols:
    print(f"=== {col.upper()} ===")
    print("\nNormal Group:")
    print(normal[col].value_counts(normalize=True).round(3))
    
    print("\nAnomaly Group:")
    print(anomaly[col].value_counts(normalize=True).round(3))
    
    print("\n" + "-"*40 + "\n") """

""" # Select only numerical columns
numerical_cols = ['age', 'bmi', 'children', 'charges']

# Compute correlation matrices
corr_normal = normal[numerical_cols].corr()
corr_anomaly = anomaly[numerical_cols].corr()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(corr_normal, annot=True, cmap='coolwarm', ax=axes[0])
axes[0].set_title('Normal Group Correlation')

sns.heatmap(corr_anomaly, annot=True, cmap='coolwarm', ax=axes[1])
axes[1].set_title('Anomaly Group Correlation')

plt.tight_layout()
plt.show() """