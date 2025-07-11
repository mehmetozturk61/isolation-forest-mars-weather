import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

df = pd.read_csv('2015_Financial_Data.csv')
df.head()
df.info()
df.describe()

df_model = df.drop(columns=["Sector", "PRICE VAR [%]", "class"], errors="ignore")
df_model = df_model.select_dtypes(include=["float64", "int64"])
df_model = df_model.fillna(df_model.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_scaled)

df['anomaly_score'] = iso.decision_function(X_scaled)
df['anomaly'] = iso.predict(X_scaled)

# View the top anomalies
anomalies = df[df['anomaly'] == -1]
print(f"Total anomalies detected: {len(anomalies)}")
print(anomalies.sort_values('anomaly_score').head())

""" print(classification_report(df['class'])) """

""" pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['anomaly'], cmap='coolwarm', s=5)
plt.title("Isolation Forest Anomaly Detection (PCA-Reduced)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Anomaly (-1=Anomaly, 1=Normal)')
plt.show() """