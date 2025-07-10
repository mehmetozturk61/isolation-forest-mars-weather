# 🪐 Mars Weather Anomaly Detection

This project analyzes Martian weather data using the **Isolation Forest algorithm** to detect anomalies in environmental conditions such as temperature and atmospheric pressure. The dataset is based on sensor readings from NASA's Curiosity rover.

## 📊 Dataset

- Source: [Mars Weather Data](https://www.kaggle.com/datasets/mexwell/mars-weather)
- Fields Used:
  - `terrestrial_date`: Earth date
  - `ls`: Solar longitude (Martian season indicator)
  - `min_temp`: Minimum temperature (°C)
  - `max_temp`: Maximum temperature (°C)
  - `pressure`: Atmospheric pressure (Pa)

## 🧪 Objective

The goal of this analysis is to:
- Detect unusual environmental patterns on Mars.
- Visualize anomaly points over time and across multiple feature relationships.

## 📌 Methodology

1. **Data Cleaning**:
   - Dropped missing values from key columns (`min_temp`, `max_temp`, `pressure`).

2. **Modeling**:
   - Applied `IsolationForest` with 1% contamination to detect anomalies.
   - Used temperature and pressure features for anomaly detection.

3. **Visualization**:
   - Scatter plots for variable relationships.
   - Time-series plots with anomalies marked.
   - Pairplots for quick multidimensional overview.

## 📁 File Structure

