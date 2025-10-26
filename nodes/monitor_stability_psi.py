from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, date_format
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Start Spark
spark = SparkSession.builder.appName("monitor_psi_features").getOrCreate()

# Load combined labeled data
df = spark.read.parquet("data/gold/combined_labeled.parquet")

# Reference = February 2024
ref_df = df.filter(col("feature_snapshot_date") == "2024-02-01").toPandas()

# Feature columns
non_features = ['Customer_ID', 'feature_snapshot_date', 'loan_id', 'label']
vec_cols = [c for c in ref_df.columns if c.endswith("_vec")]
feature_cols = [c for c in ref_df.columns if c not in non_features + vec_cols]

# PSI calculator
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    psi = np.sum((expected_bins - actual_bins) * np.log((expected_bins + 1e-6) / (actual_bins + 1e-6)))
    return round(psi, 4)

# Loop through OOT months
monitor_months = ["2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
psi_results = {}

for month in monitor_months:
    current_df = df.filter(col("feature_snapshot_date") == month).toPandas()
    psi_scores = {}

    for feat in feature_cols:
        try:
            psi_scores[feat] = calculate_psi(ref_df[feat].dropna(), current_df[feat].dropna())
        except Exception:
            psi_scores[feat] = None

    # Valid PSI scores only
    valid_psi = {k: v for k, v in psi_scores.items() if v is not None}
    avg_psi = round(np.mean(list(valid_psi.values())), 4)
    max_psi = round(max(valid_psi.values()), 4)

    # Alerting
    drifted_feats = [k for k, v in valid_psi.items() if v > 0.1]

    if avg_psi > 0.25:
        alert = "Investigate & consider retraining (high average PSI)"
        print(f"\n{month} — {alert}")
        print(f"Avg PSI: {avg_psi}")
        print("Drifted features (PSI > 0.1):", drifted_feats)

    elif avg_psi > 0.1:
        alert = "Monitor more closely (elevated average PSI)"
        print(f"\n{month} — {alert}")
        print(f"Avg PSI: {avg_psi}")
        print("Drifted features (PSI > 0.1):", drifted_feats)

    elif max_psi > 0.1:
        alert = "Monitor individual features (some drift)"
        print(f"\n{month} — {alert}")
        print(f"Max PSI: {max_psi}")
        print("Drifted features (PSI > 0.1):", drifted_feats)

    else:
        alert = "All clear (PSI stable)"
        print(f"\n{month} — {alert}")

    psi_results[month] = {
        "average_psi": avg_psi,
        "max_psi": max_psi,
        "drifted_features": drifted_feats,
        "alert": alert,
        "feature_psi": valid_psi
    }

# Save one JSON file per month
output_dir = "datamart/gold/monitoring/stability"
os.makedirs(output_dir, exist_ok=True)

for month, result in psi_results.items():
    filename = f"{output_dir}/stability_{month}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

print("\nStability (PSI) monitoring complete.")