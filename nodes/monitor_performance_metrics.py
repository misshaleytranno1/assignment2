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

# Load baseline test metrics (from model_store)
with open("model_store/xgboost_metrics/xgboost_test.json") as f:
    baseline = json.load(f)

print("\nBaseline (Test Set / February 2024) Metrics")
print(baseline)

# Load labels
labels_df = pd.read_parquet("data/gold/combined_labeled.parquet")[["loan_id", "label"]]

# Evaluation function
def evaluate(y_true, y_pred, y_prob):
    return {
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1 Score":  round(f1_score(y_true, y_pred), 4)
    }

# Percentage change calculator
def percent_change(current, reference):
    return round(((current - reference) / reference) * 100, 2)

# Store results and alerts
results = {}

for month in ["2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]:
    path = f"datamart/gold/predictions/predictions_oot_{month}.parquet"
    if os.path.exists(path):
        preds_df = pd.read_parquet(path)
        merged = preds_df.merge(labels_df, on="loan_id", how="left")

        if merged["label"].isnull().any():
            print(f"Warning: Missing labels in {month} predictions")

        y_true = merged["label"].values
        y_pred = merged["prediction"].values
        y_prob = merged["probability"].values

        # Evaluate metrics
        current_metrics = evaluate(y_true, y_pred, y_prob)
        print(f"\n=== {month} Metrics ===")
        print(current_metrics)

        # Compare against baseline
        changes = {k: percent_change(current_metrics[k], baseline[k]) for k in baseline}
        dropped = {k: v for k, v in changes.items() if v < -5}
        severe_drops = {k: v for k, v in changes.items() if v < -10}

        # Determine alert level
        if severe_drops:
            alert = "Significant performance drop"
        elif dropped:
            alert = "Moderate drop, monitor"
        else:
            alert = "All clear"

        print(f"% Change vs Test Set: {changes}")
        print(f"Alert: {alert}")
        if dropped:
            print("Metrics with drops:", dropped)

        # Store
        results[month] = {
            "metrics": current_metrics,
            "percentage_change": changes,
            "dropped_metrics": dropped,
            "alert": alert
        }

# Save one JSON file per month
output_dir = "datamart/gold/monitoring/performance"
os.makedirs(output_dir, exist_ok=True)

for month, result in results.items():
    filename = f"{output_dir}/performance_{month}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

print("\nPerformance monitoring complete.")