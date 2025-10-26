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
#report
# Load monthly PSI JSONs
stability_dir = "datamart/gold/monitoring/stability"
psi_data = {}

for file in os.listdir(stability_dir):
    if file.endswith(".json"):
        month = file.replace("stability_", "").replace(".json", "")
        with open(os.path.join(stability_dir, file)) as f:
            psi_data[month] = json.load(f)

psi_df = pd.DataFrame.from_dict(
    {month: data["average_psi"] for month, data in psi_data.items()},
    orient='index', columns=["Average PSI"]
).sort_index()
psi_df.index.name = "Month"

# Load monthly performance JSONs
performance_dir = "datamart/gold/monitoring/performance"
perf_data = {}

for file in os.listdir(performance_dir):
    if file.endswith(".json"):
        month = file.replace("performance_", "").replace(".json", "")
        with open(os.path.join(performance_dir, file)) as f:
            perf_data[month] = json.load(f)

metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
perf_df = pd.DataFrame.from_dict(
    {month: data["metrics"] for month, data in perf_data.items()},
    orient='index'
)[metrics].sort_index()
perf_df.index.name = "Month"

# Plotting
output_dir = "datamart/gold/monitoring"

# PSI Trend Plot
plt.figure(figsize=(8, 6))
sns.lineplot(data=psi_df, x=psi_df.index, y="Average PSI", marker="o")
plt.title("Stability: Average PSI Mar-Jun 2024", fontsize=16)
plt.ylabel("PSI", fontsize=14)
plt.xlabel("Month", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "psi_trend_mar_to_jun.png"))
plt.show()

# Performance Metrics Plot
plt.figure(figsize=(8, 6))
for metric in metrics:
    sns.lineplot(data=perf_df, x=perf_df.index, y=metric, marker="o", label=metric)
plt.title("Performance Metrics Mar-Jun 2024", fontsize=16)
plt.ylabel("Score", fontsize=14)
plt.xlabel("Month", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_trends_mar_to_jun.png"))