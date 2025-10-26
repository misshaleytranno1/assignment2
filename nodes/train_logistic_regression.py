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
#logistic_regression
# Start Spark
spark = SparkSession.builder \
    .appName("train_logistic_regression") \
    .master("local[*]") \
    .getOrCreate()

# Load prepared labeled dataset
df = spark.read.parquet("data/gold/combined_labeled.parquet")

# Assemble and scale features
non_features = ['Customer_ID', 'feature_snapshot_date', 'loan_id', 'label']
vec_cols = [c for c in df.columns if c.endswith("_vec")]
feature_cols = [c for c in df.columns if c not in non_features + vec_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled)
scaled = scaler_model.transform(assembled)

# Split by time
train_df = scaled.filter((col("feature_snapshot_date") >= "2023-01-01") & (col("feature_snapshot_date") <= "2023-12-01"))
val_df   = scaled.filter(col("feature_snapshot_date") == "2024-01-01")
test_df  = scaled.filter(col("feature_snapshot_date") == "2024-02-01")

# Train logistic regression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
lr_model = lr.fit(train_df)

# Evaluation function
def evaluate_model(predictions):
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    preds = predictions.withColumn("correct", when(col("prediction") == col("label"), 1).otherwise(0))
    total = preds.count()
    correct = preds.filter(col("correct") == 1).count()
    tp = preds.filter((col("prediction") == 1) & (col("label") == 1)).count()
    fp = preds.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = preds.filter((col("prediction") == 0) & (col("label") == 1)).count()

    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "AUC": round(auc, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    }

# Create folder for metrics
metrics_dir = "model_store/logreg_metrics"
os.makedirs(metrics_dir, exist_ok=True)

# Evaluate on splits
for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    pred = lr_model.transform(split)
    metrics = evaluate_model(pred)
    print(f"\n=== {name.capitalize()} Metrics ===")
    print(metrics)
    
    # Save to corresponding JSON
    metrics_path = os.path.join(metrics_dir, f"logreg_{name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"{name.capitalize()} metrics saved to {metrics_path}")

# Save model (if needed later in selection step)
model_path = "model_store/logistic_model"
if os.path.exists(model_path):
    import shutil
    shutil.rmtree(model_path)
lr_model.save(model_path)
print("\nLogistic regression model saved.")