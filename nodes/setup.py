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
#inference pipeline
# Start Spark
spark = SparkSession.builder \
    .appName("setup") \
    .master("local[*]") \
    .getOrCreate()

# Load gold layer
features = spark.read.parquet("data/gold/feature_store")
labels   = spark.read.parquet("data/gold/label_store")

# Construct loan_id in features
features = features.withColumn(
    "loan_id",
    concat_ws("_", col("Customer_ID"), date_format(col("feature_snapshot_date"), "yyyy_MM_dd"))
)

# Join labels
labeled_df = features.join(
    labels.select("loan_id", "label"),
    on="loan_id",
    how="left"
)

# Filter labeled only
combined_labeled = labeled_df.filter(col("label").isNotNull())

# Save to datamart
combined_labeled.write.mode("overwrite").parquet("data/gold/combined_labeled.parquet")

print("Setup complete: combined_labeled.parquet written.")