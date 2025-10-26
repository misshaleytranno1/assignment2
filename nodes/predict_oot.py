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
spark = SparkSession.builder \
    .appName("predict_oot") \
    .master("local[*]") \
    .getOrCreate()
#load best model
# Load transformed + scaled dataset
df = spark.read.parquet("data/gold/combined_labeled.parquet")

# Reconstruct raw_features and scale again (to match training)
from pyspark.ml.feature import VectorAssembler, StandardScaler

non_features = ['Customer_ID', 'feature_snapshot_date', 'loan_id', 'label']
vec_cols = [c for c in df.columns if c.endswith("_vec")]
feature_cols = [c for c in df.columns if c not in non_features + vec_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled)
scaled = scaler_model.transform(assembled)

# Load best model 
model = joblib.load("model_store/best_model.pkl")

# Helper to convert Spark → NumPy
def spark_df_to_numpy(df):
    pdf = df.select("loan_id", "features").toPandas()
    loan_ids = pdf["loan_id"].values
    X = np.vstack(pdf["features"].values)
    return loan_ids, X

# Loop over OOT months
oot_months = ["2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]

for month in oot_months:
    df_month = scaled.filter(col("feature_snapshot_date") == month)
    loan_ids, X = spark_df_to_numpy(df_month)

    # Predict probabilities and labels
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Create output DataFrame
    pdf_out = {
        "loan_id": loan_ids,
        "prediction": y_pred,
        "probability": y_prob,
        "snapshot_date": [month] * len(loan_ids)
    }

    import pandas as pd
    spark_out = spark.createDataFrame(pd.DataFrame(pdf_out))

    # Save to gold datamart
    output_path = f"datamart/gold/predictions/predictions_oot_{month}.parquet"
    spark_out.write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved predictions for {month} to {output_path}")
