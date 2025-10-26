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
#xgboost
# Start Spark
spark = SparkSession.builder \
    .appName("train_xgboost") \
    .master("local[*]") \
    .getOrCreate()

# Load prepared labeled data
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

# Time-based split
train_df = scaled.filter((col("feature_snapshot_date") >= "2023-01-01") & (col("feature_snapshot_date") <= "2023-12-01"))
val_df   = scaled.filter(col("feature_snapshot_date") == "2024-01-01")
test_df  = scaled.filter(col("feature_snapshot_date") == "2024-02-01")

# Helper: convert Spark DataFrame to NumPy
def spark_df_to_numpy(df):
    pdf = df.select("features", "label").toPandas()
    X = np.vstack(pdf["features"].values)
    y = pdf["label"].values
    return X, y

X_train, y_train = spark_df_to_numpy(train_df)
X_val, y_val     = spark_df_to_numpy(val_df)
X_test, y_test   = spark_df_to_numpy(test_df)

# Helper: evaluation
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1 Score":  round(f1_score(y_true, y_pred), 4)
    }

# Enhanced Hyperparameter tuning with RandomizedSearchCV
print("Configuring XGBoost model training...")

# Define comprehensive hyperparameter search space
hyperparameter_space = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 3, 4, 5, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}

# Configure RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        eval_metric='logloss', 
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=hyperparameter_space,
    n_iter=20,  # Number of parameter combinations to try
    cv=3,       # 3-fold cross-validation
    scoring='roc_auc',
    verbose=1,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

print(f"Hyperparameter search configuration:")
print(f"Search space combinations: {np.prod([len(v) for v in hyperparameter_space.values()]):,}")
print(f"Random search iterations: {20}")
print(f"Cross-validation folds: {3}")
print(f"Optimization metric: ROC-AUC")

# Perform hyperparameter search
print("Starting hyperparameter search...")
random_search.fit(X_train, y_train)

# Get the best model from randomized search
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f"\nBest hyperparameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"Best cross-validation AUC: {best_cv_score:.4f}")

# Evaluate best model on validation set
y_val_prob = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_prob)
print(f"Validation AUC with best model: {val_auc:.4f}")

# Compare with simple n_estimators tuning (your original approach)
print("\nComparison with simple tuning:")
simple_model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
simple_model.fit(X_train, y_train)
y_val_simple_prob = simple_model.predict_proba(X_val)[:, 1]
simple_auc = roc_auc_score(y_val, y_val_simple_prob)
print(f"Simple tuning (n_estimators=100) Validation AUC: {simple_auc:.4f}")
print(f"Improvement with comprehensive tuning: {val_auc - simple_auc:.4f}")

# Evaluate on all sets with best model
results = {}
for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]
    results[name] = evaluate_model(y, y_pred, y_prob)

# Print & save metrics
metrics_dir = "model_store/xgboost_metrics"
os.makedirs(metrics_dir, exist_ok=True)

for name, metrics in results.items():
    print(f"\n=== {name.capitalize()} Metrics ===")
    print(metrics)

    metrics_path = os.path.join(metrics_dir, f"xgboost_{name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"{name.capitalize()} metrics saved to {metrics_path}")

# Save best parameters and model
best_params_path = os.path.join(metrics_dir, "xgboost_best_params.json")
with open(best_params_path, "w") as f:
    json.dump(best_params, f, indent=2)
print(f"Best parameters saved to {best_params_path}")

# Save model
os.makedirs("model_store", exist_ok=True)
joblib.dump(best_model, "model_store/xgboost_model.pkl")
print("XGBoost model saved to model_store/xgboost_model.pkl")