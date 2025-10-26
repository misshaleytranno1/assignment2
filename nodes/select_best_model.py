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
#bets_model_chosen
# Load test AUCs from saved metrics
with open("model_store/logreg_metrics/logreg_test.json") as f:
    logreg_metrics = json.load(f)
with open("model_store/xgboost_metrics/xgboost_test.json") as f:
    xgb_metrics = json.load(f)

logreg_auc = logreg_metrics["AUC"]
xgb_auc = xgb_metrics["AUC"]

print("Validation AUCs:")
print(f"Logistic Regression: {logreg_auc}")
print(f"XGBoost: {xgb_auc}")

# Select best model with 3% threshold
if xgb_auc > logreg_auc and ((xgb_auc - logreg_auc) / logreg_auc) > 0.03:
    best_model = joblib.load("model_store/xgboost_model.pkl")
    joblib.dump(best_model, "model_store/best_model.pkl")
    print("\nXGBoost selected as best model and saved as best_model.pkl")
else:
    print("\nLogistic Regression has better AUC but cannot be saved with joblib.")
    print("Use PySpark's .load() from model_store/logistic_model when needed.")