import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import traceback
import os

# === Configure Logging ===
log_path = "/opt/airflow/logs/pipeline.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 6, 1),
}

# === Helper to run node scripts ===
def safe_run_script(script_name):
    try:
        logging.info(f"Running: {script_name}")
        subprocess.run(["python", f"/opt/airflow/nodes/{script_name}"], check=True)
        logging.info(f"Success: {script_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Script failed: {script_name}")
        logging.error(e)
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"Unexpected error in {script_name}: {e}")
        logging.error(traceback.format_exc())
        raise

# === DAG Definition ===
with DAG("dag",
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    feature_pipeline = PythonOperator(
        task_id="feature_pipeline",
        python_callable=lambda: safe_run_script("setup.py")
    )

    train_logistic_regression = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=lambda: safe_run_script("train_logistic_regression.py")
    )

    train_xgboost = PythonOperator(
        task_id="train_xgboost",
        python_callable=lambda: safe_run_script("train_xgboost.py")
    )

    select_best_model = PythonOperator(
        task_id="select_best_model",
        python_callable=lambda: safe_run_script("select_best_model.py")
    )

    inference_pipeline = PythonOperator(
        task_id="inference_pipeline",
        python_callable=lambda: safe_run_script("predict_oot.py")
    )

    monitor_stability_psi = PythonOperator(
        task_id="monitor_stability_psi",
        python_callable=lambda: safe_run_script("monitor_stability_psi.py")
    )

    monitor_metrics = PythonOperator(
        task_id="monitor_metrics",
        python_callable=lambda: safe_run_script("monitor_performance_metrics.py")
    )

    visual = PythonOperator(
        task_id="visual",
        python_callable=lambda: safe_run_script("visualise_performance_and_stability.py")
    )
    
    # DAG Dependencies
    feature_pipeline >> [train_logistic_regression, train_xgboost] >> select_best_model >> inference_pipeline >> monitor_stability_psi >> monitor_metrics >> visual