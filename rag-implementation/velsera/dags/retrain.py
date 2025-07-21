"""
retrain_dag.py  – nightly 01:00 UTC
 1. Re-runs preprocess
 2. Fine-tunes LoRA for 3 epochs
 3. Evaluates → pushes metrics.json artefact
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "velsera",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}
with DAG(
    dag_id="velsera_nightly_retrain",
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 1 * * *",   # every night 01:00 UTC
    catchup=False,
    default_args=default_args,
):
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /opt/velsera/preprocess.py",
    )

    lora_train = BashOperator(
        task_id="train_lora",
        bash_command="python /opt/velsera/train_lora.py --epochs 3",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python /opt/velsera/evaluate.py",
    )

    def upload_metrics():
        # dummy – replace with S3 / MinIO / MLflow client
        import shutil, pathlib, datetime as dt
        ts   = dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
        dest = pathlib.Path("/opt/velsera/metrics_archive") / f"metrics_{ts}.json"
        shutil.copy("output/metrics.json", dest)

    push_metrics = PythonOperator(
        task_id="archive_metrics",
        python_callable=upload_metrics,
    )

    preprocess >> lora_train >> evaluate >> push_metrics
