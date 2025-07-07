from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from kafka import KafkaConsumer
import json
import os
import pandas as pd

def consume_transactions(**context):
    # 1) connect to Kafka
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=['kafka:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='airflow-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    # 2) pull 10 records
    records = []
    for _ in range(10):
        msg = next(consumer)
        records.append(msg.value)

    # 3) write to JSON‐lines
    os.makedirs('/tmp/transactions', exist_ok=True)
    path = '/tmp/transactions/data.json'
    pd.DataFrame(records).to_json(path, orient='records', lines=True)

    # 4) XCom the path for the next task
    context['ti'].xcom_push(key='data_path', value=path)
    print(f"Consumed {len(records)} messages → {path}")


def validate_transactions(**context):
    ti = context['ti']
    path = ti.xcom_pull(task_ids='consume_transactions', key='data_path')

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"No data file found at {path}")

    df = pd.read_json(path, lines=True)

    # your asserts
    assert 'transaction_id' in df.columns, "Missing transaction_id column"
    assert df['amount'].gt(0).all(), "Non-positive amount found"
    assert df['fraud'].isin([0, 1]).all(), "Invalid fraud flag"

    print(f"✅ Validation passed for {len(df)} records")


default_args = {
    'owner': 'hari',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='kafka_consume_and_validate',
    default_args=default_args,
    start_date=datetime(2025, 4, 28),
    schedule_interval='@once',
    catchup=False,
    tags=['kafka', 'validation'],
) as dag:

    consume_task = PythonOperator(
        task_id='consume_transactions',
        python_callable=consume_transactions,
    )

    validate_task = PythonOperator(
        task_id='validate_transactions',
        python_callable=validate_transactions,
    )

    consume_task >> validate_task
