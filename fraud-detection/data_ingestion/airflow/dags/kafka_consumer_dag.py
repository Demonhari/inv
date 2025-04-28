from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from kafka import KafkaConsumer
import json
import os

def consume_transactions():
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='airflow-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    os.makedirs('/tmp/transactions', exist_ok=True)
    with open('/tmp/transactions/data.json', 'a') as f:
        for message in consumer:
            json.dump(message.value, f)
            f.write('\n')
            print(f"Consumed: {message.value}")
            break  # Only one message per task run

default_args = {
    'owner': 'hari',
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='kafka_consume_transactions',
    default_args=default_args,
    description='Consume transaction data from Kafka and store locally',
    start_date=datetime(2024, 4, 27),
    schedule_interval='@once',
    catchup=False
) as dag:
    consume_task = PythonOperator(
        task_id='consume_transactions_task',
        python_callable=consume_transactions
    )

    consume_task
