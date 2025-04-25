from kafka import KafkaProducer
import json
from faker import Faker
import random
import time

fake = Faker()

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
    return {
        "transaction_id": fake.uuid4(),
        "user_id": fake.uuid4(),
        "timestamp": fake.iso8601(),
        "amount": round(random.uniform(10.0, 1000.0), 2),
        "location": fake.city(),
        "card_type": random.choice(["VISA", "MASTERCARD", "AMEX"]),
        "merchant": fake.company(),
        "fraud": random.choice([0, 1])
    }

while True:
    txn = generate_transaction()
    producer.send("transactions", txn)
    print(f"Sent: {txn}")
    time.sleep(1)
