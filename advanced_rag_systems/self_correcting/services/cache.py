# services/cache.py
import redis
import json
import os

class Cache:
    def __init__(self):
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str):
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set(self, key: str, value: dict, ttl=3600):
        self.client.set(key, json.dumps(value), ex=ttl)
