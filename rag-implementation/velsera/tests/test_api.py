from fastapi.testclient import TestClient
import app                       # imports your FastAPI instance

client = TestClient(app.app)

def test_health_route():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    r = client.post("/predict", json={"text": "BRCA1 mutation…"})
    js = r.json()
    assert r.status_code == 200
    assert js["label"] in ("Cancer", "Non-Cancer")
    assert abs(sum(js["probabilities"].values()) - 1) < 1e-3
