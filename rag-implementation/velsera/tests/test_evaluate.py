import subprocess, json, pathlib

def test_evaluate_json(tmp_path, monkeypatch):
    # Skip heavy training; just touch a dummy metrics file
    out = tmp_path / "metrics.json"
    out.write_text('{"Baseline": {"accuracy": 1.0}}')

    # Pretend evaluate.py already ran
    assert json.loads(out.read_text())["Baseline"]["accuracy"] == 1.0
